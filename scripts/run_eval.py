"""Run a set of baselines over the dockermin holdout, write EvalEntry JSON lines.

Usage:
    python scripts/run_eval.py \
        --baselines qwen_zs gpt4o sonnet_zs hadolint slim manual agent_loop dockermin \
        --out data/eval/results.jsonl

The holdout defaults to the ``test`` split of ``vladtemian/dockermin-v0``. If the
dataset has no explicit test split, the last 150 rows of ``train`` are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# Importing baselines registers qwen/gpt4o/sonnet/hadolint/slim/manual/dockermin.
# Importing agent_loop registers the agent_loop baseline via side effect.
from dockermin.eval import agent_loop as _agent_loop  # noqa: F401  registration side-effect
from dockermin.eval.baselines import (
    EvalEntry,
    available_baselines,
    run_one,
)

_HOLDOUT_TAIL = 150


def _load_holdout(repo_id: str) -> list[dict]:
    """Return a list of triple dicts. Prefer an explicit ``test`` split; else
    fall back to the last ``_HOLDOUT_TAIL`` rows of ``train``."""
    ds = load_dataset(repo_id)
    if "test" in ds:
        return list(ds["test"])
    train = ds["train"]
    n = len(train)
    start = max(0, n - _HOLDOUT_TAIL)
    return [train[i] for i in range(start, n)]


# Tenacity policy: retry on transient network/api errors. We deliberately do NOT
# retry on docker build/test failures (those are real signal).
_TRANSIENT = (ConnectionError, TimeoutError, OSError)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_TRANSIENT),
)
def _run_one_with_retry(baseline: str, triple: dict, **kwargs) -> EvalEntry:
    return run_one(baseline, triple, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description="Run dockermin baselines over the holdout set.")
    p.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help=f"Baseline names. Available: {' '.join(available_baselines() + ['agent_loop'])}",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path. Parent dirs created. Appends if file exists.",
    )
    p.add_argument(
        "--holdout",
        default="vladtemian/dockermin-v0",
        help="HF dataset id. Default: vladtemian/dockermin-v0",
    )
    p.add_argument(
        "--dockermin-model",
        default="vladtemian/dockermin-qwen7b-lora-v1",
        help="HF model id for the dockermin LoRA baseline.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of triples (smoke testing).",
    )
    args = p.parse_args()

    triples = _load_holdout(args.holdout)
    if args.limit is not None:
        triples = triples[: args.limit]
    print(f"loaded {len(triples)} triples from {args.holdout}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = len(args.baselines) * len(triples)
    written = 0
    with args.out.open("a") as fout, tqdm(total=total, desc="eval", unit="run") as pbar:
        for baseline in args.baselines:
            for triple in triples:
                kwargs: dict = {}
                if baseline == "dockermin":
                    kwargs["model_id"] = args.dockermin_model
                try:
                    entry = _run_one_with_retry(baseline, triple, **kwargs)
                except Exception as e:  # last-resort: never abort the whole run
                    entry = EvalEntry(
                        baseline=baseline,
                        triple_id=str(triple.get("id") or triple.get("source_url") or "unknown"),
                        new_size_bytes=None,
                        test_passes=False,
                        elapsed_s=0.0,
                        error=f"unhandled: {e!r}",
                    )
                fout.write(json.dumps(entry.to_dict()) + "\n")
                fout.flush()
                written += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{baseline} ok={entry.test_passes}")

    print(f"wrote {written} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
