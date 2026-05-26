"""Run a set of baselines over the dockermin holdout, write EvalEntry JSON lines.

Usage:
    python scripts/run_eval.py \
        --baselines qwen_zs gpt4o sonnet_zs hadolint slim manual agent_loop dockermin \
        --out data/eval/results.jsonl

The holdout is the ``test`` split of ``vtemian/dockermin-v0``. If the
dataset has no ``test`` split we fail loudly rather than silently evaluating
on training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Importing baselines registers qwen/gpt4o/sonnet/hadolint/slim/manual/dockermin.
# Importing agent_loop registers the agent_loop baseline via side effect.
from dockermin.eval import agent_loop as _agent_loop  # noqa: F401  registration side-effect
from dockermin.eval.baselines import (
    EvalEntry,
    available_baselines,
    run_one,
)


def _load_holdout(repo_id: str) -> list[dict]:
    """Return the triple dicts of the disjoint ``test`` split.

    There is NO fallback to training rows: evaluating on data the model
    trained on would silently inflate the result. If the dataset has no
    ``test`` split we raise so the operator fixes the dataset push instead.
    """
    ds = load_dataset(repo_id)
    if "test" not in ds:
        msg = (
            f"{repo_id} has no 'test' split (found: {sorted(ds.keys())}). "
            "Refusing to evaluate on training data - push a grouped train/test "
            "split first (see scripts/push_to_hf.py)."
        )
        raise SystemExit(msg)
    return list(ds["test"])


# Retry policy lives inside the API-only baselines (gpt4o, sonnet_zs) where
# the only safely-retryable failures occur. Wrapping run_one would also retry
# expensive docker rebuilds on transient unix-socket hiccups, which we don't want.


def main() -> int:
    p = argparse.ArgumentParser(description="Run dockermin baselines over the holdout set.")
    p.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help=f"Baseline names. Available: {' '.join([*available_baselines(), 'agent_loop'])}",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path. Parent dirs created. Appends if file exists.",
    )
    p.add_argument(
        "--holdout",
        default="vtemian/dockermin-v0",
        help="HF dataset id. Default: vtemian/dockermin-v0",
    )
    p.add_argument(
        "--dockermin-model",
        default="vtemian/dockermin-qwen7b-lora-v1",
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
                    entry = run_one(baseline, triple, **kwargs)
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
