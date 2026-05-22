"""Generate synthetic 'deliberately unoptimized' Dockerfile variants via Claude.

Uses the locally-installed `claude` CLI (Claude Code) in headless print mode
rather than the anthropic SDK so we do not need ANTHROPIC_API_KEY in the
environment - Claude Code handles auth via its own keychain/login.

Strategy: take a small set of working baseline Dockerfiles (the curated
triples that passed parse + build + test), and have Claude Sonnet 4.6 produce
N unoptimized variants per base. Each variant:
  - keeps the same test_cmd + expected_substring (functional equivalence)
  - introduces realistic bloat the model can learn to undo:
    (a) heavier base image (e.g. python:3.12 instead of python:3.12-slim)
    (b) redundant RUN layers (multiple apt-get install lines)
    (c) missing --no-install-recommends / --no-cache-dir
    (d) leftover dev deps
    (e) shell-form CMD/ENTRYPOINT instead of JSON-form

Each variant validated via annotate_one (must build + pass same test).
Reward signal during training: rewrite the variant -> smaller working image.

Cost: ~$0.02 per variant via Sonnet 4.6. 5 variants/base x 20 bases ~$2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from dockermin.dataset.annotate import annotate_one
from dockermin.reward.prompts import extract_dockerfile


SYSTEM_PROMPT = (
    "You are generating training data for a Dockerfile-shrinking reinforcement-"
    "learning project. Given a clean Dockerfile, produce a DELIBERATELY "
    "UNOPTIMIZED but FUNCTIONALLY EQUIVALENT variant. Realistic bloat patterns "
    "preferred: bigger base image (python:3.12 not python:3.12-slim), extra "
    "RUN layers, missing --no-install-recommends, leftover dev deps, shell-form "
    "CMD. The variant MUST still build cleanly and pass the same test command "
    "with the same expected output substring. Output ONLY the variant "
    "Dockerfile in a single fenced ```dockerfile block. No explanation."
)


def generate_variant(baseline_df: str, test_cmd: list[str], expected: str,
                     variant_seed: int, budget_usd: float = 0.20,
                     timeout_s: int = 120) -> str | None:
    """Drive `claude -p` headless to produce a single variant Dockerfile.

    Cost note: Claude Code's default system prompt + tool defs trigger
    ~25k cache_creation tokens per session start. Budget per call ~$0.10.
    --bare would strip this but requires ANTHROPIC_API_KEY in env which
    we don't have, so we pay the cache cost.
    """
    user_msg = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Baseline Dockerfile:\n```dockerfile\n{baseline_df}\n```\n\n"
        f"Test cmd: {' '.join(test_cmd)}\n"
        f"Expected substring: {expected!r}\n\n"
        f"Generate variant #{variant_seed} (try a different bloat pattern "
        f"than other variants of this base). Output ONLY the variant in a "
        f"single fenced ```dockerfile block."
    )
    cmd = [
        "claude", "-p", user_msg,
        "--output-format", "json",
        "--model", "claude-sonnet-4-6",
        "--max-budget-usd", str(budget_usd),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    text = meta.get("result", "") or ""
    return extract_dockerfile(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="data/curated/triples.jsonl",
                   help="Path to baseline curated triples.")
    p.add_argument("--out", default="data/curated/triples_with_variants.jsonl")
    p.add_argument("--variants-per-base", type=int, default=5)
    p.add_argument("--max-bases", type=int, default=20,
                   help="Cap number of bases (variants_per_base x max_bases = total).")
    args = p.parse_args()

    bases = [json.loads(line) for line in Path(args.in_path).read_text().splitlines()
             if line.strip()]
    bases = bases[: args.max_bases]
    print(f"Generating variants for {len(bases)} baselines, "
          f"{args.variants_per_base} variants each")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-mode + dedup on startup so repeated invocations (or external
    # respawns) accumulate progress instead of truncating prior variants.
    existing_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                existing_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"resuming with {len(existing_ids)} existing entries")
    kept = len(existing_ids)
    failed = 0
    with out_path.open("a") as fout:
        for base in bases:
            if base["id"] not in existing_ids:
                fout.write(json.dumps(base) + "\n"); fout.flush()
                existing_ids.add(base["id"])
                kept += 1
            for vi in range(args.variants_per_base):
                variant_df = generate_variant(
                    base["dockerfile"], base["test_cmd"],
                    base["expected_substring"], vi)
                if variant_df is None:
                    print(f"  variant {vi}: claude returned no dockerfile or errored")
                    failed += 1
                    continue
                # Validate variant builds + tests
                res = annotate_one(variant_df, base["test_cmd"],
                                   base["expected_substring"],
                                   build_timeout_s=300, test_timeout_s=30)
                if not res.ok:
                    print(f"  variant {vi} failed annotate: {res.error[:200]}")
                    failed += 1
                    continue
                vid = hashlib.sha256(variant_df.encode()).hexdigest()[:12]
                rec_id = f"{base.get('id', 'unknown')}-v{vi}-{vid}"
                if rec_id in existing_ids:
                    print(f"  variant {vi}: already in output, skipping")
                    continue
                rec = {
                    "id": rec_id,
                    "dockerfile": variant_df,
                    "test_cmd": base["test_cmd"],
                    "expected_substring": base["expected_substring"],
                    "baseline_size": res.baseline_size,
                    "baseline_build_s": res.baseline_build_s,
                    "ecosystem": base.get("ecosystem", "unknown"),
                    "source_url": base.get("source_url", ""),
                    "license": base.get("license"),
                    "is_synthetic_variant": True,
                    "base_id": base.get("id"),
                }
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                existing_ids.add(rec_id)
                kept += 1
                print(f"  variant {vi} kept ({res.baseline_size // 1_000_000} MB)")
                time.sleep(0.5)  # gentle on api
    print(f"done: kept {kept}, failed {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
