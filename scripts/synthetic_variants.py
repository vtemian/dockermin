"""Generate synthetic 'deliberately unoptimized' Dockerfile variants via Claude.

Strategy: take a small set of working baseline Dockerfiles (the curated
triples that passed parse + build + test), and have Claude Sonnet 4.6 produce
N unoptimized variants per base. Each variant should:
  - keep the same test_cmd + expected_substring (so we can verify equivalence)
  - introduce realistic bloat patterns the model can learn to undo:
    (a) heavier base image (e.g. python:3.12 instead of python:3.12-slim)
    (b) redundant RUN layers (multiple apt-get install lines)
    (c) missing --no-install-recommends / --no-cache-dir
    (d) leftover dev deps
    (e) verbose CMD/ENTRYPOINT shell-form instead of JSON-form

Each variant is validated via annotate_one (must build + pass test).
Reward signal during training: rewrite the variant -> smaller working image.

Cost: ~$0.02 per variant via Sonnet 4.6 with prompt caching. 5 variants per
base x 20 bases = 100 variants ~ $2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import anthropic

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


def generate_variant(client: anthropic.Anthropic, baseline_df: str,
                     test_cmd: list[str], expected: str,
                     variant_seed: int) -> str | None:
    user_msg = (
        f"Baseline Dockerfile:\n```dockerfile\n{baseline_df}\n```\n\n"
        f"Test cmd: {' '.join(test_cmd)}\n"
        f"Expected substring: {expected!r}\n\n"
        f"Generate variant #{variant_seed} (try a different bloat pattern "
        f"than other variants of this base)."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    return extract_dockerfile(resp.content[0].text)


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

    client = anthropic.Anthropic()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    failed = 0
    with out_path.open("w") as fout:
        for base in bases:
            # Preserve the baseline triple itself
            fout.write(json.dumps(base) + "\n"); fout.flush()
            kept += 1
            for vi in range(args.variants_per_base):
                try:
                    variant_df = generate_variant(
                        client, base["dockerfile"],
                        base["test_cmd"], base["expected_substring"], vi)
                except anthropic.APIError as e:
                    print(f"  api error: {e!r}"); failed += 1
                    continue
                if variant_df is None:
                    print("  no fenced dockerfile in response"); failed += 1
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
                rec = {
                    "id": f"{base.get('id', 'unknown')}-v{vi}-{vid}",
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
                kept += 1
                print(f"  variant {vi} kept ({res.baseline_size // 1_000_000} MB)")
                time.sleep(0.5)  # gentle on api
    print(f"done: kept {kept}, failed {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
