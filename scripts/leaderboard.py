"""Aggregate eval JSONL into a leaderboard markdown table.

For each baseline, computes:
  * Test-pass rate           (% of triples whose rewritten image still passed the test)
  * Mean reduction | pass    (mean size reduction conditional on test_passes=True)
  * Mean elapsed seconds     (average wall-clock per Dockerfile)

We expect the input JSONL to carry rows of the EvalEntry shape, plus a sibling
holdout (the dockermin-v0 dataset) so we can recover ``baseline_size`` and
compute the reduction. The holdout id used at eval time should be passed via
``--holdout`` (default: ``vladtemian/dockermin-v0``).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load_baseline_sizes(holdout_id: str) -> dict[str, int]:
    """Map triple_id -> baseline_size from the holdout dataset."""
    from datasets import load_dataset
    ds = load_dataset(holdout_id)
    split = ds["test"] if "test" in ds else ds["train"]
    sizes: dict[str, int] = {}
    for row in split:
        key = str(row.get("id") or row.get("source_url") or "")
        if key and "baseline_size" in row:
            sizes[key] = int(row["baseline_size"])
    return sizes


def _aggregate(rows: list[dict], baseline_sizes: dict[str, int]) -> dict[str, dict]:
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_baseline[r["baseline"]].append(r)

    stats: dict[str, dict] = {}
    for name, entries in by_baseline.items():
        n = len(entries)
        n_pass = sum(1 for e in entries if e.get("test_passes"))
        elapsed = [float(e.get("elapsed_s", 0.0)) for e in entries]
        reductions: list[float] = []
        for e in entries:
            if not e.get("test_passes"):
                continue
            base = baseline_sizes.get(e["triple_id"])
            new = e.get("new_size_bytes")
            if base is None or new is None or base <= 0:
                continue
            reductions.append(max(0.0, (base - new) / base))
        stats[name] = {
            "n": n,
            "test_pass_rate": (n_pass / n) if n else 0.0,
            "mean_reduction_given_pass": statistics.mean(reductions) if reductions else 0.0,
            "mean_elapsed_s": statistics.mean(elapsed) if elapsed else 0.0,
            "n_with_reduction": len(reductions),
        }
    return stats


def _render_markdown(stats: dict[str, dict]) -> str:
    # Sort by reduction (descending) conditional on test pass; that's the
    # headline metric per Task 5.10.
    ordered = sorted(
        stats.items(),
        key=lambda kv: kv[1]["mean_reduction_given_pass"],
        reverse=True,
    )
    lines = [
        "# Dockermin leaderboard",
        "",
        "Headline metric: **mean size reduction conditional on test pass**.",
        "",
        "| Baseline | n | Test-pass rate | Mean reduction \\| pass | Mean elapsed (s) | n reductions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in ordered:
        lines.append(
            f"| {name} "
            f"| {s['n']} "
            f"| {s['test_pass_rate']:.1%} "
            f"| {s['mean_reduction_given_pass']:.1%} "
            f"| {s['mean_elapsed_s']:.1f} "
            f"| {s['n_with_reduction']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate eval results into a leaderboard.")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="Input JSONL of EvalEntry rows.")
    p.add_argument("--out", type=Path, default=Path("docs/leaderboard.md"),
                   help="Output markdown path. Default: docs/leaderboard.md")
    p.add_argument("--holdout", default="vladtemian/dockermin-v0",
                   help="HF dataset id for baseline_size lookup.")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.in_path.read_text().splitlines() if line.strip()]
    print(f"loaded {len(rows)} eval rows", file=sys.stderr)

    baseline_sizes = _load_baseline_sizes(args.holdout)
    print(f"loaded {len(baseline_sizes)} baseline sizes from {args.holdout}", file=sys.stderr)

    stats = _aggregate(rows, baseline_sizes)
    md = _render_markdown(stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"wrote leaderboard to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
