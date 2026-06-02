"""Bulk-annotate candidates.jsonl into curated/triples.jsonl, dropping non-passing.

Writes a fresh per-run output file (so we can re-run safely without clobbering
the curated set while another consumer is reading it), then atomically merges
into the curated triples on success. Dedup is on the candidate id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dockermin.dataset.annotate import annotate_one, infer_test_cmd
from dockermin.ops import acquire_lock, read_jsonl

DEFAULT_IN = Path("data/raw/candidates.jsonl")
DEFAULT_OUT = Path("data/curated/triples.jsonl")
TARGET = 200
MAX_WORKERS = 4


def process(rec: dict, builder: str | None = None, cache_dir: str | None = None) -> dict | None:
    df = rec.get("dockerfile", "")
    if not df:
        return None
    cmd, expected = infer_test_cmd(df)
    if cmd is None or expected is None:
        return None
    r = annotate_one(df, cmd, expected, build_timeout_s=300, test_timeout_s=30, builder=builder, cache_dir=cache_dir)
    if not r.ok:
        return None
    return {
        "id": rec.get("url", "unknown"),
        "dockerfile": df,
        "test_cmd": cmd,
        "expected_substring": expected,
        "baseline_size": r.baseline_size,
        "baseline_build_s": r.baseline_build_s,
        "ecosystem": rec.get("ecosystem", "unknown"),
        "source_url": rec.get("url", ""),
        "license": rec.get("license"),
    }


def _merge_into_curated(new_triples: list[dict], out_path: Path) -> tuple[int, int, int]:
    """Merge new triples into out_path, dedup by id, atomic replace.

    Returns (existing, added, total).
    """
    existing: list[dict] = read_jsonl(out_path) if out_path.exists() else []
    by_id: dict[str, dict] = {r.get("id", ""): r for r in existing}
    added = 0
    for t in new_triples:
        tid = t.get("id", "")
        if tid not in by_id:
            added += 1
        by_id[tid] = t
    merged = list(by_id.values())
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as fout:
        for r in merged:
            fout.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)
    return (len(existing), added, len(merged))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input candidates.jsonl path")
    ap.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="curated triples.jsonl path (merged into)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument(
        "--lockfile", default="logs/run_annotate.lock", help="exclusive flock path to prevent concurrent runs"
    )
    ap.add_argument(
        "--builder", default=None, help="buildx builder name (e.g. 'dockermin' on the pod); None uses the default"
    )
    ap.add_argument(
        "--cache-dir", dest="cache_dir", default=None, help="buildx cache dir (e.g. /scratch/bkcache on the pod)"
    )
    args = ap.parse_args()

    Path(args.lockfile).parent.mkdir(parents=True, exist_ok=True)
    acquire_lock(args.lockfile)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-run scratch file so we never blank the curated file mid-run.
    scratch = Path(tempfile.mkstemp(prefix="annotate_", suffix=".jsonl", dir=str(out_path.parent))[1])
    print(f"reading {in_path} -> scratch {scratch} (workers={args.workers}, target={args.target})")

    seen = 0
    kept_records: list[dict] = []
    with in_path.open() as fin, scratch.open("w") as fout, ThreadPoolExecutor(args.workers) as ex:
        futures = []
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            futures.append(ex.submit(process, rec, args.builder, args.cache_dir))
        for fut in as_completed(futures):
            # A single candidate's annotation failure (docker daemon read-timeout,
            # subprocess crash, BLE001-class errors) must not kill the whole run.
            # Skip the candidate, keep going. Diagnostics for the lost-candidate
            # case go to stderr so they survive the wider redirect to annotate.log.
            try:
                res = fut.result()
            except Exception as e:
                print(f"WARN: candidate failed: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if res:
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                kept_records.append(res)
                print(f"kept {len(kept_records)}/{args.target} (seen {seen})")
                if len(kept_records) >= args.target:
                    break
    print(f"scratch done: kept {len(kept_records)} of {seen} candidates")

    existing, added, total = _merge_into_curated(kept_records, out_path)
    print(f"merged into {out_path}: existing={existing} added={added} total={total}")


if __name__ == "__main__":
    main()
