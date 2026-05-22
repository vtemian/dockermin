"""Bulk-annotate candidates.jsonl into curated/triples.jsonl, dropping non-passing."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dockermin.dataset.annotate import annotate_one, infer_test_cmd

IN = Path("data/raw/candidates.jsonl")
OUT = Path("data/curated/triples.jsonl")
TARGET = 200
MAX_WORKERS = 8


def process(rec: dict) -> dict | None:
    df = rec.get("dockerfile", "")
    if not df:
        return None
    cmd, expected = infer_test_cmd(df)
    if cmd is None or expected is None:
        return None
    r = annotate_one(df, cmd, expected, build_timeout_s=300, test_timeout_s=30)
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


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    kept = 0
    with IN.open() as fin, OUT.open("w") as fout, ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            futures.append(ex.submit(process, rec))
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                kept += 1
                print(f"kept {kept}/{TARGET} (seen {seen})")
                if kept >= TARGET:
                    break
    print(f"done: kept {kept} of {seen} candidates")


if __name__ == "__main__":
    main()
