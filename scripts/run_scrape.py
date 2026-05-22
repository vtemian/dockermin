"""Run all scrapers, write candidates to data/raw/candidates.jsonl. CPU-only."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from dockermin.dataset.scrape import (
    fetch_awesome_compose,
    fetch_github_search,
    fetch_official_images,
)

OUT = Path("data/raw/candidates.jsonl")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT.open("w") as f:
        for c in fetch_official_images(limit=100):
            f.write(json.dumps(asdict(c)) + "\n")
            count += 1
        for c in fetch_awesome_compose(limit=50):
            f.write(json.dumps(asdict(c)) + "\n")
            count += 1
        for c in fetch_github_search(limit=150):
            f.write(json.dumps(asdict(c)) + "\n")
            count += 1
    print(f"wrote {count} candidates to {OUT}")


if __name__ == "__main__":
    main()
