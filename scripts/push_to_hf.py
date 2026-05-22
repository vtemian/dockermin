"""Push curated/triples.jsonl to HF Hub as vladtemian/dockermin-v0."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

IN = Path("data/curated/triples.jsonl")
REPO_ID = "vladtemian/dockermin-v0"


def main() -> None:
    records = [
        json.loads(line) for line in IN.read_text().splitlines() if line.strip()
    ]
    ds = Dataset.from_list(records)
    ds.push_to_hub(
        REPO_ID,
        private=False,
        commit_message=(
            f"v0: {len(records)} curated Dockerfile triples (parse+build+test verified)"
        ),
    )
    print(f"pushed {len(records)} records to {REPO_ID}")


if __name__ == "__main__":
    main()
