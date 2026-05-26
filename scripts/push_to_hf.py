"""Push the variant triple set to HF Hub as vtemian/dockermin-v0.

Pushes a grouped train/test split: variants share a ``base_id`` with their
base, so the split groups by base to keep all variants of a base on one
side (a leaked holdout is worthless). With ~11 distinct bases the test
fraction is sized to yield >= 3 holdout bases - a small-N caveat that the
dataset card should call out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict

from dockermin.dataset.split import grouped_train_test_split

IN = Path("data/curated/triples_with_variants.jsonl")
REPO_ID = "vtemian/dockermin-v0"
TEST_FRAC = 0.3
SPLIT_SEED = 0


def _require_token() -> None:
    if not (os.getenv("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").exists()):
        msg = "HF_TOKEN not set and not logged in. Run `huggingface-cli login` or set HF_TOKEN."
        raise SystemExit(msg)


def main() -> None:
    _require_token()
    records = [json.loads(line) for line in IN.read_text().splitlines() if line.strip()]
    train, test = grouped_train_test_split(records, test_frac=TEST_FRAC, seed=SPLIT_SEED)
    dsd = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(test),
        }
    )
    dsd.push_to_hub(
        REPO_ID,
        private=False,
        commit_message=(
            f"v0: {len(records)} triples, grouped split ({len(train)} train / {len(test)} test rows, by base_id)"
        ),
    )
    print(f"pushed {len(train)} train + {len(test)} test rows to {REPO_ID}")


if __name__ == "__main__":
    main()
