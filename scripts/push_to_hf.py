"""Push the variant triple set to HF Hub.

Uses the frozen v0 holdout — the 37 ids in ``data/curated/holdout_v0_ids.txt``
always go to test, everything else to train. Without this pin, growing the
training corpus drifts the holdout and invalidates v2/v3 comparisons. The
fixture was generated once from ``vtemian/dockermin-v0`` (split='test') and
is the comparison contract from v1 onward.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict

from dockermin.dataset.split import split_with_frozen_holdout

IN = Path("data/curated/triples_with_variants.jsonl")
HOLDOUT_FIXTURE = Path(__file__).parent.parent / "data" / "curated" / "holdout_v0_ids.txt"
REPO_ID = "vtemian/dockermin-v0"


def _require_token() -> None:
    if not (os.getenv("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").exists()):
        msg = "HF_TOKEN not set and not logged in. Run `huggingface-cli login` or set HF_TOKEN."
        raise SystemExit(msg)


def _load_holdout_ids() -> set[str]:
    return {line for line in HOLDOUT_FIXTURE.read_text().splitlines() if line.strip()}


def main() -> None:
    _require_token()
    records = [json.loads(line) for line in IN.read_text().splitlines() if line.strip()]
    train, test = split_with_frozen_holdout(records, _load_holdout_ids())
    dsd = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(test),
        }
    )
    dsd.push_to_hub(
        REPO_ID,
        private=False,
        commit_message=(f"{len(records)} triples, frozen v0 holdout ({len(train)} train / {len(test)} test rows)"),
    )
    print(f"pushed {len(train)} train + {len(test)} test rows to {REPO_ID}")


if __name__ == "__main__":
    main()
