"""Train/test split utilities.

Two splitters live here:

- ``grouped_train_test_split`` — the original v0 splitter; deterministic,
  groups variants with their base. Kept for backfill / one-off rebuilds.
- ``split_with_frozen_holdout`` — the v3+ splitter. The v0 holdout was an
  emergent property of ``grouped_train_test_split(seed=0)`` on the 145-row
  ``triples_with_variants.jsonl`` snapshot that existed at the time. As
  the training corpus grows, the holdout ids MUST stay constant so v2/v3/v4
  numbers remain comparable; this function pins the holdout via an explicit
  id list rather than re-deriving it.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

Row = dict[str, object]


def grouped_train_test_split(
    rows: list[Row],
    test_frac: float,
    seed: int,
) -> tuple[list[Row], list[Row]]:
    """Split ``rows`` into (train, test) grouped by ``base_id or id``.

    Group keys are shuffled deterministically by ``seed`` and the first
    ``ceil(num_groups * test_frac)`` groups go to test (so a small base
    count still yields a non-empty holdout). Whole groups stay together,
    guaranteeing the train and test base-sets are disjoint.

    Within each side, original row order is preserved.
    """
    if not 0.0 < test_frac < 1.0:
        msg = f"test_frac must be in (0, 1), got {test_frac}"
        raise ValueError(msg)

    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        key = str(row.get("base_id") or row["id"])
        groups[key].append(row)

    keys = sorted(groups)  # stable starting order before the seeded shuffle
    random.Random(seed).shuffle(keys)

    n_test = math.ceil(len(keys) * test_frac)
    test_keys = set(keys[:n_test])

    train: list[Row] = []
    test: list[Row] = []
    for key in groups:  # preserve first-seen group order for stable output
        target = test if key in test_keys else train
        target.extend(groups[key])
    return train, test


def _row_side(row: dict[str, Any], holdout_ids: set[str]) -> str | None:
    """Return ``"test"`` if the row's id is in the holdout, ``"train"`` if not,
    or ``None`` if the row has no id (silently skipped — malformed input must
    not poison the split).
    """
    rid = row.get("id")
    if rid is None:
        return None
    return "test" if rid in holdout_ids else "train"


def split_with_frozen_holdout(
    rows: list[dict[str, Any]], holdout_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``rows`` into (train, test) where test contains exactly the rows
    whose ``id`` is in ``holdout_ids``. Missing ids are silently skipped — the
    holdout fixture is the source of truth, not the corpus.
    """
    if not holdout_ids:
        msg = "holdout_ids must be non-empty; the v0 holdout is the comparison contract"
        raise ValueError(msg)
    sides: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for row in rows:
        side = _row_side(row, holdout_ids)
        if side is not None:
            sides[side].append(row)
    return sides["train"], sides["test"]
