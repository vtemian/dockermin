"""Grouped train/test split that respects base/variant grouping.

Every variant of a base shares the base's ``base_id``; a row-level split
would put a variant in test while its base sits in train, which is not a
real holdout. We group rows by ``base_id or id`` and assign whole groups
to one side, so no base appears on both sides.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

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
