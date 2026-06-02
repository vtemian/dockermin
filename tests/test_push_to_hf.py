"""Tests for the holdout-id pinning in push_to_hf."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dockermin.dataset.split import split_with_frozen_holdout

if TYPE_CHECKING:
    from pathlib import Path


def test_split_with_frozen_holdout_uses_id_file(tmp_path: Path) -> None:
    """Rows whose id is in holdout_ids go to test split; everything else to train."""
    holdout_ids = {"a", "b", "c"}
    rows = [{"id": x, "dockerfile": f"FROM scratch  # {x}"} for x in ("a", "b", "c", "d", "e", "f")]
    train, test = split_with_frozen_holdout(rows, holdout_ids)
    assert {r["id"] for r in test} == holdout_ids
    assert {r["id"] for r in train} == {"d", "e", "f"}


def test_split_with_frozen_holdout_ignores_missing(tmp_path: Path) -> None:
    """If a holdout id is not in the corpus, the function skips it without raising
    (corpora can shrink over time; the holdout fixture remains the source of truth).
    """
    holdout_ids = {"a", "b", "missing"}
    rows = [{"id": x, "dockerfile": "..."} for x in ("a", "b", "c")]
    train, test = split_with_frozen_holdout(rows, holdout_ids)
    assert {r["id"] for r in test} == {"a", "b"}
    assert {r["id"] for r in train} == {"c"}


def test_split_with_frozen_holdout_rejects_empty_holdout(tmp_path: Path) -> None:
    """An empty holdout set is a configuration error, not a degenerate input."""
    with pytest.raises(ValueError, match="holdout_ids must be non-empty"):
        split_with_frozen_holdout([{"id": "a"}], set())
