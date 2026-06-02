"""Tests for the holdout-id pinning in push_to_hf."""

from __future__ import annotations

from pathlib import Path

import pytest

from dockermin.dataset.split import split_with_frozen_holdout


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


def test_load_holdout_ids_strips_whitespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cubic flagged on PR #9: trailing whitespace in the fixture mis-partitioned
    rows whose id contained no whitespace. _load_holdout_ids must strip each line.
    """
    import importlib.util
    import sys

    fixture = tmp_path / "holdout.txt"
    fixture.write_text("clean-id\n  whitespace-id  \n\nanother-id\n")

    # Import scripts/push_to_hf.py by path (it isn't a Python package).
    spec = importlib.util.spec_from_file_location(
        "push_to_hf_test", Path(__file__).parent.parent / "scripts" / "push_to_hf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["push_to_hf_test"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "HOLDOUT_FIXTURE", fixture)

    ids = module._load_holdout_ids()
    assert ids == {"clean-id", "whitespace-id", "another-id"}
    # Negative assertion: pre-fix behaviour would have kept "  whitespace-id  ".
    assert "  whitespace-id  " not in ids
