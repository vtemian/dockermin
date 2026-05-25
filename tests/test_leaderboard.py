"""Tests for leaderboard aggregation guards.

Two honesty guards:
  * a triple missing ``baseline_size`` must be reported as a skip, not
    silently dropped from the reduction metric;
  * if the baselines did not all see the same number of triples, that
    mismatch must be surfaced loudly (it means the eval is comparing
    apples to oranges).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "leaderboard", Path(__file__).resolve().parent.parent / "scripts" / "leaderboard.py"
)
assert _spec is not None and _spec.loader is not None
leaderboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leaderboard)


def test_missing_baseline_size_is_counted_as_skip_not_dropped() -> None:
    rows = [
        {"baseline": "a", "triple_id": "t1", "new_size_bytes": 50, "test_passes": True, "elapsed_s": 1.0},
        {"baseline": "a", "triple_id": "t2", "new_size_bytes": 50, "test_passes": True, "elapsed_s": 1.0},
    ]
    baseline_sizes = {"t1": 100}  # t2 has no baseline_size
    stats = leaderboard._aggregate(rows, baseline_sizes)
    assert stats["a"]["n_missing_baseline"] == 1
    # the passing triple WITH a size still contributes a reduction
    assert stats["a"]["n_with_reduction"] == 1


def test_mismatched_triple_counts_detected() -> None:
    stats = {
        "a": {"n": 5},
        "b": {"n": 4},
    }
    mismatch = leaderboard._baseline_count_mismatch(stats)
    assert mismatch is not None
    assert mismatch == {"a": 5, "b": 4}


def test_equal_triple_counts_no_mismatch() -> None:
    stats = {"a": {"n": 5}, "b": {"n": 5}}
    assert leaderboard._baseline_count_mismatch(stats) is None


def test_no_missing_baseline_when_all_present() -> None:
    rows = [
        {"baseline": "a", "triple_id": "t1", "new_size_bytes": 50, "test_passes": True, "elapsed_s": 1.0},
    ]
    stats = leaderboard._aggregate(rows, {"t1": 100})
    assert stats["a"]["n_missing_baseline"] == 0
