"""Tests for the grouped train/test split.

The split must keep every variant of a base on the same side of the
train/test boundary - a test variant whose base sits in train is not a
real holdout, so a naive row-level split would leak.
"""

from __future__ import annotations

from dockermin.dataset.split import Row, grouped_train_test_split


def _base_of(row: Row) -> str:
    return str(row.get("base_id") or row["id"])


def test_split_keeps_all_variants_of_a_base_on_one_side() -> None:
    rows: list[Row] = [
        {"id": "b1", "base_id": None},
        {"id": "b1-v0", "base_id": "b1"},
        {"id": "b2", "base_id": None},
        {"id": "b2-v0", "base_id": "b2"},
    ]
    train, test = grouped_train_test_split(rows, test_frac=0.5, seed=0)
    train_bases = {_base_of(r) for r in train}
    test_bases = {_base_of(r) for r in test}
    assert train_bases.isdisjoint(test_bases)


def test_split_returns_all_rows_exactly_once() -> None:
    rows: list[Row] = [
        {"id": "b1", "base_id": None},
        {"id": "b1-v0", "base_id": "b1"},
        {"id": "b2", "base_id": None},
        {"id": "b3", "base_id": None},
    ]
    train, test = grouped_train_test_split(rows, test_frac=0.5, seed=0)
    got = sorted(str(r["id"]) for r in train + test)
    assert got == sorted(str(r["id"]) for r in rows)


def test_split_is_deterministic_for_a_seed() -> None:
    rows: list[Row] = [{"id": f"b{i}", "base_id": None} for i in range(10)]
    a = grouped_train_test_split(rows, test_frac=0.3, seed=42)
    b = grouped_train_test_split(rows, test_frac=0.3, seed=42)
    assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]]
    assert [r["id"] for r in a[1]] == [r["id"] for r in b[1]]


def test_split_test_fraction_yields_at_least_one_base() -> None:
    # 11 distinct bases, test_frac=0.3 -> ceil-ish to >=3 test bases.
    rows: list[Row] = [{"id": f"b{i}", "base_id": None} for i in range(11)]
    _train, test = grouped_train_test_split(rows, test_frac=0.3, seed=1)
    test_bases = {_base_of(r) for r in test}
    assert len(test_bases) >= 3


def test_split_groups_by_base_not_row() -> None:
    # One base with many variants must not be split across sides.
    rows: list[Row] = [{"id": "b1", "base_id": None}]
    rows += [{"id": f"b1-v{i}", "base_id": "b1"} for i in range(8)]
    rows += [{"id": "b2", "base_id": None}]
    train, test = grouped_train_test_split(rows, test_frac=0.5, seed=3)
    train_bases = {_base_of(r) for r in train}
    test_bases = {_base_of(r) for r in test}
    assert train_bases.isdisjoint(test_bases)
    assert {"b1", "b2"} == train_bases | test_bases
