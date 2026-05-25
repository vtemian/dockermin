"""Tests for the shared jsonl + lockfile ops module."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from dockermin.ops import (
    acquire_lock,
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    write_jsonl,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"id": "a", "n": 1}, {"id": "b", "n": 2}]
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_write_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"id": "a"}]
    write_jsonl(str(path), records)
    assert read_jsonl(str(path)) == records


def test_iter_jsonl_streams_records(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"i": i} for i in range(3)]
    write_jsonl(path, records)
    assert list(iter_jsonl(path)) == records


def test_append_jsonl_adds_one_record(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [{"id": "a"}])
    append_jsonl(path, {"id": "b"})
    assert read_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_append_jsonl_creates_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    append_jsonl(path, {"id": "a"})
    assert read_jsonl(path) == [{"id": "a"}]


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "a"}\n\n   \n{"id": "b"}\n')
    assert read_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_read_jsonl_tolerates_trailing_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c"')  # truncated mid-write
    assert read_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_iter_jsonl_tolerates_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"id": "a"}\nnot json\n{"id": "b"}\n')
    assert list(iter_jsonl(path)) == [{"id": "a"}, {"id": "b"}]


def test_read_jsonl_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl")


def test_acquire_lock_creates_file_holding_pid(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    fd = acquire_lock(str(lock))
    try:
        assert lock.exists()
        assert lock.read_text().strip() == str(os.getpid())
    finally:
        os.close(fd)
