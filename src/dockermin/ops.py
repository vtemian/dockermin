"""Shared filesystem ops: jsonl read/write and exclusive run locking.

Single source of truth for helpers that were previously copy-pasted across
the operational scripts (annotate, synthetic_variants, audit).
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _lock_held_message(path: str) -> str:
    try:
        with open(path) as f:
            holder = f.read().strip()
    except OSError:
        return f"already running (lockfile {path} held)"
    return f"already running (lockfile {path} held by pid {holder})"


def _release_lock(fd: int) -> None:
    # Closing the fd releases the flock; suppress OSError if already closed.
    with contextlib.suppress(OSError):
        os.close(fd)


def acquire_lock(path: str) -> int:
    """Acquire an exclusive flock on path. Exit with code 1 if held."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # stderr is the operator-facing channel for these CLI scripts.
        print(_lock_held_message(path), file=sys.stderr)  # noqa: T201
        sys.exit(1)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    # Keep fd open for the lifetime of the process.
    atexit.register(_release_lock, fd)
    return fd


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a .jsonl file, skipping blank/unparseable lines.

    Tolerates a trailing partial line from an interrupted write.
    """
    with Path(path).open() as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a .jsonl file (see iter_jsonl for tolerances)."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records as one JSON object per line, overwriting any existing file."""
    with Path(path).open("w") as fout:
        for rec in records:
            fout.write(json.dumps(rec) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single record as a JSON line, creating the file if absent."""
    with Path(path).open("a") as fout:
        fout.write(json.dumps(record) + "\n")
