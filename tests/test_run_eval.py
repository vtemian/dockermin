"""CLI smoke tests for ``scripts/run_eval.py``.

These tests shell out to ``--help`` rather than importing the script, so they
verify the user-facing argparse surface without booting the HF / docker stack.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_eval_cli_accepts_dockermin_subfolder() -> None:
    """CLI must expose --dockermin-subfolder so we can eval per-step adapters
    (e.g. step_100 vs step_250) from the same HF repo."""
    result = subprocess.run(  # noqa: S603 — argv is sys.executable + repo-internal script path; no shell, no user input
        [sys.executable, str(Path(__file__).parent.parent / "scripts" / "run_eval.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dockermin-subfolder" in result.stdout
