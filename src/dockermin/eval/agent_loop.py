"""Claude Code headless agent-loop baseline.

Sonnet 4.6 driven via ``claude -p`` with a docker-build / docker-run bash
allowlist. Up to 5 turns, $0.50 budget per Dockerfile. The agent edits
``./Dockerfile`` in place; we read the final state back and verify with
``annotate_one``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dockermin.eval.baselines import (
    EvalEntry,
    _entry,
    _error_entry,
    register_baseline,
)


def run_agent_loop(dockerfile: str, test_cmd: list[str], expected: str,
                   max_turns: int = 5) -> dict:
    """Drive Claude Code headless to optimize a single Dockerfile.

    Returns a dict with the final Dockerfile text and run metadata
    (cost_usd, turns, exit_code). The caller is responsible for verifying the
    rewritten Dockerfile builds and tests pass.
    """
    workdir = Path(tempfile.mkdtemp(prefix="agentloop_"))
    (workdir / "Dockerfile").write_text(dockerfile)
    prompt = (
        f"Optimize ./Dockerfile to be smaller. Build it with docker build. "
        f"Verify test_cmd passes inside the built image: {' '.join(test_cmd)} "
        f"should output a string containing {expected!r}. "
        f"Iterate up to {max_turns} times. Stop when both: image is smaller "
        f"than original AND test passes."
    )
    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--output-format", "json",
                "--model", "claude-sonnet-4-6",
                "--max-turns", str(max_turns),
                "--max-budget-usd", "0.50",
                "--permission-mode", "bypassPermissions",
                "--allowedTools",
                "Bash(docker build *)", "Bash(docker run *)", "Read", "Edit", "Write",
                "--cwd", str(workdir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        meta = json.loads(result.stdout) if result.returncode == 0 else {}
        final_df = (workdir / "Dockerfile").read_text()
        return {
            "final_dockerfile": final_df,
            "cost_usd": meta.get("total_cost_usd", 0),
            "turns": meta.get("num_turns", 0),
            "exit_code": result.returncode,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def baseline_agent_loop(triple: dict) -> EvalEntry:
    """Wrap run_agent_loop with annotate-verified size measurement."""
    t0 = time.perf_counter()
    try:
        res = run_agent_loop(
            triple["dockerfile"],
            triple["test_cmd"],
            triple.get("expected_substring", ""),
        )
        new_df = res.get("final_dockerfile") or ""
        if not new_df.strip():
            return _error_entry("agent_loop", triple, t0, "agent produced empty dockerfile")
        return _entry("agent_loop", triple, new_df, t0)
    except FileNotFoundError as e:
        return _error_entry("agent_loop", triple, t0, f"claude binary not found: {e!r}")
    except subprocess.TimeoutExpired:
        return _error_entry("agent_loop", triple, t0, "agent loop timed out")
    except Exception as e:
        return _error_entry("agent_loop", triple, t0, f"agent loop error: {e!r}")


# Side-channel registration so the dispatcher in baselines.py can route to us
# without forming an import cycle.
register_baseline("agent_loop", baseline_agent_loop)
