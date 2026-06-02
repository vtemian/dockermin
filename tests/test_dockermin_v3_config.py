"""Smoke: the v3 config exists, is valid TOML, and encodes the hyperparam choices
from docs/plans/2026-06-02-grpo-v3-manifest-gate-and-dataset-expansion.md."""
from __future__ import annotations

import tomllib
from pathlib import Path


def test_v3_config_max_steps_in_range() -> None:
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert 400 <= cfg["max_steps"] <= 500


def test_v3_config_group_size_unchanged_from_v2() -> None:
    """Agent 4: no prime-rl precedent for group_size=32 on a 7B + LoRA."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["orchestrator"]["group_size"] == 16


def test_v3_config_batch_size_raised() -> None:
    """Agent 4: prime-rl reference configs run at batch_size 256-512; v2's 16 was unusually small.
    v3 raises to at least 32 to reduce variance estimate noise."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["orchestrator"]["batch_size"] >= 32


def test_v3_config_ckpt_interval_at_50() -> None:
    """Eval at steps 100, 200, 300, 400, 500 requires ckpts at the 50-step grid
    (the eval is offline — we read STABLE broadcasts post-hoc)."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["ckpt"]["interval"] == 50
