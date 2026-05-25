"""Tests for the dockermin_reward wiring."""
import pytest

from dockermin.dataset.annotate import BuildResult, ParseResult, TestResult
from dockermin.reward import dockermin_reward as dr_mod
from dockermin.reward.dockermin_reward import dockermin_reward


def test_dockermin_reward_garbage_completion_returns_negative():
    # No fence -> extract returns None -> parse_gate fails on "" -> score = -0.1
    completion = [{"role": "assistant", "content": "just prose"}]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}
    score = dockermin_reward(completion=completion, info=info)
    assert score == pytest.approx(-0.1)


def test_dockermin_reward_happy_path_clean_shrink(monkeypatch: pytest.MonkeyPatch):
    """All gates pass, size shrinks 100 -> 80: dense=0.2, base=0.5, +0.5*0.2 = 0.6.

    We mock the three gates (parse/build/test) imported into the
    dockermin_reward module namespace so the test runs without docker.
    """
    fake_df = "FROM python:3.12-slim\nCMD [\"true\"]\n"
    completion = [
        {"role": "assistant", "content": f"```dockerfile\n{fake_df}\n```"},
    ]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}

    monkeypatch.setattr(dr_mod, "parse_gate",
                        lambda df: ParseResult(ok=True, command_count=3))
    monkeypatch.setattr(dr_mod, "build_gate",
                        lambda df, timeout_s=300: BuildResult(ok=True, tag="t", size_bytes=80))
    monkeypatch.setattr(dr_mod, "run_test_gate",
                        lambda tag, cmd, expected, timeout_s=30: TestResult(ok=True, output="ok"))

    score = dockermin_reward(completion=completion, info=info)
    # 0.5 base + 0.5 * 0.2 reduction + 0 shape (no distroless/alpine/multi-stage cues)
    assert score == pytest.approx(0.6, abs=0.001)
