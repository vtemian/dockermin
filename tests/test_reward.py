"""Tests for the dockermin_reward wiring."""
import pytest

from dockermin.reward.dockermin_reward import dockermin_reward


def test_dockermin_reward_garbage_completion_returns_negative():
    # No fence -> extract returns None -> parse_gate fails on "" -> score = -0.1
    completion = [{"role": "assistant", "content": "just prose"}]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}
    score = dockermin_reward(completion=completion, info=info)
    assert score == pytest.approx(-0.1)
