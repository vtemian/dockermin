"""Tests for reward gates pure scoring."""
import pytest

from dockermin.reward.gates import compute_score

def test_compute_score_parse_fail_returns_minus_point_1():
    s = compute_score(parse_ok=False, build_ok=False, test_ok=False,
                      command_count=0, baseline_size=100, new_size=0, dockerfile_text="")
    assert s == pytest.approx(-0.1)

def test_compute_score_too_few_commands_returns_minus_point_2():
    s = compute_score(parse_ok=True, build_ok=False, test_ok=False,
                      command_count=1, baseline_size=100, new_size=0, dockerfile_text="FROM scratch")
    assert s == pytest.approx(-0.2)

def test_compute_score_build_fail_returns_zero():
    s = compute_score(parse_ok=True, build_ok=False, test_ok=False,
                      command_count=3, baseline_size=100, new_size=0, dockerfile_text="...")
    assert s == pytest.approx(0.0)

def test_compute_score_build_pass_test_fail_returns_point_05():
    s = compute_score(parse_ok=True, build_ok=True, test_ok=False,
                      command_count=3, baseline_size=100, new_size=80, dockerfile_text="...")
    assert s == pytest.approx(0.05)

def test_compute_score_test_pass_full_reduction_returns_around_one():
    s = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                      command_count=3, baseline_size=100, new_size=10,
                      dockerfile_text="FROM gcr.io/distroless/python3-debian12\nCMD [\"x\"]")
    assert 0.95 <= s <= 1.10  # 0.5 base + 0.5 * 0.9 reduction + small distroless bonus

def test_shape_bonus_gated_on_test_pass():
    # Build passes but test fails: NO shape bonus from alpine string
    s = compute_score(parse_ok=True, build_ok=True, test_ok=False,
                      command_count=3, baseline_size=100, new_size=20,
                      dockerfile_text="FROM alpine\nRUN whatever")
    assert s == pytest.approx(0.05)  # NO alpine shape bonus when test failed

def test_latest_tag_penalty_applied_when_test_passes():
    s_with = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                           command_count=3, baseline_size=100, new_size=80,
                           dockerfile_text="FROM python:latest\nCMD [\"x\"]")
    s_without = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                              command_count=3, baseline_size=100, new_size=80,
                              dockerfile_text="FROM python:3.12-slim\nCMD [\"x\"]")
    assert s_without > s_with
