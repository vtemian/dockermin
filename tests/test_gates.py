"""Tests for reward gates pure scoring."""

from __future__ import annotations

import pytest

from dockermin.reward.gates import (
    BUILD_FAIL_CMD_CREDIT_MAX,
    BUILD_FAIL_SCORE,
    MANIFEST_FAIL_SCORE,
    PARSE_FAIL_SCORE,
    TEST_FAIL_CMD_CREDIT_MAX,
    TEST_FAIL_SCORE,
    compute_score,
)


def test_compute_score_parse_fail_returns_minus_point_1() -> None:
    s = compute_score(
        parse_ok=False,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=0,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    assert s == pytest.approx(-0.1)


def test_compute_score_too_few_commands_returns_minus_point_2() -> None:
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=1,
        baseline_size=100,
        new_size=0,
        dockerfile_text="FROM scratch",
    )
    assert s == pytest.approx(-0.2)


def test_compute_score_build_fail_returns_smoothed_credit() -> None:
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=3,
        baseline_size=100,
        new_size=0,
        dockerfile_text="...",
    )
    assert s == pytest.approx(0.06)


def test_compute_score_test_fail_returns_smoothed_credit() -> None:
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=False,
        command_count=3,
        baseline_size=100,
        new_size=80,
        dockerfile_text="...",
    )
    assert s == pytest.approx(0.41)


def test_compute_score_test_pass_full_reduction_returns_around_one() -> None:
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=10,
        dockerfile_text='FROM gcr.io/distroless/python3-debian12\nCMD ["x"]',
    )
    assert 0.95 <= s <= 1.0  # 0.5 base + 0.5 * 0.9 reduction + small distroless bonus, capped at 1.0


def test_shape_bonus_gated_on_test_pass() -> None:
    # Build passes but test fails: NO shape bonus from alpine string
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=False,
        command_count=3,
        baseline_size=100,
        new_size=20,
        dockerfile_text="FROM alpine\nRUN whatever",
    )
    assert s == pytest.approx(
        0.41
    )  # smoothed test-fail credit (0.35 floor + 0.06 cmd credit); alpine shape bonus still gated off


def test_from_scratch_with_leading_copy_no_penalty() -> None:
    # Regression: prior implementation matched " copy " with surrounding spaces,
    # so a leading "COPY app /app" at line start was missed and the -0.10
    # "from scratch with no COPY" penalty was wrongly applied.
    df = 'FROM scratch\nCOPY app /app\nCMD ["/app"]'
    s_with_copy = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=80,
        dockerfile_text=df,
    )
    s_no_copy = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=80,
        dockerfile_text='FROM scratch\nCMD ["/x"]',
    )
    # The COPY variant must NOT take the -0.10 penalty, so its score is higher.
    assert s_with_copy > s_no_copy
    # And specifically: the gap should be ~0.10 (the penalty we avoided).
    assert s_with_copy - s_no_copy == pytest.approx(0.10, abs=0.001)


def test_from_scratch_echo_cheat_scores_low() -> None:
    """FROM scratch + RUN echo (the classic cheat) must not out-score an
    honest shrink, even if it somehow passed the test gate."""
    cheat = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=2,
        baseline_size=100,
        new_size=1,
        dockerfile_text="FROM scratch\nRUN echo ok",
    )
    honest = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=4,
        baseline_size=100,
        new_size=80,
        dockerfile_text='FROM python:3.12-slim\nRUN pip install --no-cache-dir flask\nCMD ["python","-m","flask"]',
    )
    assert cheat < honest


def test_suspiciously_tiny_image_soft_penalised() -> None:
    """An image <2% of baseline with no multi-stage COPY is likely empty -
    soft penalty, not a hard reject (legit static distroless binaries exist)."""
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=2,
        baseline_size=100,
        new_size=1,
        dockerfile_text="FROM scratch\nRUN echo ok",
    )
    s_legit = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=1,
        dockerfile_text='FROM scratch\nCOPY --from=build /app /app\nENTRYPOINT ["/app"]',
    )
    assert s < s_legit


def test_latest_tag_penalty_applied_when_test_passes() -> None:
    s_with = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=80,
        dockerfile_text='FROM python:latest\nCMD ["x"]',
    )
    s_without = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=3,
        baseline_size=100,
        new_size=80,
        dockerfile_text='FROM python:3.12-slim\nCMD ["x"]',
    )
    assert s_without > s_with


def test_compute_score_manifest_fail_returns_minus_005() -> None:
    """Hallucinated FROM tag (manifest_ok=False) scores MANIFEST_FAIL_SCORE,
    strictly between PARSE_FAIL and BUILD_FAIL with no command-count credit."""
    score = compute_score(
        parse_ok=True,
        manifest_ok=False,
        build_ok=False,
        test_ok=False,
        command_count=10,
        baseline_size=1_000_000,
        new_size=500_000,
        dockerfile_text="FROM hallucinated:tag\nRUN echo hi",
        baseline_command_count=8,
    )
    assert score == MANIFEST_FAIL_SCORE
    assert MANIFEST_FAIL_SCORE == -0.05  # contract — bump on purpose only


def test_compute_score_manifest_invariant_holds() -> None:
    """max(manifest_fail) < min(build_fail with no credit) < min(test_fail) < pass floor."""
    assert MANIFEST_FAIL_SCORE < BUILD_FAIL_SCORE
    assert MANIFEST_FAIL_SCORE > PARSE_FAIL_SCORE  # manifest_fail is "more progress" than parse_fail
    assert BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX < TEST_FAIL_SCORE
    assert TEST_FAIL_SCORE + TEST_FAIL_CMD_CREDIT_MAX < 0.5  # pass floor
