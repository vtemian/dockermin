"""Tests for the dockermin_reward wiring."""

from __future__ import annotations

import asyncio

import pytest

from dockermin.dataset.annotate import BuildResult, ParseResult, TestResult
from dockermin.reward import dockermin_reward as dr_mod
from dockermin.reward.dockermin_reward import dockermin_reward
from dockermin.reward.gates import (
    BUILD_FAIL_CMD_CREDIT_MAX,
    BUILD_FAIL_SCORE,
    CMD_CREDIT_PER_CMD,
    PARSE_FAIL_SCORE,
    compute_score,
)


class _AssistantMessage:
    """Stand-in for the pydantic message objects verifiers passes at rollout time."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


def test_completion_text_reads_pydantic_message_objects() -> None:
    # Verifiers passes message OBJECTS (not dicts) during live rollouts; the
    # extractor must read .content off them or every rollout scores as empty.
    completion = [_AssistantMessage(role="assistant", content="```dockerfile\nFROM x\n```")]
    assert dr_mod._completion_text(completion) == "```dockerfile\nFROM x\n```"


def test_completion_text_reads_message_dicts() -> None:
    completion = [{"role": "assistant", "content": "hello"}]
    assert dr_mod._completion_text(completion) == "hello"


def test_dockermin_reward_garbage_completion_returns_negative() -> None:
    # No fence -> extract returns None -> parse_gate fails on "" -> score = -0.1
    completion = [{"role": "assistant", "content": "just prose"}]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}
    score = asyncio.run(dockermin_reward(completion=completion, info=info))
    assert score == pytest.approx(-0.1)


def test_reward_is_async_and_handles_malformed_info() -> None:
    """Malformed info must yield a defined score, not raise (verifiers would
    swallow a raise to 0.0, indistinguishable from a real build failure)."""
    score = asyncio.run(
        dockermin_reward(
            completion=[{"role": "assistant", "content": "no fence here"}],
            info={},  # missing baseline_size/test_cmd
        )
    )
    assert score == 0.0


def test_reward_garbage_completion_still_negative() -> None:
    score = asyncio.run(
        dockermin_reward(
            completion=[{"role": "assistant", "content": "prose only"}],
            info={"baseline_size": 100, "test_cmd": ["true"], "expected_substring": "x"},
        )
    )
    assert score == pytest.approx(-0.1)  # parse gate fails on empty df


def test_dockermin_reward_happy_path_clean_shrink(monkeypatch: pytest.MonkeyPatch) -> None:
    """All gates pass, size shrinks 100 -> 80: dense=0.2, base=0.5, +0.5*0.2 = 0.6.

    We mock the three gates (parse/build/test) imported into the
    dockermin_reward module namespace so the test runs without docker.
    """
    fake_df = 'FROM python:3.12-slim\nCMD ["true"]\n'
    completion = [
        {"role": "assistant", "content": f"```dockerfile\n{fake_df}\n```"},
    ]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}

    monkeypatch.setattr(dr_mod, "parse_gate", lambda df: ParseResult(ok=True, command_count=3))
    monkeypatch.setattr(dr_mod, "build_gate", lambda df, timeout_s=300: BuildResult(ok=True, tag="t", size_bytes=80))
    monkeypatch.setattr(
        dr_mod, "run_test_gate", lambda tag, cmd, expected, timeout_s=30: TestResult(ok=True, output="ok")
    )

    score = asyncio.run(dockermin_reward(completion=completion, info=info))
    # 0.5 base + 0.5 * 0.2 reduction + 0 shape (no distroless/alpine/multi-stage cues)
    assert score == pytest.approx(0.6, abs=0.001)


def test_build_fail_partial_credit_scales_with_command_count() -> None:
    """A 5-instruction build-fail must score lower than a 20-instruction build-fail,
    so GRPO groups with mixed-complexity build failures have non-zero reward variance."""
    small = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=5,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    big = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=20,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    assert small == pytest.approx(BUILD_FAIL_SCORE + 5 * CMD_CREDIT_PER_CMD)
    assert big > small
    assert big <= BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX  # cap honoured


def test_build_fail_partial_credit_saturates_at_cap() -> None:
    """Even a 200-instruction build-fail must not score higher than the cap, otherwise
    a degenerate verbose Dockerfile could out-reward a buildable one."""
    huge = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=200,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    assert huge == pytest.approx(BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX)


def test_test_fail_partial_credit_separates_from_build_fail() -> None:
    """A test-failing build must always score higher than a build-failing one with the
    same command count — preserves the gate order even with smoothing."""
    bf = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=10,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    tf = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=False,
        command_count=10,
        baseline_size=100,
        new_size=80,
        dockerfile_text="",
    )
    assert tf > bf


def test_full_pass_unchanged_by_smoothing() -> None:
    """The pass-rung formula (0.5 + 0.5*dense + shape) must not be touched."""
    s = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=True,
        command_count=5,
        baseline_size=1000,
        new_size=500,
        dockerfile_text="",
    )
    # 50% reduction, no shape -> 0.5 + 0.5*0.5 = 0.75
    assert s == pytest.approx(0.75)


def test_parse_fail_unchanged() -> None:
    """The parse-fail rung is not smoothed (a non-parseable Dockerfile has no useful
    command_count to credit)."""
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
    assert s == PARSE_FAIL_SCORE


def test_partial_credit_caps_at_baseline_plus_two_when_baseline_provided() -> None:
    """Pad-hack guard: a 100-cmd build-fail with baseline=5 cmds must score the same
    as a 7-cmd build-fail (baseline+2), preventing reward gaming via no-op LABELs."""
    padded = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=100,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
        baseline_command_count=5,
    )
    capped = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=7,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
        baseline_command_count=5,
    )
    assert padded == pytest.approx(capped)  # 0.0 + 0.02 * 7 = 0.14


def test_partial_credit_unchanged_when_baseline_not_provided() -> None:
    """Back-compat: callers that omit baseline_command_count get the old behavior
    (uncapped per-cmd credit, capped only at BUILD_FAIL_CMD_CREDIT_MAX)."""
    score = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=100,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    assert score == pytest.approx(BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX)  # 0.30


def test_reward_threads_manifest_ok_through_every_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every compute_score call inside dockermin_reward must pass manifest_ok.

    compute_score's manifest_ok parameter is keyword-only and required: a missing
    arg raises TypeError, which verifiers would silently coerce to 0.0 and pollute
    the gradient. This smoke test exercises the parse-fail, build-fail, and
    full-pipeline-pass branches and asserts each returns a float (i.e. did not
    TypeError on the compute_score call).
    """
    fake_df = 'FROM python:3.12-slim\nCMD ["true"]\n'
    completion = [{"role": "assistant", "content": f"```dockerfile\n{fake_df}\n```"}]
    info = {
        "baseline_size": 50_000_000,
        "test_cmd": ["true"],
        "expected_substring": "",
        "baseline_command_count": 3,
    }

    monkeypatch.setattr(dr_mod, "parse_gate", lambda _df: ParseResult(ok=False, error="x"))
    assert isinstance(asyncio.run(dockermin_reward(completion=completion, info=info)), float)

    monkeypatch.setattr(dr_mod, "parse_gate", lambda _df: ParseResult(ok=True, command_count=3))
    monkeypatch.setattr(dr_mod, "build_gate", lambda _df, timeout_s=300: BuildResult(ok=False, error="rc=1"))
    assert isinstance(asyncio.run(dockermin_reward(completion=completion, info=info)), float)

    monkeypatch.setattr(dr_mod, "build_gate", lambda _df, timeout_s=300: BuildResult(ok=True, tag="t", size_bytes=80))
    monkeypatch.setattr(
        dr_mod, "run_test_gate", lambda _tag, _cmd, _expected, timeout_s=30: TestResult(ok=True, output="ok")
    )
    assert isinstance(asyncio.run(dockermin_reward(completion=completion, info=info)), float)


def test_rung_order_preserved_at_caps() -> None:
    """Max build_fail must remain strictly less than min test_fail (gate ordering).

    The lowest cmd_count that still reaches the test_fail rung is MIN_COMMANDS=2
    (anything smaller is captured by the too_few rung at -0.2), so the reachable
    minimum test_fail score is TEST_FAIL_SCORE + 2 * CMD_CREDIT_PER_CMD = 0.39.
    """
    from dockermin.reward.gates import MIN_COMMANDS, TEST_FAIL_SCORE

    max_bf = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=False,
        test_ok=False,
        command_count=200,
        baseline_size=100,
        new_size=0,
        dockerfile_text="",
    )
    min_tf = compute_score(
        parse_ok=True,
        manifest_ok=True,
        build_ok=True,
        test_ok=False,
        command_count=MIN_COMMANDS,
        baseline_size=100,
        new_size=80,
        dockerfile_text="",
    )
    assert max_bf < min_tf
    assert max_bf == pytest.approx(0.30)
    assert min_tf == pytest.approx(TEST_FAIL_SCORE + MIN_COMMANDS * CMD_CREDIT_PER_CMD)  # 0.39
