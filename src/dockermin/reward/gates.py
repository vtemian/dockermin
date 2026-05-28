"""Reward scoring: gates -> dense -> shape. Pure function for testability."""

from __future__ import annotations

import re

MIN_COMMANDS = 2

PARSE_FAIL_SCORE = -0.1
TOO_FEW_COMMANDS_SCORE = -0.2
BUILD_FAIL_SCORE = 0.0
TEST_FAIL_SCORE = 0.05

# Smoothing on the failure rungs. Without this, ~40% of early-training rollouts
# collapse to BUILD_FAIL_SCORE (0.0) — and 8-rollout GRPO groups with >=5
# build-fails have zero advantage -> wasted batch. Adding a small command-count
# credit ensures two distinct-but-broken Dockerfiles do not score identically.
# Caps are kept well below TEST_FAIL_SCORE->PASS gradient (0.05 -> 0.5) so the
# gate-order incentives (broken < buildable < passing) are preserved.
CMD_CREDIT_PER_CMD = 0.005
BUILD_FAIL_CMD_CREDIT_MAX = 0.10  # saturates at command_count >= 20
TEST_FAIL_CMD_CREDIT_MAX = 0.15  # buildable-but-failing earns more than broken


def _cmd_partial_credit(command_count: int, cap: float) -> float:
    """Per-instruction partial credit on the failure rungs, capped."""
    return min(cap, CMD_CREDIT_PER_CMD * max(0, command_count))


# Cache-hygiene substrings and their additive shape bonus.
_SUBSTRING_BONUSES = (
    ("rm -rf /var/lib/apt/lists", 0.02),
    ("--no-install-recommends", 0.02),
)


def _shape_bonus(text: str) -> float:
    """Additive bonuses for small-base / cache-hygiene patterns."""
    bonus = 0.0
    if re.search(r"from\s+\S*distroless", text):
        bonus += 0.05
    if re.search(r"from\s+\S*alpine", text):
        bonus += 0.03
    if text.count("from ") >= MIN_COMMANDS:  # multi-stage
        bonus += 0.05
    bonus += sum(value for needle, value in _SUBSTRING_BONUSES if needle in text)
    return bonus


# An image this small relative to its baseline with no multi-stage artifact is
# almost certainly empty (the FROM scratch + RUN echo cheat), not an honest shrink.
_TINY_IMAGE_FRACTION = 0.02
_TINY_IMAGE_PENALTY = -0.25


def _shape_penalty(text: str) -> float:
    """Negative shaping for unpinned tags and empty `FROM scratch`.

    The :latest tag penalty and bare-FROM (no tag) penalty compound
    separately. The original combined regex matched any single-token FROM
    line including tagged ones because colons are non-whitespace.
    """
    penalty = 0.0
    if ":latest" in text:
        penalty -= 0.05
    if re.search(r"^from\s+[^\s:@]+\s*$", text, re.M):
        # bare image reference: no ":tag" and no "@digest" pin
        penalty -= 0.05
    if "from scratch" in text and not re.search(r"\bcopy\b", text):
        penalty -= 0.10
    return penalty


def _tiny_image_penalty(text: str, baseline_size: int, new_size: int) -> float:
    """Soft anti-cheat: an image <2% of baseline with no `copy --from` artifact
    is most likely an empty shell (the FROM scratch + RUN echo cheat).

    Legit static binaries pulled from a builder stage (`COPY --from=`) are
    exempt - they can honestly be tiny.
    """
    if baseline_size <= 0:
        return 0.0
    if new_size >= baseline_size * _TINY_IMAGE_FRACTION:
        return 0.0
    if re.search(r"copy\s+--from=", text):
        return 0.0
    return _TINY_IMAGE_PENALTY


def compute_score(  # noqa: PLR0913 — every gate outcome + size measurement feeds the reward; collapsing them would hide the scoring inputs
    *,
    parse_ok: bool,
    build_ok: bool,
    test_ok: bool,
    command_count: int,
    baseline_size: int,
    new_size: int,
    dockerfile_text: str,
) -> float:
    """Composite reward. Keyword-only API is fixed by callers and tests."""
    # Gate ladder: first failing gate fixes the score; full pipeline must pass
    # to reach dense + shape scoring below. Build/test rungs now carry a small
    # per-command-count credit so two distinct-but-broken Dockerfiles do not
    # score identically — that's what was collapsing GRPO advantage in v0.
    gate_failures = (
        (not parse_ok, PARSE_FAIL_SCORE),
        (command_count < MIN_COMMANDS, TOO_FEW_COMMANDS_SCORE),
        (not build_ok, BUILD_FAIL_SCORE + _cmd_partial_credit(command_count, BUILD_FAIL_CMD_CREDIT_MAX)),
        (not test_ok, TEST_FAIL_SCORE + _cmd_partial_credit(command_count, TEST_FAIL_CMD_CREDIT_MAX)),
    )
    for failed, score in gate_failures:
        if failed:
            return score
    reduction = max(0.0, (baseline_size - new_size) / max(1, baseline_size))
    dense = min(1.0, reduction)
    text = dockerfile_text.lower()
    shape = _shape_bonus(text) + _shape_penalty(text) + _tiny_image_penalty(text, baseline_size, new_size)
    return min(1.0, 0.5 + 0.5 * dense + shape)
