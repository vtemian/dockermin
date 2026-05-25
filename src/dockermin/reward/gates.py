"""Reward scoring: gates -> dense -> shape. Pure function for testability."""

from __future__ import annotations

import re

MIN_COMMANDS = 2

PARSE_FAIL_SCORE = -0.1
TOO_FEW_COMMANDS_SCORE = -0.2
BUILD_FAIL_SCORE = 0.0
TEST_FAIL_SCORE = 0.05

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
    # to reach dense + shape scoring below.
    gate_failures = (
        (not parse_ok, PARSE_FAIL_SCORE),
        (command_count < MIN_COMMANDS, TOO_FEW_COMMANDS_SCORE),
        (not build_ok, BUILD_FAIL_SCORE),
        (not test_ok, TEST_FAIL_SCORE),
    )
    for failed, score in gate_failures:
        if failed:
            return score
    reduction = max(0.0, (baseline_size - new_size) / max(1, baseline_size))
    dense = min(1.0, reduction)
    text = dockerfile_text.lower()
    shape = _shape_bonus(text) + _shape_penalty(text)
    return min(1.0, 0.5 + 0.5 * dense + shape)
