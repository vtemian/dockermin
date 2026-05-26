"""Deterministic reward integration smoke — no model, no API key, real Docker.

Drives `dockermin_reward` through each gate outcome with hand-built completions so
the parse -> build -> test -> size-reward chain is exercised end-to-end against a live
Docker daemon. Cheap pre-pilot check that the reward plumbing actually scores real
builds the way the scoring unit tests (which mock the daemon) assume it does.

Run: python scripts/smoke_reward_replay.py   (needs a running Docker daemon)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dockermin.reward.dockermin_reward import dockermin_reward

_FENCE = "```dockerfile\n{body}\n```"


@dataclass(frozen=True)
class Case:
    name: str
    completion: str
    info: dict[str, object]
    expect: str  # human-readable assertion on the score


def _fenced(body: str) -> str:
    return _FENCE.format(body=body)


CASES = (
    Case(
        name="honest shrink (alpine vs 100MB baseline)",
        completion=_fenced('FROM alpine:3.20\nRUN apk add --no-cache coreutils\nCMD ["true"]'),
        info={"baseline_size": 100_000_000, "test_cmd": ["sh", "-c", "echo PYOK"], "expected_substring": "PYOK"},
        expect="high positive (build+test pass, ~87% smaller, alpine bonus)",
    ),
    Case(
        name="non-dockerfile output",
        completion="I'm sorry, I can't optimize that Dockerfile.",
        info={"baseline_size": 100_000_000, "test_cmd": ["sh", "-c", "echo PYOK"], "expected_substring": "PYOK"},
        expect="parse-fail penalty (-0.1)",
    ),
    Case(
        name="builds but test fails (wrong output)",
        completion=_fenced('FROM alpine:3.20\nRUN apk add --no-cache coreutils\nCMD ["true"]'),
        info={"baseline_size": 100_000_000, "test_cmd": ["sh", "-c", "echo NOPE"], "expected_substring": "PYOK"},
        expect="test-fail score (0.05)",
    ),
    Case(
        name="build failure (RUN exits non-zero)",
        completion=_fenced('FROM alpine:3.20\nRUN exit 1\nCMD ["true"]'),
        info={"baseline_size": 100_000_000, "test_cmd": ["sh", "-c", "echo PYOK"], "expected_substring": "PYOK"},
        expect="build-fail score (0.0)",
    ),
)


async def _score(case: Case) -> float:
    completion = [{"role": "assistant", "content": case.completion}]
    return await dockermin_reward(completion=completion, info=case.info)


def main() -> None:
    for case in CASES:
        score = asyncio.run(_score(case))
        print(f"{case.name:42s} score={score:+.3f}  expect: {case.expect}")


if __name__ == "__main__":
    main()
