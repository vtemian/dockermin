"""Top-level reward function. Signature matches verifiers Rubric inspection."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dockermin.dataset.annotate import build_gate, parse_gate, run_test_gate

from .gates import compute_score
from .prompts import extract_dockerfile

# Verifiers runs every rollout on one event loop; a sync 300s docker build would
# block all concurrent rollouts. We offload the blocking docker calls to threads
# and cap concurrent builds with a semaphore acquired in the async layer (an
# asyncio.Semaphore is not thread-safe, so it must NOT be touched inside a thread).
_BUILD_SEM = asyncio.Semaphore(int(os.getenv("DOCKERMIN_MAX_BUILDS", "6")))


def _block_text(block: dict[str, Any]) -> str | None:
    """Text of one Anthropic-style content block, or None to skip it."""
    text = block.get("text") or block.get("content") or ""
    return text if isinstance(text, str) else None


def _message_parts(msg: dict[str, Any]) -> list[str]:
    """Text fragments contributed by one message; empty for skipped messages."""
    role = msg.get("role", "assistant")
    if role and role != "assistant":
        return []
    content = msg.get("content", "")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        # Anthropic-style content blocks: [{type:'text', text:'...'}, ...]
        block_texts = (_block_text(block) for block in content if isinstance(block, dict))
        return [t for t in block_texts if t is not None]
    return [str(content)]


def _completion_text(completion: str | list[dict[str, Any]]) -> str:
    """Verifiers passes completion either as str or list[message].

    Assumption: when given a list, we concatenate the ``content`` of every
    assistant-role message (and any dict without an explicit role) so that
    multi-block completions (e.g. assistant + tool_use + assistant) are
    surfaced to ``extract_dockerfile``. Non-string contents are stringified.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.extend(_message_parts(msg))
        return "\n".join(parts)
    return ""


async def dockermin_reward(  # noqa: PLR0911, PLR0915 — each return is a distinct gate outcome (malformed/parse/build/test); collapsing them would hide the score path
    completion: str | list[dict[str, Any]], info: dict[str, Any], **_kwargs: object
) -> float:
    """Composite reward. Async so the docker build does not block the rollout
    event loop. Signature accepts arbitrary kwargs per verifiers Rubric convention.

    Verifiers swallows reward exceptions to 0.0, so a malformed `info` row would
    silently look like a bad Dockerfile and pollute the gradient. We return a
    defined 0.0 for malformed input and score docker hiccups as build failures
    rather than ever raising.
    """
    try:
        text = _completion_text(completion)
        new_df = extract_dockerfile(text) or ""
        baseline_size = info["baseline_size"]
        test_cmd = info["test_cmd"]
        expected = info.get("expected_substring", "")
    except (KeyError, TypeError):
        return 0.0  # malformed sample: defined score, not a swallowed raise

    # parse_gate is pure and runs in microseconds; keep it inline.
    p = parse_gate(new_df)
    if not p.ok:
        return compute_score(
            parse_ok=False,
            build_ok=False,
            test_ok=False,
            command_count=0,
            baseline_size=baseline_size,
            new_size=0,
            dockerfile_text=new_df,
        )
    async with _BUILD_SEM:
        try:
            b = await asyncio.to_thread(build_gate, new_df, 300)
            if not b.ok:
                return compute_score(
                    parse_ok=True,
                    build_ok=False,
                    test_ok=False,
                    command_count=p.command_count,
                    baseline_size=baseline_size,
                    new_size=0,
                    dockerfile_text=new_df,
                )
            t = await asyncio.to_thread(run_test_gate, b.tag, test_cmd, expected, 30)
        except Exception:  # noqa: BLE001 - docker hiccup -> score as build failure, never crash the loop
            return compute_score(
                parse_ok=True,
                build_ok=False,
                test_ok=False,
                command_count=p.command_count,
                baseline_size=baseline_size,
                new_size=0,
                dockerfile_text=new_df,
            )
    return compute_score(
        parse_ok=True,
        build_ok=True,
        test_ok=t.ok,
        command_count=p.command_count,
        baseline_size=baseline_size,
        new_size=b.size_bytes,
        dockerfile_text=new_df,
    )
