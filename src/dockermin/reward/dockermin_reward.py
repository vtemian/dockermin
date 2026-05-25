"""Top-level reward function. Signature matches verifiers Rubric inspection."""

from __future__ import annotations

from typing import Any

from dockermin.dataset.annotate import build_gate, parse_gate, run_test_gate

from .gates import compute_score
from .prompts import extract_dockerfile


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


def dockermin_reward(completion: str | list[dict[str, Any]], info: dict[str, Any], **_kwargs: object) -> float:
    """Composite reward. Signature accepts arbitrary kwargs per verifiers Rubric convention."""
    text = _completion_text(completion)
    new_df = extract_dockerfile(text) or ""
    p = parse_gate(new_df)
    if not p.ok:
        return compute_score(
            parse_ok=False,
            build_ok=False,
            test_ok=False,
            command_count=0,
            baseline_size=info["baseline_size"],
            new_size=0,
            dockerfile_text=new_df,
        )
    b = build_gate(new_df, timeout_s=300)
    if not b.ok:
        return compute_score(
            parse_ok=True,
            build_ok=False,
            test_ok=False,
            command_count=p.command_count,
            baseline_size=info["baseline_size"],
            new_size=0,
            dockerfile_text=new_df,
        )
    t = run_test_gate(b.tag, info["test_cmd"], info.get("expected_substring", ""), timeout_s=30)
    return compute_score(
        parse_ok=True,
        build_ok=True,
        test_ok=t.ok,
        command_count=p.command_count,
        baseline_size=info["baseline_size"],
        new_size=b.size_bytes,
        dockerfile_text=new_df,
    )
