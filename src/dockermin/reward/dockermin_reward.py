"""Top-level reward function. Signature matches verifiers Rubric inspection."""
from __future__ import annotations
from .prompts import extract_dockerfile
from .gates import compute_score
from dockermin.dataset.annotate import parse_gate, build_gate, run_test_gate


def _completion_text(completion) -> str:
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
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "assistant")
            if role and role != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Anthropic-style content blocks: [{type:'text', text:'...'}, ...]
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content") or ""
                        if isinstance(text, str):
                            parts.append(text)
            else:
                parts.append(str(content))
        return "\n".join(parts)
    return ""


def dockermin_reward(completion, info, **kwargs) -> float:
    """Composite reward. Signature accepts arbitrary kwargs per verifiers Rubric convention."""
    text = _completion_text(completion)
    new_df = extract_dockerfile(text) or ""
    p = parse_gate(new_df)
    if not p.ok:
        return compute_score(parse_ok=False, build_ok=False, test_ok=False,
                             command_count=0, baseline_size=info["baseline_size"],
                             new_size=0, dockerfile_text=new_df)
    b = build_gate(new_df, timeout_s=300)
    if not b.ok:
        return compute_score(parse_ok=True, build_ok=False, test_ok=False,
                             command_count=p.command_count,
                             baseline_size=info["baseline_size"],
                             new_size=0, dockerfile_text=new_df)
    t = run_test_gate(b.tag, info["test_cmd"], info.get("expected_substring", ""), timeout_s=30)
    return compute_score(
        parse_ok=True, build_ok=True, test_ok=t.ok,
        command_count=p.command_count,
        baseline_size=info["baseline_size"], new_size=b.size_bytes,
        dockerfile_text=new_df,
    )
