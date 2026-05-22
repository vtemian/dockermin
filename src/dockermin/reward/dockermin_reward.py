"""Top-level reward function. Signature matches verifiers Rubric inspection."""
from __future__ import annotations
from .prompts import extract_dockerfile
from .gates import compute_score
from dockermin.dataset.annotate import parse_gate, build_gate, run_test_gate


def _completion_text(completion) -> str:
    """Verifiers passes completion either as str or list[message]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
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
