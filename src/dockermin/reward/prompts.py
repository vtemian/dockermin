"""Prompt template and Dockerfile extraction for Dockermin GRPO."""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a Dockerfile optimization engineer. Rewrite the given Dockerfile to be "
    "smaller while keeping it functionally equivalent. The rewritten image MUST remain "
    "functionally equivalent (same packages importable, same entrypoint). Output "
    "ONLY the new Dockerfile in a single fenced code block tagged ```dockerfile. Do not "
    "include any explanation, prose, or additional code blocks."
)

USER_TEMPLATE = (
    "Optimize this Dockerfile to be smaller while keeping it functionally "
    "equivalent - same runtime, same installed packages, same entrypoint "
    "behaviour.\n\n"
    "Original Dockerfile:\n```dockerfile\n{dockerfile}\n```\n\n"
    "Output the optimized Dockerfile only, in a single fenced ```dockerfile block."
)

_FENCE = re.compile(r"```(?:dockerfile|Dockerfile)?\s*\n(.*?)\n```", re.DOTALL)


def extract_dockerfile(text: str) -> str | None:
    """Return the first fenced ```dockerfile (or bare ```) block's body, or None."""
    m = _FENCE.search(text)
    return m.group(1).strip() if m else None


def format_messages(dockerfile: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(dockerfile=dockerfile.strip()),
        },
    ]
