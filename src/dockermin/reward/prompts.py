"""Prompt template and Dockerfile extraction for Dockermin GRPO."""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are a Dockerfile optimization engineer. Rewrite the given Dockerfile to be "
    "smaller while keeping it functionally equivalent. The rewritten image MUST still "
    "pass the provided test command and produce the expected output substring. Output "
    "ONLY the new Dockerfile in a single fenced code block tagged ```dockerfile. Do not "
    "include any explanation, prose, or additional code blocks."
)

USER_TEMPLATE = (
    "Optimize this Dockerfile.\n\n"
    "Original Dockerfile:\n```dockerfile\n{dockerfile}\n```\n\n"
    "Test command (run inside the built image): {test_cmd}\n"
    "Expected output substring: {expected}\n\n"
    "Output the optimized Dockerfile only."
)

_FENCE = re.compile(r"```(?:dockerfile|Dockerfile)?\s*\n(.*?)\n```", re.DOTALL)

def extract_dockerfile(text: str) -> str | None:
    """Return the first fenced ```dockerfile (or bare ```) block's body, or None."""
    m = _FENCE.search(text)
    return m.group(1).strip() if m else None

def format_messages(dockerfile: str, test_cmd: list[str], expected: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            dockerfile=dockerfile.strip(),
            test_cmd=" ".join(test_cmd),
            expected=expected,
        )},
    ]
