"""Reward scoring: gates -> dense -> shape. Pure function for testability."""
from __future__ import annotations
import re

def compute_score(*, parse_ok: bool, build_ok: bool, test_ok: bool,
                  command_count: int, baseline_size: int, new_size: int,
                  dockerfile_text: str) -> float:
    if not parse_ok:
        return -0.1
    if command_count < 2:
        return -0.2
    if not build_ok:
        return 0.0
    if not test_ok:
        return 0.05
    reduction = max(0.0, (baseline_size - new_size) / max(1, baseline_size))
    dense = min(1.0, reduction)
    text = dockerfile_text.lower()
    shape = 0.0
    if re.search(r"from\s+\S*distroless", text): shape += 0.05
    if re.search(r"from\s+\S*alpine", text):    shape += 0.03
    if text.count("from ") >= 2:                shape += 0.05  # multi-stage
    if "rm -rf /var/lib/apt/lists" in text:     shape += 0.02
    if "--no-install-recommends" in text:       shape += 0.02
    # :latest tag penalty and bare-FROM (no tag) penalty: compound separately.
    # The original combined regex matched any single-token FROM line including
    # tagged ones because colons are non-whitespace.
    if ":latest" in text:
        shape -= 0.05
    if re.search(r"^from\s+[^\s:@]+\s*$", text, re.M):
        # bare image reference: no ":tag" and no "@digest" pin
        shape -= 0.05
    if "from scratch" in text and " copy " not in text:
        shape -= 0.10
    return 0.5 + 0.5 * dense + shape
