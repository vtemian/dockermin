"""Pull 20 random rollouts from the latest checkpoint, eyeball reward-hacking patterns."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dockermin.ops import read_jsonl
from dockermin.reward.dockermin_reward import _completion_text
from dockermin.reward.prompts import extract_dockerfile


def audit(rollouts_dir: Path) -> dict:
    rollouts = []
    for f in rollouts_dir.glob("*.jsonl"):
        rollouts.extend(read_jsonl(f))
    sample = random.sample(rollouts, min(20, len(rollouts)))
    # prime-rl emits completions as list[AssistantMessage]; the reward path unwraps via
    # _completion_text + extract_dockerfile, and the audit must mirror that or every
    # substring check silently misses (str(list) noise instead of dockerfile text).
    texts = []
    for r in sample:
        extracted = extract_dockerfile(_completion_text(r.get("completion", ""))) or ""
        texts.append(extracted.lower())
    stats = {
        "n": len(sample),
        "from_scratch": sum(1 for t in texts if "from scratch" in t),
        "latest_tag": sum(1 for t in texts if ":latest" in t),
        "no_cmd": sum(1 for t in texts if "cmd" not in t and "entrypoint" not in t),
        "mean_lines": sum(len(t.splitlines()) for t in texts) / max(1, len(sample)),
    }
    return stats


if __name__ == "__main__":
    import sys

    stats = audit(Path(sys.argv[1] if len(sys.argv) > 1 else "rollouts/"))
    print(json.dumps(stats, indent=2))
