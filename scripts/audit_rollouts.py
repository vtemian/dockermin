"""Pull 20 random rollouts from the latest checkpoint, eyeball reward-hacking patterns."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dockermin.ops import read_jsonl


def audit(rollouts_dir: Path) -> dict:
    rollouts = []
    for f in rollouts_dir.glob("*.jsonl"):
        rollouts.extend(read_jsonl(f))
    sample = random.sample(rollouts, min(20, len(rollouts)))
    # Normalize to lowercase once so substring checks are case-insensitive
    # (catches ":LATEST", ":Latest", "Cmd", "EntryPoint", etc.).
    texts = [r.get("completion", "").lower() for r in sample]
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
