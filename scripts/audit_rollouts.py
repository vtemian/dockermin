"""Pull 20 random rollouts from the latest checkpoint, eyeball reward-hacking patterns."""
from __future__ import annotations
import random, re, json
from pathlib import Path

def audit(rollouts_dir: Path) -> dict:
    files = list(rollouts_dir.glob("*.jsonl"))
    rollouts = []
    for f in files:
        for line in f.read_text().splitlines():
            try: rollouts.append(json.loads(line))
            except: continue
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
    stats = audit(Path(sys.argv[1] if len(sys.argv)>1 else "rollouts/"))
    print(json.dumps(stats, indent=2))
