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
    stats = {
        "n": len(sample),
        "from_scratch": sum(1 for r in sample if "from scratch" in r.get("completion","").lower()),
        "latest_tag": sum(1 for r in sample if ":latest" in r.get("completion","")),
        "no_cmd": sum(1 for r in sample if "CMD" not in r.get("completion","") and "ENTRYPOINT" not in r.get("completion","")),
        "mean_lines": sum(len(r.get("completion","").splitlines()) for r in sample) / max(1, len(sample)),
    }
    return stats

if __name__ == "__main__":
    import sys
    stats = audit(Path(sys.argv[1] if len(sys.argv)>1 else "rollouts/"))
    print(json.dumps(stats, indent=2))
