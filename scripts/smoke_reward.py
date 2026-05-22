"""Run base Qwen 2.5 Coder 7B Instruct (proxied via Claude Haiku) on 10 dataset entries.

Sends each prompt through the formatted messages, asks the model to emit an optimized
Dockerfile, then scores it with the same dockermin_reward used by GRPO. This is a cheap
end-to-end smoke test before paying for GPU rollouts.
"""
from __future__ import annotations

import json
from pathlib import Path

from anthropic import Anthropic

from dockermin.reward.dockermin_reward import dockermin_reward
from dockermin.reward.prompts import format_messages

TRIPLES_PATH = Path("data/curated/triples.jsonl")
MODEL = "claude-haiku-4-5-20251001"
SAMPLE_N = 10
MAX_TOKENS = 1024


def main() -> None:
    triples = [
        json.loads(line) for line in TRIPLES_PATH.read_text().splitlines() if line.strip()
    ]
    triples = triples[:SAMPLE_N]
    client = Anthropic()

    for t in triples:
        msgs = format_messages(t["dockerfile"], t["test_cmd"], t["expected_substring"])
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=msgs[0]["content"],
            messages=[msgs[1]],
        )
        completion_text = response.content[0].text
        completion = [{"role": "assistant", "content": completion_text}]
        score = dockermin_reward(
            completion=completion,
            info={
                "baseline_size": t["baseline_size"],
                "test_cmd": t["test_cmd"],
                "expected_substring": t["expected_substring"],
            },
        )
        baseline_mb = t["baseline_size"] / 1e6
        tag = str(t.get("id", ""))[-30:]
        print(f"id={tag:30s} score={score:+.3f} size={baseline_mb:.1f}MB")


if __name__ == "__main__":
    main()
