# Project journal

Append per session. Last entry at top. 3-5 sentences per entry. What worked, what broke, what was not obvious.

## 2026-05-22 Friday evening (Vlad + Bot, pre-execution)

Plan read. Memories saved. Repo initialized at github.com/vtemian/dockermin with skeleton + pinned deps. Decisions locked in: GRPO via verifiers committed, no mid-project pivot to SFT/steering even on null result, agent-loop baseline via Claude Code CLI, dataset target 200 not 1000, egress allowlist proxy not single mirror, DooD privileged daemon with pids-limit 2048 and 300s watchdog. Top risks flagged: build latency dominates rollout, docker daemon concurrency bottleneck unverified, Sonnet 4.6 agent loop likely strong, vllm 0.7.3 LoRA hotswap on Qwen 7B unverified. Saturday runbook drafted.
