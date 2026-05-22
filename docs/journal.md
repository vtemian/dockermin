# Project journal

Append per session. Last entry at top. 3-5 sentences per entry. What worked, what broke, what was not obvious.

## 2026-05-22 Friday late evening (Vlad + Bot, parallel implementation burst)

Dispatched 6 parallel subagents on non-overlapping file sets. All 6 returned clean (AST + TOML validation passed). 2232 lines of code across 29 new files written in roughly 4 minutes of wall-clock. Commits f5d8d9b through 497ad0a cover the reward stack, annotate gates, dataset orchestration, pod ops scripts, eval baselines, and CLI + prime-rl environment package.

Real bugs caught by the agents and fixed:
- Plan's `smoke_lora_hotswap.py` had a NameError (`base` vs `base_out`). Fixed.
- Plan's `configs/dockermin_pilot.toml` was invalid TOML (`model` declared as both string and table). Agent promoted to `[model]` table with `name=` field; prime-rl schema TBD-verified Saturday against the alphabet_sort example.
- Plan's `Candidate.__dict__` would fail on frozen dataclass; agent used `dataclasses.asdict`.
- Plan's run_annotate early-break was unreachable; agent restructured to break on as_completed.

Open verifications for Saturday:
- prime-rl entry-point group name (`verifiers.environments` is the educated guess) - probe with `vf-install --help` and `importlib.metadata.entry_points` on the pod
- TOML schema for prime-rl model config - read examples/alphabet_sort/rl.toml on the pod before first run
- `dockerfile.GoParseError` symbol stability (lib is upstream-deprecated)
- `dockermin_env` pyproject may need explicit `packages = ["dockermin_env"]` if setuptools auto-discovery balks

29 files in but ZERO tests have actually run yet - they will run on the pod Saturday in real conditions. This is the right tradeoff for tonight: write code at parallel speed, verify in real conditions tomorrow.

## 2026-05-22 Friday evening (Vlad + Bot, pre-execution)

Plan read. Memories saved. Repo initialized at github.com/vtemian/dockermin with skeleton + pinned deps. Decisions locked in: GRPO via verifiers committed, no mid-project pivot to SFT/steering even on null result, agent-loop baseline via Claude Code CLI, dataset target 200 not 1000, egress allowlist proxy not single mirror, DooD privileged daemon with pids-limit 2048 and 300s watchdog. Top risks flagged: build latency dominates rollout, docker daemon concurrency bottleneck unverified, Sonnet 4.6 agent loop likely strong, vllm 0.7.3 LoRA hotswap on Qwen 7B unverified. Saturday runbook drafted.
