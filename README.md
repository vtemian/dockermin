# dockermin

GRPO-fine-tuned Qwen 2.5 Coder 7B Instruct that rewrites working Dockerfiles into smaller functionally-equivalent ones.

**Status:** Active build, 3-weekend project starting 2026-05-23.
**Plan:** `/Users/whitemonk/projects/dd/plans/dockermin-rl-poc.md` (local, not in repo).
**Author:** Vlad Temian.

## What this is

A reinforcement-learning fine-tune of Qwen 2.5 Coder 7B Instruct using GRPO (Group Relative Policy Optimization) via Will Brown's `verifiers` library on Prime Intellect compute. The reward signal is built from real `docker build` + `docker run <test_cmd>` pairs: smaller working image wins, broken image scores zero.

## What this is not

- A static linter (see `hadolint` for that)
- A runtime-tracing minimizer (see `slimtoolkit/slim` for that, claims 5-30x reductions)
- A general code-rewriting model

## Planned deliverables

- HF adapter: `vladtemian/dockermin-qwen7b-lora-v1`
- HF dataset: `vladtemian/dockermin-v0` (Dockerfile + test_cmd + expected_output triples)
- `dockermin` CLI
- Benchmark suite + leaderboard vs 7 baselines (incl. Claude Sonnet 4.6 in agent loop)
- Blog post writeup
- `NEGATIVE_RESULT.md` if RL did not beat the agent-loop baseline (dataset+benchmark still ship)

## Honest disclosures

- This is a 3-weekend learning project with $400 compute budget. Not a product.
- Hard kill criteria are documented. Null result is a valid outcome and ships the dataset+benchmark as the contribution.
- The strong baseline (Claude Sonnet 4.6 via Claude Code CLI in 5-iteration build-fix loop) is included in eval specifically to see whether a 7B GRPO LoRA earns its training cost. Honest comparison.

## License

TBD (likely Apache 2.0 for code, dataset license per upstream Dockerfile sources).
