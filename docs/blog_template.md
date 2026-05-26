# Dockermin: RL-fine-tuning Qwen 2.5 Coder 7B to Shrink Dockerfiles

> Status: [DRAFT / SHIPPED / NULL RESULT - PICK ONE]
> Date: 2026-06-XX
> Compute: ~$XXX on Prime Intellect
> Code: github.com/vtemian/dockermin
> Adapter: huggingface.co/vtemian/dockermin-qwen7b-lora-v1
> Dataset: huggingface.co/datasets/vtemian/dockermin-v0

## TL;DR

[One paragraph: what you built, headline number, honest comparison vs baselines.]

## The question

Can a 7B code model fine-tuned with GRPO match a Sonnet 4.6 agent loop on Dockerfile minimization? The honest answer matters because if a $200/mo Claude Code subscription beats a custom 7B LoRA, the RL training story is hard to sell.

## The setup

- Base model: Qwen 2.5 Coder 7B Instruct
- Training: GRPO via Will Brown's verifiers library on prime-rl, Prime Intellect on-demand H100s
- Reward: gates (parse + build + test) then dense size-reduction signal then shape bonuses
- Dataset: 16 real Dockerfiles + 48 Claude-Sonnet-4.6 unoptimized variants. Total ~62 working triples.
- Eval: 7 baselines including Claude Code agent loop (the load-bearing comparison)

## What the agent-loop baseline does

[Describe the bare-Sonnet-4.6 + claude -p + --max-budget-usd setup verbatim from src/dockermin/eval/agent_loop.py.]

## Results

[Insert leaderboard table from docs/leaderboard.md once eval runs.]

| Baseline | Mean reduction | % test pass | Cost |
|---|---|---|---|
| ... | ... | ... | ... |

## Things that surprised us

[3-5 sentences. Reward hacking patterns? Failures of bigger models? Surprises about base-image semantics?]

## Reward hacking we caught

[The empty expected_substring trick. The :latest tag exploit. Etc. From audit_rollouts.py logs.]

## What did not work

- prime-rl PR #1392 had to be merged before LoRA + NCCL would not crash
- The dockerfile python lib is upstream-archived
- Synthetic variants from official-images failed when bloat patterns pushed apt-get past Debian repo dep edges
- [...]

## What we would do next

- Larger dataset (100+ working triples, more ecosystem coverage)
- Multi-step agent training instead of single-shot rewrite
- Open-weight comparison: how does this stack against a competently fine-tuned 14B?

## Acknowledgements

Will Brown for verifiers, Prime Intellect Lab Hosted Training, the official-images team, the SlimToolkit maintainers for the strongest mechanical baseline.
