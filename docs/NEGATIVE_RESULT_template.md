# Dockermin: Null Result

> Status: NULL RESULT
> Date: 2026-06-XX
> Compute spent: ~$XXX
> Code: github.com/vtemian/dockermin
> Dataset: huggingface.co/datasets/vladtemian/dockermin-v0 (DOES ship)

## The honest claim

A GRPO-fine-tuned Qwen 2.5 Coder 7B LoRA did NOT meaningfully beat a Claude Sonnet 4.6 agent-loop baseline on Dockerfile minimization across our 150-Dockerfile holdout. The RL contribution argument does not hold for this task at this model scale.

## What we shipped instead

- Dataset (`vladtemian/dockermin-v0`): 16 real Dockerfiles + 48 synthetic variants with functional test triples (parse + build + test verified).
- Benchmark suite: 7 baselines including agent loop, all with reproducible run scripts.
- Leaderboard: see docs/leaderboard.md.
- Recipe for the failure: see below.

## Why the RL did not pay off

[Hypotheses, ordered by what evidence supports]:
1. Dataset too small (62 triples) - LoRA had insufficient signal to generalize past the synthetic-variant distribution
2. Reward function had reward-hacking surface despite gates
3. Single-shot rewrite is harder than multi-step agent loop where the model can read build errors
4. Base model's Dockerfile knowledge already very strong - GRPO mostly reinforced existing tendencies

## What's still useful here

- The synthetic-variants generation pipeline (scripts/synthetic_variants.py): reusable for any code-rewriting RL setup
- The verifiers Environment package: reference implementation for "real docker build + test in reward"
- The honest comparison methodology: agent-loop vs LoRA as the same task

## Citation / Disclosure

Learning project, not a benchmark contribution. 3-weekend timeline, ~$XXX in compute. We did not iterate the dataset to ship-quality.
