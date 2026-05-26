# Decision: prime-rl pin after PR #1392 status check

**Date:** 2026-05-22 (verified 2026-05-26)
**Context:** Task 1.3 of the implementation plan. prime-rl is the training entrypoint.
The open risk was PR #1392 (LoRA + NCCL `adapter_only` crash) — if unmerged, we would
have had to pin to a pre-LoRA-auto-config commit and pass explicit non-NCCL broadcast
args.

## Finding

PR #1392 "Fix auto-configure LoRA" is **MERGED**.

- Merge commit: `91182b7d647285a3e9e32f7959fdc3ff044d9330`
- Merged at: 2025-12-07
- `main` HEAD at verification time: `3f5ee350f636655f66509ee8f62e681d9555cd72`
  ("fix(sft): sync validation iteration to prevent FSDP deadlock (#2636)")
- `main` is 686 commits ahead of the fix — the `adapter_only` fix is safely included.

## Decision

Install prime-rl from `main`. The `adapter_only` bug is resolved, so no special
broadcast args are required for the LoRA path.

**Caveat:** `main` is 686 commits past the merge, so the codebase has churned heavily
since the plan was written (the current HEAD is an active-development SFT fix). Treat the
exact `git rev-parse HEAD` captured at install time on the pod as the real pin, and record
it below. If the alphabet_sort smoke (Task 1.4) or hotswap smoke (Task 1.5) fails on
bleeding-edge main, fall back to a tagged release rather than debugging upstream churn.

## Pin captured on pod

- prime-rl commit installed: `<fill on pod: git rev-parse HEAD>`
- vLLM version resolved: `<fill on pod: uv run python -c "import vllm; print(vllm.__version__)">`
- `uv run rl --help | grep -i lora` shows `--model.experimental.lora` / `--model.max_lora_rank`: `<yes/no>`
