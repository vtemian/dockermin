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

## Pin captured on pod (2026-05-26, massedcompute 1xH100 PCIe)

- prime-rl commit installed: `3f5ee350f636655f66509ee8f62e681d9555cd72`
- Python `3.12.13` (uv fetched it; pod system Python is 3.10), torch `2.11.0+cu128`,
  vLLM `0.21.0+cu129.r42434.pr39568.a106aa6` (prime custom build), verifiers `0.1.15.dev9`.
- The old plan assumptions are all stale: NOT vllm 0.7.3, NOT verifiers 0.1.4, NOT the
  `--model.experimental.lora` flags. LoRA is configured in the TOML under
  `[trainer.model.lora] rank/alpha`.

## Install recipe that worked

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
git config --global url."https://github.com/".insteadOf "git@github.com:"   # submodules use ssh URLs that fail on the pod
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
uv sync --all-extras   # ~15 min: pulls py3.12, torch 2.11, vllm 0.21, flash-attn
```

## Single-GPU RL run recipe + the three gotchas it took to get a passing smoke

`rl` (the unified launcher) CANNOT colocate on one GPU in this commit: it assigns
inference→GPU0, trainer→GPU1 by count and errors with "Requested 2 GPUs ... only 1
available". The docs' `--trainer-gpu-ids/--inference-gpu-ids` flags do NOT exist in this
commit (docs are ahead of code). prime-rl has no stable release tags (only dev tags), so
fixing forward on main is the norm — there is no cleaner tag to fall back to.

Single-GPU = run the three components manually, colocated on GPU 0:

```bash
# 0. Generate sub-configs from the example
uv run --no-sync rl @ examples/alphabet_sort/rl.toml --dry-run   # writes outputs/configs/{inference,orchestrator,trainer}.toml

# 1. Inference server (GPU 0, half the memory)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  uv run --no-sync inference @ outputs/configs/inference.toml \
  --gpu-memory-utilization 0.5 --model.max-model-len 8192

# 2. Trainer (GPU 0, via torchrun — NOT the bare `trainer` entrypoint)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync torchrun --nproc-per-node=1 --rdzv-endpoint=localhost:29507 --rdzv-id=smoke \
  -m prime_rl.trainer.rl.train @ outputs/configs/trainer.toml

# 3. Orchestrator (CPU-side, no GPU)
uv run --no-sync orchestrator @ outputs/configs/orchestrator.toml
```

Gotchas, each of which crashed a component until fixed:
1. **deep_gemm / `VLLM_USE_DEEP_GEMM=0`** — this vllm build defaults `VLLM_USE_DEEP_GEMM=True`,
   but the shipped deep_gemm wheel needs `libcudart.so.13` (CUDA 13) while the stack is
   CUDA 12.8/12.9 on the `ubuntu_22_cuda_12` image → `RuntimeError: DeepGEMM backend not
   available`. Disable it; bf16 non-MoE models don't need it.
2. **`--model.max-model-len 8192`** — Qwen3-4B advertises a 256K context; at 0.5 mem util
   vllm wants 36 GiB of KV cache but has ~27 GiB → startup ValueError. Cap it (task only
   uses ~2-3K tokens).
3. **trainer needs torchrun** — bare `uv run trainer` dies with `RANK expected, but not
   set`. Launch as `torchrun --nproc-per-node=1 -m prime_rl.trainer.rl.train`.

Weight broadcast type is `filesystem` (trainer writes weights to disk, inference reloads),
so colocation needs no GPU↔GPU NCCL. Colocated footprint for the 4B smoke: ~69/81 GiB
(inference 40 + trainer ~28), no OOM.

**Implication for the real run:** the 8xH100 run will use the default split deployment
(`rl` with inference on some GPUs, trainer on others) — that path does NOT need the
single-GPU colocation hacks, but it WILL still need `VLLM_USE_DEEP_GEMM=0` (unless we
install a CUDA-13-matched deep_gemm) and a sane `--model.max-model-len` for Qwen 7B.
