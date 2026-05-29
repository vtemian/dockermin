#!/bin/bash
# Resume v1.1 GRPO from a checkpoint. Same env vars as launch_v1_1.sh but passes
# --ckpt.resume_step <N> through `rl --dry-run` so the generated sub-configs include the
# resume state.
set -uo pipefail

RESUME_STEP="${RESUME_STEP:-125}"
cd "$HOME/prime-rl"
export PATH="$HOME/.local/bin:$PATH"
export DOCKERMIN_MAX_BUILDS=12
export WANDB_MODE=offline

# Truncate logs so monitors see fresh content
: > "$HOME/inference.log"; : > "$HOME/trainer.log"; : > "$HOME/orchestrator.log"

# Dry-run with resume step (writes new outputs/configs/*.toml seeded for resume)
uv run --no-sync rl @ "$HOME/dockermin/configs/dockermin_full.toml" --dry-run \
  --ckpt.resume_step "$RESUME_STEP"

# Kill prior of our processes
pkill -f "prime_rl.trainer.rl.train" 2>/dev/null || true
pkill -f "outputs/configs/inference.toml" 2>/dev/null || true
pkill -f "outputs/configs/orchestrator.toml" 2>/dev/null || true
sleep 2

# 1. INFERENCE
INF_LOG="$HOME/inference.log"
: > "$INF_LOG"
setsid env PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0 \
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  nohup "$HOME/.local/bin/uv" run --no-sync inference @ outputs/configs/inference.toml \
  --gpu-memory-utilization 0.30 --model.max-model-len 5120 \
  </dev/null >"$INF_LOG" 2>&1 &
INF_PID=$!
echo "inference pid=$INF_PID  log=$INF_LOG"

# Wait for inference ready
for i in $(seq 1 120); do
  if grep -qE "Application startup complete|Uvicorn running on" "$INF_LOG" 2>/dev/null; then
    echo "  inference ready (attempt $i)"; break
  fi
  if ! kill -0 "$INF_PID" 2>/dev/null; then echo "  inference DIED"; tail -25 "$INF_LOG"; exit 1; fi
  sleep 5
done

# 2. TRAINER
TRN_LOG="$HOME/trainer.log"
: > "$TRN_LOG"
setsid env PATH="$HOME/.local/bin:$PATH" CUDA_VISIBLE_DEVICES=0 \
  VLLM_USE_DEEP_GEMM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  nohup "$HOME/.local/bin/uv" run --no-sync torchrun --nproc-per-node=1 \
  --rdzv-endpoint=localhost:29507 --rdzv-id=dockermin-v1-1-resume \
  -m prime_rl.trainer.rl.train @ outputs/configs/trainer.toml \
  </dev/null >"$TRN_LOG" 2>&1 &
echo "trainer pid=$!"

# 3. ORCHESTRATOR
ORC_LOG="$HOME/orchestrator.log"
: > "$ORC_LOG"
setsid env PATH="$HOME/.local/bin:$PATH" DOCKERMIN_MAX_BUILDS=12 WANDB_MODE=offline \
  nohup "$HOME/.local/bin/uv" run --no-sync orchestrator @ outputs/configs/orchestrator.toml \
  </dev/null >"$ORC_LOG" 2>&1 &
echo "orchestrator pid=$!"

echo "=== launched-with-resume=$RESUME_STEP at $(date -u) ==="
