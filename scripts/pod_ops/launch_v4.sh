#!/bin/bash
# Pod-side launcher for the v4 GRPO arc (Gemma 4 12B-it base). Spawns inference,
# trainer, orchestrator as detached setsid processes; logs land in
# $HOME/{inference,trainer,orchestrator}.log. Re-runnable: kills any prior of
# these processes first.
#
# Plan: docs/plans/2026-06-09-v4-gemma-grpo.md
# Config: configs/dockermin_v4.toml
# Cost envelope: $46-71 for the 200-step arc inside the $80 cap.
set -uo pipefail

cd "$HOME/prime-rl"
export PATH="$HOME/.local/bin:$PATH"
export DOCKERMIN_MAX_BUILDS=4             # Vlad's cost-fix: lambdalabs 26vCPU > 20
export WANDB_MODE=offline                  # no wandb token staged; offline runs go to wandb/offline-run-*

# install_v4.sh runs `usermod -aG docker ubuntu`; the group is not active in
# this SSH session yet. `sg docker -c CMD` runs CMD in a subshell where the
# group is live, without re-login or chmoding the socket. The trainer and
# orchestrator subprocesses inherit the active group from this script's PID
# tree after the first sg-wrapped command is run as a no-op activator.
sg docker -c "true"

# Truncate ALL component logs upfront so monitors see only this attempt's content
: > "$HOME/inference.log"; : > "$HOME/trainer.log"; : > "$HOME/orchestrator.log"

# Generate sub-configs (idempotent; cleans stale rollouts/broadcasts)
uv run --no-sync rl @ "$HOME/dockermin/configs/dockermin_v4.toml" --dry-run

# Kill prior of our processes (idempotent re-launch). Avoid pkill self-match by matching
# argv with a distinctive substring.
pkill -f "prime_rl.trainer.rl.train" 2>/dev/null || true
pkill -f "outputs/configs/inference.toml" 2>/dev/null || true
pkill -f "outputs/configs/orchestrator.toml" 2>/dev/null || true
sleep 2

# 1. INFERENCE (GPU 0, 0.40 mem)
INF_LOG="$HOME/inference.log"
: > "$INF_LOG"
setsid env CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  nohup uv run --no-sync inference @ outputs/configs/inference.toml \
  --gpu-memory-utilization 0.40 --model.max-model-len 2560 \
  </dev/null >"$INF_LOG" 2>&1 &
INF_PID=$!
echo "inference pid=$INF_PID  log=$INF_LOG"

# 2. Wait for inference to be ready
echo "waiting for inference startup..."
for i in $(seq 1 120); do
  if grep -qE "Application startup complete|Uvicorn running on" "$INF_LOG" 2>/dev/null; then
    echo "  inference ready at attempt $i ($(date -u +%H:%M:%S))"
    break
  fi
  if ! kill -0 "$INF_PID" 2>/dev/null; then
    echo "  inference DIED before ready; last lines:"
    tail -25 "$INF_LOG"
    exit 1
  fi
  sleep 5
done

# 3. TRAINER (torchrun, GPU 0, expandable_segments)
TRN_LOG="$HOME/trainer.log"
: > "$TRN_LOG"
setsid env CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  nohup uv run --no-sync torchrun --nproc-per-node=1 \
  --rdzv-endpoint=localhost:29507 --rdzv-id=dockermin-v4 \
  -m prime_rl.trainer.rl.train @ outputs/configs/trainer.toml \
  </dev/null >"$TRN_LOG" 2>&1 &
TRN_PID=$!
echo "trainer pid=$TRN_PID  log=$TRN_LOG"

# 4. ORCHESTRATOR (CPU-side, with cost-fix concurrency)
ORC_LOG="$HOME/orchestrator.log"
: > "$ORC_LOG"
setsid env DOCKERMIN_MAX_BUILDS=4 WANDB_MODE=offline \
  nohup uv run --no-sync orchestrator @ outputs/configs/orchestrator.toml \
  </dev/null >"$ORC_LOG" 2>&1 &
ORC_PID=$!
echo "orchestrator pid=$ORC_PID  log=$ORC_LOG"

# Stash PIDs
echo "$INF_PID" > "$HOME/.inference.pid"
echo "$TRN_PID" > "$HOME/.trainer.pid"
echo "$ORC_PID" > "$HOME/.orchestrator.pid"

echo "=== v4 launched at $(date -u) ==="
echo "tail -f inference.log trainer.log orchestrator.log"
