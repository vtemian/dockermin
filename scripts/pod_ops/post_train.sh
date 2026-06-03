#!/bin/bash
# Pod-side watcher: waits for trainer to complete max_steps, pushes the final LoRA
# adapter to HF (as v2 — distinct from v1), then SELF-TERMINATES the pod.
# Watchdog backstop at WATCHDOG_SECONDS (default 16h) prevents runaway billing.
set -uo pipefail

POD_ID="${1:?usage: post_train.sh <POD_ID>}"
# Both HF_REPO and MAX_STEPS must be set explicitly per run. The previous v2
# defaults silently terminated v3 training at step 250 (v3 config wanted 500)
# AND pushed v3 adapters into the v2 HF repo, overwriting v2 weights —
# see docs/decisions/2026-06-03 for the postmortem.
HF_REPO="${HF_REPO:?HF_REPO env var required (e.g. vtemian/dockermin-qwen7b-lora-v3)}"
MAX_STEPS="${MAX_STEPS:?MAX_STEPS env var required (must match the run's config max_steps)}"
RUN_DIR="${RUN_DIR:-$HOME/prime-rl/outputs/run_default}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-86400}"  # 24h (A100 is ~2x slower than H100)

export PATH="$HOME/.local/bin:$PATH"
LOG="$HOME/post_train.log"

terminate() { prime pods terminate "$POD_ID" --yes >>"$LOG" 2>&1 || true; }
trap terminate EXIT
( sleep "$WATCHDOG_SECONDS"; echo "[$(date -u)] WATCHDOG fired" >>"$LOG"; terminate ) &

echo "[$(date -u)] post_train watcher start (pod=$POD_ID, target=step_$MAX_STEPS)" >>"$LOG"

# Wait for the trainer's final step OR a fatal error
while true; do
  LAST_STEP=$(grep -oE "Step [0-9]+ \|" "$HOME/trainer.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  LAST_STEP=${LAST_STEP:-0}
  if [ "$LAST_STEP" -ge "$MAX_STEPS" ]; then
    echo "[$(date -u)] reached step $LAST_STEP (>= $MAX_STEPS)" >>"$LOG"
    break
  fi
  # Fatal patterns — exit early to push whatever ckpt exists
  if grep -qE "ChildFailedError|CUDA out of memory|RuntimeError: All [0-9]+ rollouts" "$HOME/trainer.log" "$HOME/orchestrator.log" 2>/dev/null; then
    echo "[$(date -u)] fatal error detected; pushing whatever ckpt exists" >>"$LOG"
    break
  fi
  sleep 90
done

# Find the latest broadcast (LoRA adapter shards live here, marked STABLE).
LATEST_CKPT=$(ls -d "$RUN_DIR"/broadcasts/step_* 2>/dev/null | sort -V | tail -1)
# Only push if it has the STABLE marker (means trainer flushed it cleanly)
if [ -n "$LATEST_CKPT" ] && [ ! -f "$LATEST_CKPT/STABLE" ]; then
  echo "[$(date -u)] $LATEST_CKPT missing STABLE marker; backing off one" >>"$LOG"
  LATEST_CKPT=$(ls -d "$RUN_DIR"/broadcasts/step_* 2>/dev/null | sort -V | tail -2 | head -1)
fi
echo "[$(date -u)] LATEST_CKPT=$LATEST_CKPT" >>"$LOG"

if [ -z "$LATEST_CKPT" ]; then
  echo "[$(date -u)] NO checkpoint found; cannot push, terminating anyway" >>"$LOG"
  exit 0
fi

# Push adapter to HF (best-effort; do not block terminate)
"$HOME/prime-rl/.venv/bin/python" - "$LATEST_CKPT" "$HF_REPO" >>"$LOG" 2>&1 <<'PYEOF' || echo "[push failed]" >>"$LOG"
import sys, pathlib
ckpt, repo = sys.argv[1], sys.argv[2]
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
# Walk the checkpoint dir, upload each file
for f in pathlib.Path(ckpt).rglob("*"):
    if f.is_file():
        rel = f.relative_to(ckpt).as_posix()
        try:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rel, repo_id=repo, repo_type="model")
            print("pushed", rel)
        except Exception as e:
            print("push failed", rel, repr(e))
PYEOF

echo "[$(date -u)] post_train done; terminating pod" >>"$LOG"
