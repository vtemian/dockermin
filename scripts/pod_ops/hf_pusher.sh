#!/bin/bash
# Push every new STABLE LoRA broadcast to HF as soon as it appears, so a pod death
# doesn't lose the latest converged adapter. Idempotent: tracks already-pushed
# broadcasts in $HOME/.hf_pushed.
set -uo pipefail
# HF_REPO must be set explicitly per run (v2 / v3 / v3-ablation-a / ...). The
# previous default `vtemian/dockermin-qwen7b-lora-v2` silently overwrote v2
# adapters during the v3 training session — see docs/decisions/2026-06-03
# for the postmortem. Fail loudly if the caller forgot.
HF_REPO="${HF_REPO:?HF_REPO env var required - e.g. vtemian/dockermin-qwen7b-lora-v3}"
RUN_DIR="${RUN_DIR:-$HOME/prime-rl/outputs/run_default}"
LOG="$HOME/hf_pusher.log"
PUSHED_FILE="$HOME/.hf_pushed"
touch "$PUSHED_FILE"

push_one() {
  local ckpt="$1"
  local step="$2"
  "$HOME/prime-rl/.venv/bin/python" - "$ckpt" "$HF_REPO" "$step" <<'PYEOF'
import sys, pathlib
ckpt, repo, step = sys.argv[1], sys.argv[2], sys.argv[3]
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
for f in pathlib.Path(ckpt).rglob("*"):
    if f.is_file():
        rel = f"step_{step}/" + f.relative_to(ckpt).as_posix()
        try:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=rel, repo_id=repo, repo_type="model")
            print("pushed", rel)
        except Exception as e:
            print("push failed", rel, repr(e))
PYEOF
}

while true; do
  TS=$(date -u +%FT%TZ)
  # Iterate over every STABLE broadcast we haven't pushed yet
  for ckpt in "$RUN_DIR"/broadcasts/step_*; do
    [ -d "$ckpt" ] || continue
    [ -f "$ckpt/STABLE" ] || continue
    step=$(basename "$ckpt" | sed 's/step_//')
    if grep -qx "$step" "$PUSHED_FILE" 2>/dev/null; then continue; fi
    echo "[$TS] pushing step_$step" >>"$LOG"
    push_one "$ckpt" "$step" >>"$LOG" 2>&1 && echo "$step" >> "$PUSHED_FILE"
  done
  sleep 300
done
