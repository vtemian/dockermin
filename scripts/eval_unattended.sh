#!/bin/bash
# Unattended full eval on a GPU pod: run all baselines over the holdout, render
# the leaderboard, push artifacts to HF, then SELF-TERMINATE the pod (with a
# watchdog backstop). Designed to run detached (setsid) so an operator can walk
# away without leaving the pod billing idle after completion.
#
# Usage (on the pod, after setup):  bash eval_unattended.sh <POD_ID>
#
# Prereqs on the pod: prime-rl .venv with dockermin + eval deps installed,
# docker (DooD) up, hadolint/slim/claude on PATH, and these env vars exported:
#   OPENAI_API_KEY, ANTHROPIC_API_KEY (for gpt4o / sonnet / agent_loop),
#   HF token configured (for the results push), prime CLI authed (for terminate).
set -uo pipefail

POD_ID="${1:?usage: eval_unattended.sh <POD_ID>}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-21600}"  # 6h hard cap
HF_REPO="${HF_REPO:-vtemian/dockermin-qwen7b-lora-v1}"

cd "$HOME/prime-rl"
export PATH="$HOME/.local/bin:$PATH"
PY="$HOME/prime-rl/.venv/bin/python"
RUN_EVAL="$HOME/dockermin/scripts/run_eval.py"
LEADERBOARD="$HOME/dockermin/scripts/leaderboard.py"
OUT="$HOME/dockermin/data/eval/results.jsonl"
LB="$HOME/dockermin/data/eval/leaderboard.md"
LOG="$HOME/eval.log"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

terminate() { prime pods terminate "$POD_ID" --yes >>"$LOG" 2>&1 || true; }
# Always terminate on exit (success, error, or kill) + a watchdog for hangs.
trap terminate EXIT
( sleep "$WATCHDOG_SECONDS"; echo "[$(date -u)] WATCHDOG fired" >>"$LOG"; terminate ) &

echo "[$(date -u)] eval start (pod $POD_ID)" >>"$LOG"

# GPU split: model baselines each get their own process so the 7B handles do not
# coexist on the GPU. Non-GPU/API baselines run together first.
# slim (broken build-instrumentation on this pod) and agent_loop (a ~6h/$18
# build-fix loop that would blow the watchdog) are deferred to a focused run;
# sonnet_zs is the Sonnet bar tonight. qwen_zs + dockermin (both transformers)
# each get their own process so the two 7B models never coexist on the GPU.
"$PY" "$RUN_EVAL" --baselines hadolint manual gpt4o sonnet_zs --out "$OUT" >>"$LOG" 2>&1
"$PY" "$RUN_EVAL" --baselines qwen_zs --out "$OUT" >>"$LOG" 2>&1
"$PY" "$RUN_EVAL" --baselines dockermin --out "$OUT" >>"$LOG" 2>&1

echo "[$(date -u)] rendering leaderboard" >>"$LOG"
"$PY" "$LEADERBOARD" --in "$OUT" --out "$LB" >>"$LOG" 2>&1 || true

echo "[$(date -u)] pushing artifacts to HF $HF_REPO" >>"$LOG"
"$PY" - "$OUT" "$LB" "$HF_REPO" >>"$LOG" 2>&1 <<'PYEOF' || true
import sys
from huggingface_hub import HfApi
out, lb, repo = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
for local, remote in ((out, "eval/results.jsonl"), (lb, "eval/leaderboard.md")):
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="model")
        print("pushed", remote)
    except Exception as e:  # noqa: BLE001 - best effort; terminate must still run
        print("push failed", remote, repr(e))
PYEOF

echo "[$(date -u)] EVAL DONE — terminating pod" >>"$LOG"
# trap EXIT will run terminate; exit explicitly to trigger it now.
