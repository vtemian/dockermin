#!/bin/bash
# Pod-side runner for the Gemma 4 12B-it zero-shot probe.
#
# Pre-warms the docker cache from the 37-row holdout's FROM lines, runs the
# eval, and pushes results to HF every 60s so a pod death loses at most ~60s
# of work. Self-terminates the pod on completion or after a 4h watchdog.
#
# Outputs:
#   $HOME/dockermin/data/eval/gemma_zs_probe.jsonl  (local)
#   vtemian/dockermin-gemma4-12b-it-probe/eval/results.jsonl  (HF)
set -uo pipefail
exec > >(tee -a "$HOME/probe.log") 2>&1

POD_ID="${1:?usage: run_gemma_probe.sh <POD_ID>}"
HF_REPO_OUT="vtemian/dockermin-gemma4-12b-it-probe"

cd "$HOME/dockermin"
# shellcheck disable=SC1091
source "$HOME/gemma-probe-venv/bin/activate"

# Pod self-termination is optional: the prime CLI is not installed in the
# probe venv. If it happens to be on PATH (e.g. from a different venv),
# call it; otherwise rely on the operator's manual terminate from outside.
# Either way the eval will write its final HF push before exiting.
terminate() {
    if command -v prime >/dev/null 2>&1; then
        prime pods terminate "$POD_ID" --yes 2>&1 | head -1 || true
    else
        echo "[$(date -u)] prime CLI not on pod — operator must terminate $POD_ID externally"
    fi
}
trap terminate EXIT
( sleep 14400; echo "[$(date -u)] WATCHDOG 4h"; terminate ) &

# Background pusher: every 60s, push results.jsonl to HF if it exists.
nohup bash -c "
while true; do
  if [ -f '$HOME/dockermin/data/eval/gemma_zs_probe.jsonl' ]; then
    python -c '
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj=\"$HOME/dockermin/data/eval/gemma_zs_probe.jsonl\",
    path_in_repo=\"eval/results.jsonl\",
    repo_id=\"$HF_REPO_OUT\",
    repo_type=\"model\",
)
print(\"pushed\")
' || true
  fi
  sleep 60
done
" >/dev/null 2>&1 &

# Pre-warm the docker cache from the 37-row holdout's FROM tags. A row whose
# base fails to pull will still fail at build-gate but won't artificially
# disadvantage Gemma vs the prior Qwen evals run on different pods.
python - <<'PYEOF'
from datasets import load_dataset
import re, subprocess
ds = load_dataset("vtemian/dockermin-v0", split="test")
bases = set()
for ex in ds:
    for m in re.finditer(r"^FROM\s+(\S+)", ex["dockerfile"], re.I | re.M):
        tag = m.group(1)
        if tag.lower() != "scratch" and ":" in tag:
            bases.add(tag)
print(f"pre-warming {len(bases)} unique base images from holdout")
for b in sorted(bases):
    try:
        r = subprocess.run(["docker", "pull", b], capture_output=True, timeout=300)
        print(f"  {'OK' if r.returncode == 0 else 'SKIP'} {b}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {b}")
PYEOF

# Pre-warm common rewrite targets so build-gate cache hits match the prior
# controlled re-eval pod (same provider, same warm-cache assumptions).
for img in python:3.9-slim python:3.12-slim python:3.12-slim-bookworm \
           eclipse-temurin:25-jdk-resolute eclipse-temurin:25-jdk-noble \
           php:8.5-apache-trixie node:14-slim node:14-buster-slim; do
    docker pull "$img" >/dev/null 2>&1 && echo "OK $img (rewrite prewarm)" || echo "SKIP $img"
done

# Run the eval — same temperature / max_new_tokens as the v2 / v3 baselines
# so the comparison is apples-to-apples.
mkdir -p "$HOME/dockermin/data/eval"
python scripts/run_eval.py \
    --baselines gemma_zs \
    --holdout vtemian/dockermin-v0 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --out "$HOME/dockermin/data/eval/gemma_zs_probe.jsonl"

# Final push (in case the 60s loop missed the last row).
python - <<PYEOF
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="$HOME/dockermin/data/eval/gemma_zs_probe.jsonl",
    path_in_repo="eval/results.jsonl",
    repo_id="$HF_REPO_OUT",
    repo_type="model",
)
print("final push OK")
PYEOF

echo "[$(date -u)] === GEMMA_ZS PROBE DONE — terminating pod ==="
