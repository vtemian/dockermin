#!/bin/bash
# Pod-side installer for the v4 GRPO trainer pod (google/gemma-4-12B-it).
#
# Variant of scripts/pod_ops/install.sh that bakes in the 8 gotchas from
# docs/decisions/2026-06-08-gemma-zs-probe-result.md and vendors the prime-rl
# @ main bump (Gemma 4 needs transformers >= 5.10.2; project pin updated).
#
# Idempotent — safe to re-run.
set -euo pipefail
exec > >(tee -a "$HOME/install.log") 2>&1
echo "=== $(date -u) install_v4 start ==="

: "${HF_TOKEN:?HF_TOKEN must be set on the pod}"

# Docker group access (gotcha #6: usermod -aG, NOT chmod 666 on the socket).
# launch_v4.sh uses sg docker -c for socket access; no chmod 666 (local
# privilege-escalation hole on a multi-tenant-style pod).
sudo usermod -aG docker ubuntu || true

# Pod-survival sysctls: dockermin's per-rollout builds create overlayfs mounts
# that linger; three prior pods died with mount/disk exhaustion patterns.
echo "fs.mount-max=1000000" | sudo tee /etc/sysctl.d/99-dockermin.conf >/dev/null
sudo sysctl --system >/dev/null 2>&1 || true

# uv (skip if already present). `|| true` because uv's installer chmods
# /home/ubuntu/.config/fish on some images and aborts on Permission denied even
# though uv itself installed fine.
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv install failed; aborting"; exit 1; }

# Gotcha #1: massedcompute ubuntu_22_cuda_12 ships python3.10 only, but
# prime-rl @ main requires ~=3.12.0. Force a uv-managed 3.12 toolchain.
uv python install 3.12

# prime-rl
if [ ! -d "$HOME/prime-rl" ]; then
    cd "$HOME"
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git
    cd prime-rl
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
fi
cd "$HOME/prime-rl"
# Explicit pin to match pyproject.toml — main HEAD as of the v4 bump. Without
# this checkout the clone follows upstream main and silently drifts past the
# transformers/vllm versions the v4 stack was tested against.
git fetch origin
git checkout d5ce4ce84d6b1830bb3d33636a01e1cad17a8971
PRIME_RL_HEAD=$(git rev-parse HEAD)
echo "=== prime-rl HEAD: $PRIME_RL_HEAD ==="

# uv sync (the slow step, ~10-15 min). --python 3.12 forces the 3.12 toolchain
# we installed above; prime-rl main rejects 3.10/3.11.
echo "=== $(date -u) uv sync start ==="
uv sync --all-extras --python 3.12
echo "=== $(date -u) uv sync done ==="

# Gotcha #3: verify transformers >= 5.10.2 actually resolved. Anything below
# rejects Gemma 4's model_type=gemma4_unified at config load.
RESOLVED_TFM=$(uv run --no-sync python -c "import transformers; print(transformers.__version__)")
echo "resolved transformers=$RESOLVED_TFM"
uv run --no-sync python -c "import transformers, sys; v = transformers.__version__; sys.exit(0 if tuple(map(int, v.split('.')[:3])) >= (5, 10, 2) else 1)" \
    || { echo "transformers < 5.10.2 — Gemma 4 will not load; aborting"; exit 1; }

# vLLM version is informational only — known issue: <= 0.22.1 lacks
# Gemma4UnifiedForConditionalGeneration and falls back to the transformers
# backend. The Phase 3 smoke test catches the crash if it manifests.
RESOLVED_VLLM=$(uv run --no-sync python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "missing")
echo "resolved vllm=$RESOLVED_VLLM"

# dockermin (no deps — already pulled by prime-rl's environment). DOCKERMIN_REV
# pins what runs on the pod; default main for reproducibility. For the v4
# launch, the operator passes DOCKERMIN_REV=feat/v4-gemma-grpo until the PR
# merges. The script never silently follows a moving branch tip.
if [ ! -d "$HOME/dockermin" ]; then
    cd "$HOME"
    git clone https://github.com/vtemian/dockermin.git
fi
cd "$HOME/dockermin"
git fetch origin --tags
DOCKERMIN_REV="${DOCKERMIN_REV:-main}"
git checkout "$DOCKERMIN_REV"
# Only fast-forward if the rev is a branch (not a tag or SHA, which are immutable).
if git symbolic-ref -q HEAD >/dev/null; then
    git pull --ff-only
fi
DOCKERMIN_HEAD=$(git rev-parse HEAD)
echo "=== dockermin HEAD: $DOCKERMIN_HEAD (rev=$DOCKERMIN_REV) ==="

# Install dockermin into prime-rl's venv (+ the env-adapter package + missing
# runtime deps). Gotcha #5: dockerfile==3.3.1 is a top-level import in
# dataset/annotate.py and --no-deps skips it. Gotcha #4 NOTE: the probe omitted
# peft because gemma_zs never needs it; the v4 trainer DOES want peft for the
# dockermin_v4 LoRA-adapter eval baseline after training, so include it here.
cd "$HOME/prime-rl"
uv pip install --no-deps -e "$HOME/dockermin"
uv pip install --no-deps -e "$HOME/dockermin/prime_env/dockermin_env"
uv pip install dockerfile==3.3.1 "docker==7.1.*" tenacity peft accelerate "huggingface_hub>=0.26"

# wandb config: v4's TOML already carries per-process [trainer.wandb] /
# [orchestrator.wandb] blocks with offline = true (configs/dockermin_v4.toml,
# Task 2.1 of the plan). No install-time sed patch needed — unlike v2/v3.

# hadolint binary (one of the eval baselines + parser dep).
if ! command -v hadolint >/dev/null; then
    sudo curl -sSL -o /usr/local/bin/hadolint \
        https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
    sudo chmod +x /usr/local/bin/hadolint
fi

# slim binary (for slim_zs baseline / sanity).
if ! command -v slim >/dev/null; then
    cd /tmp
    curl -sLO https://downloads.dockerslim.com/releases/1.40.11/dist_linux.tar.gz
    tar -xzf dist_linux.tar.gz
    sudo mv dist_linux/slim /usr/local/bin/
    sudo mv dist_linux/slim-sensor /usr/local/bin/ 2>/dev/null || true
    rm -rf dist_linux dist_linux.tar.gz
fi

# Stage HF token. Gotcha #7: huggingface-cli is deprecated in
# huggingface_hub >= 1; use `hf auth whoami`.
mkdir -p "$HOME/.cache/huggingface"
echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
uv run --no-sync hf auth whoami

# Pre-flight: verify Gemma 4 architecture is recognized before paying for the
# 24GB weights download. Mirrors install_gemma_probe.sh:92-97.
uv run --no-sync python - <<'PYEOF'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-12B-it")
assert cfg.model_type == "gemma4_unified", f"unexpected model_type={cfg.model_type}"
print(f"config OK (model_type={cfg.model_type}, transformers_version={cfg.transformers_version})")
PYEOF

echo "=== $(date -u) install_v4 done ==="
echo "prime-rl: $PRIME_RL_HEAD"
echo "dockermin: $DOCKERMIN_HEAD (rev=$DOCKERMIN_REV)"
hadolint --version
slim --version 2>&1 | head -1
