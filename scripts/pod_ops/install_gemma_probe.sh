#!/bin/bash
# Pod-side installer for the Gemma 4 12B-it zero-shot eval probe.
#
# Idempotent: re-running re-uses the venv but re-installs the editable dockermin.
# Runs OUTSIDE the dockermin/prime-rl uv venv because Gemma 4 needs
# transformers>=5.10.2 (project pin is transformers==4.46, which rejects
# model_type=gemma4_unified). See docs/plans/2026-06-08-gemma4-zero-shot-probe.md.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set on the pod}"

cd "$HOME"
if [ ! -d dockermin ]; then
    git clone https://github.com/vtemian/dockermin.git
fi
cd dockermin
git fetch origin
git checkout feat/gemma-zero-shot-baseline
git pull --ff-only

# Bootstrap python3-venv — massedcompute ubuntu_22_cuda_12 image ships
# python3.10 without the venv module.
if ! python3 -c "import venv" 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -qq -y python3-venv python3.10-venv
fi

# Separate venv at $HOME/gemma-probe-venv — must not share state with the
# transformers==4.46 venv prime-rl owns. Uses whatever python3 ships on the
# pod (transformers 5.x requires 3.10+; the project pin is 3.11 but the probe
# venv runs eval only, not training, so 3.10 is fine).
#
# Idempotency: detect an incomplete venv (e.g. an earlier run that aborted
# before ensurepip ran) by checking for bin/activate, not just the dir.
if [ ! -f "$HOME/gemma-probe-venv/bin/activate" ]; then
    rm -rf "$HOME/gemma-probe-venv"
    python3 -m venv "$HOME/gemma-probe-venv"
fi
# shellcheck disable=SC1091
source "$HOME/gemma-probe-venv/bin/activate"

pip install -U pip wheel
pip install \
    "transformers>=5.10.2" \
    "torch>=2.5,<2.7" \
    "accelerate>=1.1" \
    "huggingface_hub>=0.26" \
    "docker==7.1.*" \
    "tenacity>=8" \
    "tqdm>=4.66" \
    "datasets>=3" \
    "anthropic" \
    "openai"

# Editable dockermin without --no-deps would re-pin transformers down to 4.46.
pip install -e . --no-deps

# Stage HF token (mirrors the prime-rl pod pattern).
mkdir -p "$HOME/.cache/huggingface"
echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
huggingface-cli whoami

# Verify the architecture is recognized before paying for the weights download.
python - <<'PYEOF'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-12B-it")
assert cfg.model_type == "gemma4_unified", f"unexpected model_type={cfg.model_type}"
print(f"config OK (model_type={cfg.model_type}, transformers_version={cfg.transformers_version})")
PYEOF

echo "[install_gemma_probe] done."
