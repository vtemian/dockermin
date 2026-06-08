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
git fetch origin --tags
# DOCKERMIN_REV pins what runs on the pod. Default is main for reproducibility.
# When iterating on the probe scripts themselves, pass a branch or commit SHA
# (e.g. DOCKERMIN_REV=feat/gemma-zero-shot-baseline) to test the in-flight
# version. The script never silently follows a moving branch tip.
DOCKERMIN_REV="${DOCKERMIN_REV:-main}"
git checkout "$DOCKERMIN_REV"
# Only fast-forward if the rev is a branch (not a tag or SHA, which are immutable).
if git symbolic-ref -q HEAD >/dev/null; then
    git pull --ff-only
fi

# Bootstrap python3-venv — massedcompute ubuntu_22_cuda_12 image ships
# python3.10's venv module but is missing ensurepip, which python3 -m venv
# needs to bootstrap pip into a fresh venv.
if ! python3 -c "import ensurepip" 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -qq -y python3.10-venv
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
# transformers 5.10.2 references torch.float8_e8m0fnu (microexponents fp8)
# which lands in torch 2.7. Lower torch pins won't pass `import transformers`.
# dockerfile is a runtime import in dataset/annotate.py.
# peft is deliberately omitted — its import chain bottoms out in transformers'
# fp8 module which loads at import time, and the gemma_zs baseline never
# needs PEFT (only the dockermin LoRA adapter baseline does).
pip install \
    "transformers>=5.10.2" \
    "torch>=2.7" \
    "accelerate>=1.1" \
    "huggingface_hub>=0.26" \
    "docker==7.1.*" \
    "dockerfile==3.3.1" \
    "tenacity>=8" \
    "tqdm>=4.66" \
    "datasets>=3" \
    "anthropic" \
    "openai"

# Massedcompute pods don't put the ubuntu user in the docker group by default,
# so any docker call from the eval venv hits "permission denied" on the unix
# socket. Add the user to the docker group — the run script uses `sg docker -c`
# to activate that group inside the same SSH session without re-login. We
# deliberately do NOT chmod 666 the socket: that's a local privilege-escalation
# path that would let anything on the pod own the docker daemon, even though
# the pod is single-purpose and ephemeral.
sudo usermod -aG docker ubuntu

# Editable dockermin without --no-deps would re-pin transformers down to 4.46.
# --ignore-requires-python because pyproject pins ==3.11.* (the trainer's
# constraint), while this probe venv runs on the pod's system python3.10 —
# fine for eval-only code which is just stdlib + transformers/datasets/docker.
pip install -e . --no-deps --ignore-requires-python

# Stage HF token (mirrors the prime-rl pod pattern).
mkdir -p "$HOME/.cache/huggingface"
echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
hf auth whoami

# Verify the architecture is recognized before paying for the weights download.
python - <<'PYEOF'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("google/gemma-4-12B-it")
assert cfg.model_type == "gemma4_unified", f"unexpected model_type={cfg.model_type}"
print(f"config OK (model_type={cfg.model_type}, transformers_version={cfg.transformers_version})")
PYEOF

echo "[install_gemma_probe] done."
