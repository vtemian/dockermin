#!/bin/bash
# Self-contained install on a fresh lambdalabs ubuntu_22_cuda_12 pod.
# Idempotent — safe to re-run.
set -euo pipefail
exec > >(tee -a "$HOME/install.log") 2>&1
echo "=== $(date -u) install start ==="

# Docker group access (one-time; takes effect on next login / sg)
sudo usermod -aG docker ubuntu || true

# Pod-survival sysctls: dockermin's per-rollout builds create overlayfs mounts
# that linger; three prior pods died with mount/disk exhaustion patterns.
echo "fs.mount-max=1000000" | sudo tee /etc/sysctl.d/99-dockermin.conf >/dev/null
sudo sysctl --system >/dev/null 2>&1 || true

# uv (skip if already present). `|| true` because uv's installer chmods /home/ubuntu/.config/fish
# on some images and aborts on Permission denied even though uv itself installed fine.
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv install failed; aborting"; exit 1; }

# prime-rl
if [ ! -d "$HOME/prime-rl" ]; then
  cd "$HOME"
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git
  cd prime-rl
  git config --global url."https://github.com/".insteadOf "git@github.com:"
  git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
fi
cd "$HOME/prime-rl"
PRIME_RL_HEAD=$(git rev-parse HEAD)
echo "=== prime-rl HEAD: $PRIME_RL_HEAD ==="

# uv sync (the slow step, ~10-15 min)
echo "=== $(date -u) uv sync start ==="
uv sync --all-extras
echo "=== $(date -u) uv sync done ==="

# dockermin (no deps — already pulled by prime-rl's environment)
if [ ! -d "$HOME/dockermin" ]; then
  cd "$HOME"
  git clone https://github.com/vtemian/dockermin.git
fi
cd "$HOME/dockermin"
git fetch origin && git checkout main && git pull origin main
DOCKERMIN_HEAD=$(git rev-parse HEAD)
echo "=== dockermin HEAD: $DOCKERMIN_HEAD ==="

# Install dockermin into prime-rl's venv (+ the env-adapter package + missing runtime deps)
cd "$HOME/prime-rl"
uv pip install --no-deps -e "$HOME/dockermin"
uv pip install --no-deps -e "$HOME/dockermin/prime_env/dockermin_env"
uv pip install dockerfile==3.3.1 "docker==7.1.*" tenacity

# Patch wandb config to per-process offline (top-level [wandb].offline is rejected by `rl`)
python3 - <<'PY'
import re, pathlib
p = pathlib.Path.home()/"dockermin/configs/dockermin_full.toml"
src = p.read_text()
if "[trainer.wandb]" not in src:
    # Replace the single [wandb] block with two per-process blocks (offline = true)
    src = src.replace(
        '[wandb]\nproject = "dockermin"\nname = "full-200step"\n',
        '[trainer.wandb]\nproject = "dockermin"\nname = "full-200step"\noffline = true\n\n'
        '[orchestrator.wandb]\nproject = "dockermin"\nname = "full-200step"\noffline = true\n',
    )
    p.write_text(src)
    print("wandb config patched to per-process offline")
PY

# hadolint binary (one of the eval baselines + parser dep)
if ! command -v hadolint >/dev/null; then
  sudo curl -sSL -o /usr/local/bin/hadolint \
    https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
  sudo chmod +x /usr/local/bin/hadolint
fi

# slim binary (for slim_zs baseline / sanity)
if ! command -v slim >/dev/null; then
  cd /tmp
  curl -sLO https://downloads.dockerslim.com/releases/1.40.11/dist_linux.tar.gz
  tar -xzf dist_linux.tar.gz
  sudo mv dist_linux/slim /usr/local/bin/
  sudo mv dist_linux/slim-sensor /usr/local/bin/ 2>/dev/null || true
  rm -rf dist_linux dist_linux.tar.gz
fi

echo "=== $(date -u) install done ==="
echo "prime-rl: $PRIME_RL_HEAD"
echo "dockermin: $DOCKERMIN_HEAD"
hadolint --version
slim --version 2>&1 | head -1
