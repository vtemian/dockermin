# Pilot runbook — 50-step GRPO go/no-go (Phase 3)

Sequence for the pilot pod session. Builds on the proven Phase 1 single-GPU recipe
(docs/decisions/2026-05-22-prime-rl-pin.md) plus the dockermin-specific steps. Bot drives;
Vlad adds credits + authorizes the rental.

**Decisions baked in:**
- 1xH100 colocated (inference + trainer + orchestrator on GPU 0) — proven in Phase 1.
- **No egress proxy for the pilot** — full egress on a throwaway pod (v0-accepted risk per
  docs/decisions execution log). setup_pod_proxy.sh has an untested HTTPS-filter bug; fix it
  before the full run, not now.
- Dataset `vtemian/dockermin-v0` is public → no HF token needed to read it.
- The reward's docker builds run in the **orchestrator** process (CPU-side), so they do not
  fight the GPU — but build throughput gates the whole loop. This session **calibrates
  builds/min** to produce a real full-run estimate.

## 0. Rent pod
Prefer a 1xH100 with the most vCPUs available (build concurrency); cheapest 1xH100 is fine
for the pilot. `prime availability list --gpu-type H100_80GB --gpu-count 1 --output json`,
pick cheapest in-stock on-demand, `prime pods create --id <id> --name dockermin-pilot --yes --plain`.
SSH user is `ubuntu` (key: `ssh -i ~/.ssh/id_rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa ubuntu@<ip>`).

## 1. Install prime-rl (Phase 1 recipe, ~15 min)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
uv sync --all-extras
```

## 2. Install dockermin + env adapter INTO the prime-rl venv
```bash
cd ~ && git clone https://github.com/vtemian/dockermin.git
cd ~/prime-rl
uv pip install -e ~/dockermin                      # the dockermin package (reward, gates)
uv pip install -e ~/dockermin/prime_env/dockermin_env
uv run --no-sync python -c "import dockermin_env, dockermin.reward.dockermin_reward; print('env import OK')"
```

## 3. DooD setup (NO proxy)
```bash
bash ~/dockermin/scripts/setup_pod_docker.sh    # sudo-aware; ends with "docker setup ok"
```
If it added the `ubuntu` user to the docker group, open a fresh shell (or `newgrp docker`)
so the group takes effect for the orchestrator process.

## 4. Validate the config (no GPU spend yet)
```bash
cd ~/prime-rl
uv run --no-sync rl @ ~/dockermin/configs/dockermin_pilot.toml --dry-run
```
Fix any schema mismatch here (the config mirrors alphabet_sort but is unproven). Generates
outputs/configs/{inference,orchestrator,trainer}.toml.

## 5. Launch pilot — single-GPU colocated (3 processes)
```bash
cd ~/prime-rl && export WANDB_MODE=offline DOCKERMIN_MAX_BUILDS=6

# Inference (GPU 0, half mem, capped context for Qwen-7B 32K)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 setsid \
  uv run --no-sync inference @ outputs/configs/inference.toml \
  --gpu-memory-utilization 0.5 --model.max-model-len 8192 > ~/inference.log 2>&1 < /dev/null &
# wait for: curl -sf http://localhost:8000/v1/models

# Trainer (GPU 0, via torchrun)
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid \
  uv run --no-sync torchrun --nproc-per-node=1 --rdzv-endpoint=localhost:29507 --rdzv-id=pilot \
  -m prime_rl.trainer.rl.train @ outputs/configs/trainer.toml > ~/trainer.log 2>&1 < /dev/null &

# Orchestrator (CPU; runs our env -> docker builds). Needs docker on PATH + DOCKERMIN_MAX_BUILDS.
DOCKERMIN_MAX_BUILDS=6 setsid \
  uv run --no-sync orchestrator @ outputs/configs/orchestrator.toml > ~/orchestrator.log 2>&1 < /dev/null &
```

## 6. Monitor + CALIBRATE
- `grep "Reward:" ~/orchestrator.log` — reward should move (baseline ~0.5 from the +0.5 floor).
- **Calibrate:** time per orchestrator step over the first 3-5 steps → builds/min =
  (batch_size×group_size) / step_minutes. This converts the full-run estimate from a range to
  a number. 64 builds/step here.
- Watch ~/orchestrator.log for docker errors (egress, timeouts, OOM-in-build).
- GPU memory: `nvidia-smi` — inference + trainer should fit (~69 GiB like Phase 1).

## 7. Reward-hacking audit
```bash
uv run --no-sync python ~/dockermin/scripts/audit_rollouts.py   # % FROM scratch, % no CMD, :latest, mean lines
```

## 8. Terminate + log
`prime pods terminate <id> --yes`, confirm `prime pods list` shows 0, update docs/cost_log.md
+ docs/journal.md, commit. **Then I give the real full-run hour/$ estimate from the calibration.**

## Go/no-go gate (per project kill criteria)
- Reward trends up over the pilot AND audit shows no rampant reward-hacking → proceed to full run.
- Flat/NaN reward, or audit shows degenerate Dockerfiles dominating → stop, diagnose before spending on the full run.
