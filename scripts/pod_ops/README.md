# scripts/pod_ops — pod-side operational scripts for v1.1 GRPO runs

Snapshot of the scripts used on the GRPO v1.1 training runs (2026-05-28 through
2026-05-29). They live on the GPU pod (`~/`) and run detached via `setsid nohup`.

| Script | Role |
|---|---|
| `install.sh` | One-shot pod bootstrap: uv, prime-rl, dockermin, env-adapter package, runtime deps, wandb-offline config patch, `fs.mount-max=1000000`. Skips fish-completions on permission error. Hadolint download is non-critical (eval-only) and may fail without stopping the script logic since the script's `set -e` exits before the final marker; in that case the env still loads fine for training. |
| `launch_v1_1.sh` | Spawns inference + trainer + orchestrator (single-GPU colocated). `DOCKERMIN_MAX_BUILDS=4` to limit docker pressure (3× lower than the pod-#3 setting that caused disk fill). |
| `launch_resume.sh` | Same as above but passes `--ckpt.resume_step <N>` so the run continues from the last STABLE checkpoint after a crash. |
| `post_train.sh` | Watches `~/trainer.log` for completion (or fatal error), pushes the final LoRA adapter to HF, then self-terminates the pod. 24h watchdog backstop. |
| `docker_pruner.sh` | Every 5 min, runs `docker system prune -af --volumes` + `docker buildx prune -af`. Logs mount-table size, overlay mounts, disk usage to `~/docker_prune.log` so post-mortem evidence survives. |
| `hf_pusher.sh` | Every 5 min, pushes any new STABLE broadcast (`broadcasts/step_*/STABLE`) to `vtemian/dockermin-qwen7b-lora-v2/step_N/`. Tracks pushed steps in `~/.hf_pushed`. Ensures that a pod death never costs us a working adapter. |

## Why these exist

The first three v1.1 attempts died external to training, not algorithmically:

| Pod | Provider | Step at death | Apparent cause |
|---|---|---|---|
| #1 | lambdalabs SXM5 H100 | ~35 | UNKNOWN — platform eviction or kernel mount/disk pressure (unverified) |
| #2 | lambdalabs SXM5 H100 | ~234 | same external symptom (`ACTIVE → PENDING → UNKNOWN`) |
| #3 | massedcompute SXM4 A100 | ~130 (post-resume) | confirmed disk full from docker accumulation (1.3 TB images + buildx cache), recovered once via prune, died again ~1h later |

The fix is the combination of the 4 background scripts (pruner, pusher, post-train, watchdog)
plus the lowered `MAX_BUILDS=4` in `launch_v1_1.sh` and the `fs.mount-max` raise in
`install.sh`. See `docs/decisions/2026-05-29-pod-failure-postmortem.md` for the deep
analysis.

## Usage on a fresh pod

```bash
# 1. Bootstrap (idempotent)
bash ~/install.sh

# 2. Stage secrets — HF token at ~/.cache/huggingface/token, prime CLI at ~/.prime/config.json

# 3. Start the four background services
setsid nohup bash ~/docker_pruner.sh </dev/null >>~/docker_prune.log 2>&1 &
setsid nohup bash ~/hf_pusher.sh     </dev/null >>~/hf_pusher.log    2>&1 &
setsid nohup bash ~/post_train.sh <POD_ID> </dev/null >~/post_train.out 2>&1 &

# 4. Launch training
setsid nohup bash ~/launch_v1_1.sh </dev/null >~/launch.out 2>&1 &
```

Resume after a crash uses `launch_resume.sh` with `RESUME_STEP=<last STABLE>`.
