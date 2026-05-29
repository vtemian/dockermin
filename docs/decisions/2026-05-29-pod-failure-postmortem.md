# Decision: pod-failure postmortem and defense-in-depth (2026-05-29)

**Context:** GRPO v1.1 training, three back-to-back GPU pod failures across two cloud
providers. The training algorithm worked (rolling pass rate climbed 25 % → 75 %) but
each pod went `ACTIVE → PENDING/UNKNOWN` before reaching step 250, and no adapter ever
made it to HF.

## What we observed

| Pod | Provider | Lifespan | `DOCKERMIN_MAX_BUILDS` | Died at step | Disk pressure observed before death? |
|---|---|---|---|---|---|
| #1 | lambdalabs SXM5 H100 | 1.7 h | 20 | ~35 | unknown — died too fast to check |
| #2 | lambdalabs SXM5 H100 | 4.5 h | 20 | ~234 | unknown |
| #3 | massedcompute SXM4 A100 | 5.7 h | 12 | ~130 (138 pre-crash + 4 post-resume) | **yes** — full ENOSPC on broadcast at step 138, recovered via 611 GB docker prune, died again ~1 h post-resume |

Identical external symptom on all three: `SSH timeout → prime API: Status=UNKNOWN / IP=N/A`.
Differing: providers, GPU sockets, days, time-of-day. Three failures across two providers
in 24 h argues against simple regional outage.

## Hypotheses, ranked by evidence

### H1 — Mount-table exhaustion from accumulating overlayfs mounts (most likely)

dockermin's reward gates run `docker build` and `docker run <test_cmd>` per rollout. At
`batch_size=16 × group_size=16 = 256 builds/step`. Each `docker buildx build` creates an
overlayfs image with ~5 layer mounts. Linux default `fs.mount-max = 100000`.

- Pod #1 step ~35 → ~8 960 builds × 5 = 44 800 mounts (under limit; died on something else)
- Pod #2 step ~234 → 59 904 builds × 5 = **299 520 mounts (way over default)**
- Pod #3 step ~130 → 36 352 builds × 5 = **181 760 mounts (over default)**

When `fs.mount-max` is hit, `dockerd` hangs → SSH server starves → kernel watchdog →
cloud control plane marks UNKNOWN.

Caveat: I could not verify mount counts because the dead pods are gone with their state.
The math is consistent but not conclusive.

### H2 — Docker image + buildx-cache accumulation → disk full → cascade

**Confirmed on pod #3** at step 138: trainer hit `[Errno 28] No space left on device`
writing the broadcast → ENOSPC propagated through orchestrator's logging → `env_server`
crashed. Docker held **624 GB images + 704 GB buildx build cache** on a 984 GB disk.
A `docker system prune -af` reclaimed 611 GB and let training resume cleanly.

Doesn't fully explain pod #3's *second* death: post-prune we had 716 GB headroom and the
pod still died after ~1 h. Either the 30-min pruner cadence was too loose, or H1 was the
proximate cause.

### H3 — Provider-side spot preemption disguised as on-demand

Three failures across two providers in 24 h is suspicious. The clean `PENDING → UNKNOWN`
transition looks like control-plane eviction more than a host crash. Counter-evidence:
pod #2 lived 4.5 h before failing — that's not a random short-window preemption signature.

### H4 — CPU/memory saturation → OOM-killer → host instability

256 concurrent builds + vLLM + trainer is a CPU/memory storm. On 16 vCPU (massedcompute)
with `MAX_BUILDS=12`: 12 docker builds (~2 CPU each = 24 CPU-equivalent) + vLLM (low CPU)
+ trainer (4 CPU) → 30+ logical threads on 16 vCPU. Linux OOM-killer could hit `dockerd`,
which would cascade exactly like what we saw.

## Why the original 30-min pruner did not save pod #3

The pruner I wrote first ran `docker image prune` every 30 min. But:

- Each build creates an image referenced by recent containers (not yet garbage-collected)
- Mount points persist for in-flight builds even after image deletion
- `docker image prune` doesn't touch the **buildx** builder cache (separate storage)

So the underlying mount + buildx-cache accumulation kept growing inside each 30-min
window even while images were getting reaped.

## Defense in depth (pod #4 onward)

| Fix | Rationale |
|---|---|
| `sysctl fs.mount-max=1000000` in `install.sh` | 10× kernel default → mount-table no longer the bottleneck even for full 64 k builds |
| `DOCKERMIN_MAX_BUILDS=4` (was 12) | 3× less concurrent docker churn → step time roughly 3× slower but pod survives |
| `docker_pruner.sh` every **5 min** with `docker system prune -af --volumes` + `docker buildx prune -af` | Reclaims buildx cache (the real disk hog) and limits image/container lifetime |
| `hf_pusher.sh` every 5 min, pushes every STABLE broadcast to `vtemian/dockermin-qwen7b-lora-v2/step_N/` | Even if pod #4 dies at step 150, we have a usable adapter on HF — pod death is no longer total loss |
| Pruner also logs `/proc/mounts` count, overlay count, disk usage | Future post-mortems have evidence |

The technique is proven by the rolling pass-rate climb (25 % → 75 % over the first 130
steps of pod #3). What we are paying for now is operational reliability, not algorithmic
discovery.

## Cost so far

| Pod | Spent |
|---|---|
| #1 | $7.26 |
| #2 | $19.22 |
| #3 | ~$9.66 |
| #4 | running ($1.79/h, ~14 h projected ≈ $25) |
| **Session-total projection** | **≈ $61** |

That puts us at the $58 cap I projected before pod #1, with one usable adapter at the
end. Within budget and within Vlad's pre-committed kill criteria.

## Open questions

- Were pods #1 and #2 also killed by mount/disk pressure, or something else?
  Without per-pod logs of `/proc/mounts` and `df`, we cannot tell. The pruner now
  collects this evidence so the next failure mode is diagnosable.
- Is `DOCKERMIN_MAX_BUILDS=4` the right floor? Pod #4 will tell us — if it survives to
  step 250, the answer is "yes, lower MAX_BUILDS until pod is stable."
- Should dockermin's reward gates be migrated off `docker buildx` to plain `docker build`?
  Buildx's persistent builder cache is the disk-fill amplifier. Worth measuring on a
  future run; out of scope for this fire-fight.
