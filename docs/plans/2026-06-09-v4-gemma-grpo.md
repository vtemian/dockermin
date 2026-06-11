# GRPO v4 — Gemma 4 12B-it Base Swap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a single GRPO arc on `google/gemma-4-12B-it` (LoRA, ~200 steps) and beat the gemma_zs floor on the n=37 holdout: pass-rate ≥ 81.1% **AND** reduction-on-pass ≥ 50%. Success clears the Branch-B veto from `docs/plans/2026-06-08-gemma4-zero-shot-probe.md`.

**Architecture:** Stack-swap, not algorithm swap. The reward ladder (`parse → manifest → build → test → size`) is unchanged. Prime-rl's homegrown LoRA, verifiers env adapter, hf_pusher/post_train pattern, and 37-row holdout are unchanged. What moves: (a) the base model weights, (b) the transformers/torch/vLLM versions on the training pod, (c) one LoRA hparam knob (G=8 instead of v3's G=16) to fit 12B inside A100-80GB colocation, (d) a new pod-ops install script that bakes in the 8 gotchas from the gemma_zs probe.

**Tech Stack:** Python 3.12, prime-rl `main` (post-PR-#2347 if merged, else separate trainer venv per Decision D1), transformers ≥ 5.10.2, torch ≥ 2.7, vLLM nightly `0.22.1rc1.dev+` with `Gemma4UnifiedForConditionalGeneration`, verifiers ≥ 0.1.15, Gemma 4 12B-it + LoRA (r=32 α=64, target_modules = standard 7 projections on `language_model` sub-tree), single A100 80GB SXM4 colocated trainer+inference.

---

## Subagent investigation summary (for executor context)

Four parallel research subagents established:

1. **prime-rl × Gemma 4 (Agent 1):** The pinned prime-rl SHA `91182b7d` (per `pyproject.toml:13`) carries `transformers>=4.56` and vLLM 0.21.0, neither of which loads `Gemma4UnifiedForConditionalGeneration` (verified against vLLM v0.21.0 registry — only `main` registers it). prime-rl `main` already moved to `transformers==5.6.2` + vLLM ≥ 0.22.0; Gemma 4 first-class training support is in **open** PR #2347 (`fix-gemma`, SDPA-only, seqlen ≤ 16k). Open issue #2362 documents an unresolved gradient-amplification problem on Gemma 4 SFT+LoRA (1k–1M pre-clip vs 15k–80k baseline) — **no GRPO repro on file**.

2. **Reward stack (Agent 2):** Default verdict holds — **no changes**. `parse_gate` requires fenced blocks (intentional; Gemma's 2-3 fence-less parse failures are not salvageable from logged data); `build_gate` is model-agnostic; size-reward (`gates.py:143-147`) already clamps negatives and divide-by-zero; `manifest_gate` (`annotate.py:237-260`) is model-agnostic and will simply fire less under Gemma's zero-hallucinated-tag behavior.

3. **LoRA + chat template (Agent 3):** v3 hparams (r=32, α=64, lr=5e-6, batch=32, G=16, seq_len=5120, max_completion_tokens=2048) at 12B colocation **do not fit** — vLLM share at `gpu-memory-utilization=0.30` (24 GB) leaves zero KV headroom (base weights alone are 23.9 GB). Workable combos at A100-80GB single-GPU: `G=4 @ max_ml=5120 @ vllm=0.45`; `G=8 @ max_ml=2560 @ vllm=0.40`; `G=16 @ max_ml=2560 @ vllm=0.50`. Chat-template plumbing **needs zero dockermin changes**: `prime_env/dockermin_env/dockermin_env.py:25-31` emits role dicts only, and prime-rl's `[orchestrator.renderer] name = "default"` (per `configs/dockermin_v3.toml:115-116`) calls `tokenizer.apply_chat_template`, which Gemma's tokenizer resolves natively.

4. **Cost envelope (Agent 4):** Step-time multiplier 7B→12B = 1.5×–2.0× (median 1.7×). 200-step arc projects $42–$54 + ~$2 ops + ~$2 eval ≈ **$46–$58 envelope**. 250-step arc projects $52–$68 (just inside $80). **500-step arc projects $104–$136 (BUSTS the $80 envelope and burns ~$5 below the $200 pause-checkpoint)**. Provider primary = massedcompute A100 80GB SXM4 ($1.23/hr, only platform with a completed end-to-end GRPO arc per `cost_log.md:22`); fallback = crusoe ($1.79/hr, 1/1 success rate).

---

## Pre-execution decisions for Vlad

These four questions gate Phase 1. The plan's task list assumes the **default** option in each; if Vlad picks differently, edit the affected tasks before `/ship-it`.

### D1 — How to resolve the transformers/prime-rl version conflict

| Option | Steps | Risk |
|---|---|---|
| **A. Separate trainer venv** (default fallback) | Build `~/gemma-trainer-venv` with transformers 5.10.2, torch 2.7, vLLM nightly, hand-installed prime-rl deps (`verifiers, torchtitan, dion, liger-kernel, ring-flash-attn, mooncake-transfer-engine`); launch trainer + orchestrator + inference each from their own venv via `setsid env VIRTUAL_ENV=...`. | Highest install complexity (8+ transitive deps to hand-pin). Probe venv proved the forward+generate path works; trainer adds LoRA-write + checkpoint paths that are unverified. |
| **B. Bump prime-rl pin to `main` HEAD** (**default**) | Edit `pyproject.toml:13` to track a post-PR-#2347-merge commit (or `main` if PR open); `uv sync --all-extras` resolves transformers 5.6.2 + vLLM ≥ 0.22.0 automatically. | If PR #2347 unmerged, Gemma 4 won't be in `VLM_REGISTRY` — falls back to plain HF AutoModel load; vLLM may crash on global-attn `o_proj` per lna-lab recipe. Mitigation: vendor PR #2347 patches into a thin wrapper. |
| C. Sed-patch pin on pod | Sed `transformers>=4.56.0` → `transformers>=5.10.2` in `~/prime-rl/pyproject.toml` before `uv sync` AND swap vLLM to nightly wheel. | uv.lock has git-pinned submodules; sed is shallow; vLLM 0.21.0 actively excludes transformers 5.0.*–5.5.0 in its requirements. Likely to fail at install. |
| D. Fork prime-rl | Maintain `vtemian/prime-rl` with bumped pins + cherry-picked PR #2347. | 686+ commits of upstream drift since v2 (`docs/decisions/2026-05-22-prime-rl-pin.md:24`); fork compounds maintenance. |

**Default: Option B** — bump pin. If PR #2347 is unmerged at execution start, the plan splits into B+vendor-patch (apply diff manually in install script). If `main` HEAD has known regressions, fall back to **Option A**.

### D2 — vLLM version

| Option | What | Default |
|---|---|---|
| `vllm>=0.22.1` (released) | Falls back to transformers backend for `gemma4_unified`; crashes in global-attn `o_proj` per lna-lab recipe. | ❌ |
| `vllm>=0.22.1rc1.dev` (**default**) | Registers `Gemma4UnifiedForConditionalGeneration` natively; supports LoRA via inheritance. | ✅ |
| `--from-source` build | Build vLLM from `main` against pod's CUDA + torch. | Reserved for fallback only — wheel must match cu128/cu129 of the pod image. |

### D3 — LoRA hyperparameters (memory-fit on A100-80GB)

| Knob | v3 (Qwen 7B) | v4 default (Gemma 12B) | Rationale |
|---|---|---|---|
| LoRA rank | 32 | **32** | Prior-art neutral; Unsloth recipe suggests 8–32 for 12B; r=16 saves ~1GB optim but loses expressiveness. |
| LoRA alpha | 64 | **64** | Match α = 2r ratio. |
| target_modules | (prime-rl default = all-linear of `model.layers`) | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` on `model.language_model.layers.*` | Standard 7-proj set; exclude `mm_*` and `audio_*` projections (text-only task). Verify with `python -c "from transformers import AutoModelForCausalLM; ..."` on pod (Phase 1 Task 4). |
| Learning rate | 5e-6 | **5e-6** | Unsloth Gemma 4 GRPO docs match exactly. |
| max_steps | 500 | **200** | Cost envelope. 200 ≈ $46–$58. 250 ≈ $52–$68 (within $80). 500 busts. |
| ckpt interval | 50 | **25** | Eval grid 50/100/150/200 + 1 headroom; tighter checkpoints absorb a spot-pod death better at the cost of ~5% more disk I/O. |
| batch_size | 32 | **16** | Halved to absorb vLLM memory tax. v2 used 16; signal-to-noise stayed adequate. |
| group_size G | 16 | **8** | At max_model_len=2560 colocation, G=16 OOMs vLLM. G=8 keeps advantage-est noise at √2× v3 — still inside prime-rl precedent. |
| seq_len | 5120 | **4096** | Median train prompt ~1500 tok + completion 1024 → 4096 covers with margin and recovers ~2 GB of trainer activation. |
| max_completion_tokens | 2048 | **1024** | Align with `gemma_zs` probe + eval (`baselines.py:197`). Prevents the v2/v3 train-vs-eval footgun. |
| inference `--gpu-memory-utilization` | 0.30 (Qwen 7B) | **0.40** (Gemma 12B) | Weights alone are 23.9 GB; 0.40 of 80 GB = 32 GB → 8 GB KV headroom. |
| inference `--max-model-len` | 5120 | **2560** | Halved with seq_len. |
| `[ckpt] keep_last` | 6 | **6** | Same. |

### D4 — Primary + fallback provider

| Role | Provider | Why |
|---|---|---|
| Primary | **massedcompute A100 80GB SXM4** ($1.23–$1.79/hr) | Only platform with end-to-end GRPO arc completion on record (`cost_log.md:22` v1.1 retry 4); only platform where gemma_zs probe ran end-to-end (`cost_log.md:28`). 3 prior eval-pod deaths absorbed by hf_pusher+resume pattern. |
| Fallback | **crusoe A100 80GB** ($1.79/hr) | 1/1 success rate (`cost_log.md:25`). Lambdalabs and nebius excluded — 0/2 and 0/1 respectively. |

---

## Quality gate (mandatory before every commit)

Per `CLAUDE.md` § "Pre-Commit Gate":

```bash
make quality   # ruff + mypy strict + pytest -m "not docker"
```

A green local gate is the contract for green CI. Branch per task: `git checkout -b feat/v4-<scope>`.

---

## Phase 1 — Stack integration (transformers / vLLM / prime-rl / Gemma)

### Task 1.1: Bump prime-rl pin (D1 option B)

**Files:**
- Modify: `pyproject.toml:13` (replace `prime-rl @ git+https://...@91182b7d647285a3e9e32f7959fdc3ff044d9330` with a post-PR-#2347 commit OR `main` HEAD)
- Modify: `docs/decisions/2026-05-22-prime-rl-pin.md` — append v4 bump rationale (single paragraph, why we moved off `91182b7d`).
- Test: `tests/test_install_ordering.py` if exists; otherwise rely on `make quality`.

**Step 1: Resolve target commit**

```bash
# Check PR #2347 state
gh pr view 2347 --repo PrimeIntellect-ai/prime-rl --json mergeCommit,state
```

If `state=MERGED`, use `mergeCommit.oid`. If `OPEN`, use `git ls-remote https://github.com/PrimeIntellect-ai/prime-rl main` and apply PR diff in install script (Task 1.4).

**Step 2: Update pin**

```diff
- "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl.git@91182b7d647285a3e9e32f7959fdc3ff044d9330",
+ "prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl.git@<NEW_SHA>",
```

**Step 3: Run quality gate**

```bash
make quality
```

Expected: PASS (the pin change is metadata only; no source changes; tests don't import prime-rl beyond the env adapter's `import verifiers as vf`).

**Step 4: Commit**

```bash
git checkout -b feat/v4-prime-rl-pin-bump
git add pyproject.toml docs/decisions/2026-05-22-prime-rl-pin.md
git commit -m "chore(deps): bump prime-rl pin to <SHA> for Gemma 4 transformers 5.x support"
```

### Task 1.2: dockermin transformers floor

**Files:**
- Modify: `pyproject.toml` — find the existing transformers pin (`transformers==4.46.*` per probe decision note). Change to `transformers>=5.10.2`.
- Test: `tests/test_baselines.py` should still pass (gemma_zs uses transformers 5.10.2 already).

**Step 1: Edit pin**

```bash
grep -n "transformers" pyproject.toml
```

```diff
- "transformers==4.46.*",
+ "transformers>=5.10.2",
```

**Step 2: Quality gate**

```bash
make quality
```

Expected: PASS. `_qwen_zs_baseline` may need a smoke check — if it pinned to transformers 4.x APIs, this is the moment to find out. **Do not bypass** — if mypy or tests fail, surface to Vlad before touching baselines.py.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): bump transformers floor to >=5.10.2 for Gemma 4"
```

### Task 1.3: Add `dockermin` baseline registry stub for `gemma_v4`

**Files:**
- Modify: `src/dockermin/eval/baselines.py` — add `baseline_dockermin_gemma` mirroring `baseline_dockermin` but `model_id="google/gemma-4-12B-it"` and `adapter_id="vtemian/dockermin-gemma4-12b-it-v4"`.
- Modify: `src/dockermin/eval/baselines.py` `_REGISTRY` — add `"dockermin_v4": baseline_dockermin_gemma`.
- Modify: `scripts/run_eval.py:129` — add `"dockermin_v4"` to the `{"qwen_zs", "gemma_zs", "dockermin"}` set so temperature/max-new-tokens thread through.
- Test: `tests/test_baselines.py:231` — add `"dockermin_v4"` to literal set.

**Step 1: RED — failing test**

```python
def test_dockermin_v4_in_registry() -> None:
    from dockermin.eval.baselines import _REGISTRY
    assert "dockermin_v4" in _REGISTRY
```

**Step 2: Run**

```bash
uv run pytest tests/test_baselines.py::test_dockermin_v4_in_registry -v
```

Expected: FAIL.

**Step 3: GREEN — implement**

Add the function and registry entry mirroring `baseline_dockermin` exactly, only changing model id + adapter id. **No new code paths** — both run through the same `apply_chat_template` + `_load_lora_model` plumbing.

**Step 4: Run quality**

```bash
make quality
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/dockermin/eval/baselines.py scripts/run_eval.py tests/test_baselines.py
git commit -m "feat(eval): add dockermin_v4 baseline (Gemma 4 12B-it + LoRA adapter)"
```

### Task 1.4: Pod-ops install script for v4 trainer

**Files:**
- Create: `scripts/pod_ops/install_v4.sh` — modeled on `install.sh` but with the 8 gemma probe gotchas baked in.

**Differences vs `install.sh`:**

1. Require Python 3.12 (uv-managed): `uv python install 3.12.13` before `uv sync`.
2. After `uv sync --all-extras`, verify `transformers>=5.10.2` and `vllm>=0.22.1rc1.dev` are resolved (fail loudly if not).
3. After `uv pip install --no-deps -e dockermin`, ALSO `uv pip install dockerfile==3.3.1 "docker==7.1.*" tenacity peft accelerate "huggingface_hub>=0.26"` — same set as the probe.
4. If PR #2347 is unmerged, apply its diff: `cd ~/prime-rl && git fetch origin pull/2347/head:fix-gemma && git cherry-pick --no-commit fix-gemma || true` (idempotent guard).
5. `sudo usermod -aG docker ubuntu` — no chmod, use `sg docker -c` in launch.
6. `hf auth whoami` (not `huggingface-cli whoami`).
7. Verify Gemma loads on first attempt: `python -c "from transformers import AutoConfig; cfg = AutoConfig.from_pretrained('google/gemma-4-12B-it'); assert cfg.model_type == 'gemma4_unified'"`.

**Step 1: Write script** (full content; copy `install.sh` and adjust).

**Step 2: Smoke-test it idempotency-wise**

```bash
shellcheck scripts/pod_ops/install_v4.sh
```

Expected: 0 errors.

**Step 3: Commit**

```bash
git add scripts/pod_ops/install_v4.sh
git commit -m "feat(pod_ops): add v4 install script with Gemma probe gotchas baked in"
```

---

## Phase 2 — v4 GRPO config

### Task 2.1: Write `configs/dockermin_v4.toml`

**Files:**
- Create: `configs/dockermin_v4.toml`

**Contents** (derived from `configs/dockermin_v3.toml` with D3 deltas):

```toml
# configs/dockermin_v4.toml — GRPO v4 (Gemma 4 12B-it base swap)
#
# Plan: docs/plans/2026-06-09-v4-gemma-grpo.md
# Probe result: docs/decisions/2026-06-08-gemma-zs-probe-result.md (81.1% pass, 38.2% reduction)
# Decision rule: this arc succeeds iff pass-rate >= 81.1 AND reduction|pass >= 50%.
#
# Key deltas vs configs/dockermin_v3.toml (Qwen 7B):
#   - [model] name = "google/gemma-4-12B-it"  (NEW)
#   - [trainer.model.lora] target_modules pinned to language_model sub-tree
#       (Gemma 4 Unified has vision/audio projections we must skip)
#   - max_steps: 500 -> 200  (12B step-time ~1.7x; cost envelope)
#   - [ckpt] interval: 50 -> 25  (denser eval grid; tighter death-recovery)
#   - [ckpt] keep_last: 6 -> 6  (unchanged)
#   - [orchestrator] batch_size: 32 -> 16  (12B memory pressure)
#   - [orchestrator] group_size: 16 -> 8   (12B + max-model-len 2560)
#   - seq_len: 5120 -> 4096
#   - max_completion_tokens: 2048 -> 1024  (align with eval; remove footgun)
#   - LAUNCH inference at --gpu-memory-utilization 0.40 --max-model-len 2560
#     (Qwen 7B used 0.30/5120; Gemma 12B weights alone are 23.9 GB)

max_steps = 200
seq_len = 4096

[ckpt]
interval = 25
keep_last = 6

[model]
name = "google/gemma-4-12B-it"

[trainer.model]
impl = "auto"

[trainer.model.ac]
freq = 1

[trainer.model.lora]
rank = 32
alpha = 64
# target_modules: the 7-projection standard set, restricted to language_model.
# Vision/audio sub-trees in Gemma 4 Unified are dead weight for our text-only task.
# Module path pattern verified at install time (Task 1.4 step 7).
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[trainer.optim]
lr = 5e-6

[trainer.weight_broadcast]
type = "filesystem"

[trainer.dataset]
id = "vtemian/dockermin-v1"

[trainer.wandb]
project = "dockermin"
name = "v4-gemma4-12b-it"
offline = true

[orchestrator]
batch_size = 16
group_size = 8

[orchestrator.wandb]
project = "dockermin"
name = "v4-gemma4-12b-it"
offline = true

[orchestrator.train.sampling]
max_completion_tokens = 1024

[[orchestrator.filters]]
type = "zero_advantage"
enforce = false

[[orchestrator.train.env]]
id = "dockermin-env"
name = "dockermin-env"
args = {}

[inference]

[orchestrator.renderer]
name = "default"
```

**Step 2: Quality gate**

```bash
make quality
```

Expected: PASS. (TOML is config-only; ruff/mypy/pytest don't read it.)

**Step 3: Commit**

```bash
git checkout -b feat/v4-config
git add configs/dockermin_v4.toml
git commit -m "feat(config): add v4 GRPO config for Gemma 4 12B-it base"
```

### Task 2.2: Pod-ops launcher `launch_v4.sh`

**Files:**
- Create: `scripts/pod_ops/launch_v4.sh` — copy `launch_v1_1.sh` and swap:
  - Config path → `dockermin_v4.toml`.
  - `--gpu-memory-utilization 0.30 --model.max-model-len 5120` → `--gpu-memory-utilization 0.40 --model.max-model-len 2560`.
  - `--rdzv-id=dockermin-v1-1` → `--rdzv-id=dockermin-v4`.
  - Add `DOCKER='sg docker -c'` and replace direct `docker` calls in any inline build pre-warm with `$DOCKER "docker ..."` (mirror `run_gemma_probe.sh:25,79`).
- Create: `scripts/pod_ops/launch_v4_resume.sh` — copy `launch_resume.sh` with the same swaps; `RESUME_STEP` env var.

**Step 1: Write script** (full content).

**Step 2: shellcheck**

```bash
shellcheck scripts/pod_ops/launch_v4.sh scripts/pod_ops/launch_v4_resume.sh
```

Expected: 0 errors.

**Step 3: Commit**

```bash
git add scripts/pod_ops/launch_v4.sh scripts/pod_ops/launch_v4_resume.sh
git commit -m "feat(pod_ops): v4 launcher + resume (Gemma 12B colocation knobs)"
```

### Task 2.3: post_train + hf_pusher repo-name guards

**Files:**
- Modify: nothing (existing scripts already require `HF_REPO` env var). Document the v4 repo: `vtemian/dockermin-gemma4-12b-it-lora-v4`.
- Verify: `scripts/pod_ops/post_train.sh:15` watchdog default is 86400 (24h). For v4 the high-cost envelope is 38h; `WATCHDOG_SECONDS=144000` (40h) is the right override at launch — this caps spend at $1.79 × 40 = $71.60, inside the $80 envelope.

**Step 1: Document the override in this plan; no script change.**

The v4 launch invocation MUST set:

```bash
HF_REPO=vtemian/dockermin-gemma4-12b-it-lora-v4 \
MAX_STEPS=200 \
WATCHDOG_SECONDS=144000 \
nohup ./scripts/pod_ops/post_train.sh "$POD_ID" >>"$HOME/post_train.log" 2>&1 &
```

No commit (documentation-only step; appears in the runbook at Phase 5).

---

## Phase 3 — Single-step smoke test (cheap, before full arc)

### Task 3.1: Spin a $5 smoke pod

**Goal:** Verify the full install + 1 GRPO step succeeds before paying for 200 steps.

**Steps:**

```bash
# 1. Spin pod (10 min)
prime pods create --gpu-type A100_80GB --provider massedcompute
POD_ID=$(prime pods list --json | jq -r '.[0].id')

# 2. Install (one-shot)
prime pods ssh "$POD_ID" "DOCKERMIN_REV=feat/v4-config bash -s" < scripts/pod_ops/install_v4.sh

# 3. Override config to max_steps=1, ckpt.interval=1
prime pods ssh "$POD_ID" "sed -i 's/^max_steps = 200/max_steps = 1/; s/^interval = 25/interval = 1/' ~/dockermin/configs/dockermin_v4.toml"

# 4. Launch
prime pods ssh "$POD_ID" "bash ~/dockermin/scripts/pod_ops/launch_v4.sh"

# 5. Tail trainer for ~25 minutes; success criteria:
#    - inference: "Application startup complete"
#    - trainer: "Step 1 |" line
#    - orchestrator: "rollout done" or equivalent advantage line
#    - One STABLE broadcast file in ~/prime-rl/outputs/run_default/broadcasts/step_1/

# 6. If GREEN: terminate, proceed to Phase 4.
# 7. If RED: capture last 100 lines of each log to docs/decisions/2026-06-09-v4-smoke-fail.md; do NOT proceed.
```

**Expected smoke budget:** ~1h pod × $1.23 = **$1.23**.

**Failure escalation:** if smoke fails on a known issue from research (head_dim=512 Flash-Attn, `final_logit_softcapping`, gradient amplification), see "What could invalidate this plan" section below for the fallback decision tree.

### Task 3.2: Document smoke result

**Files:**
- Create: `docs/decisions/2026-06-09-v4-smoke.md` — single page, 3 sections:
  - "Smoke pass/fail"
  - "What broke (if anything)"
  - "Go/no-go for full arc"

If smoke fails, this is where Vlad decides whether to proceed, fall back to D1 option A (separate venv), or kill the project.

---

## Phase 4 — Full training arc

### Task 4.1: Launch 200-step run

**Pre-flight:**

```bash
prime --plain wallet  # confirm balance >= $80
gh pr view 22  # confirm PR #22 (gemma_zs baseline) is merged into main, OR rebase v4 branch on top
```

**Launch:**

```bash
POD_ID=$(prime pods create --gpu-type A100_80GB --provider massedcompute --output id)

# Install + bg pushers + post_train all in one ssh batch
prime pods ssh "$POD_ID" bash <<'SSH'
set -e
export DOCKERMIN_REV=feat/v4-config
bash ~/dockermin/scripts/pod_ops/install_v4.sh
export HF_REPO=vtemian/dockermin-gemma4-12b-it-lora-v4
export MAX_STEPS=200
export WATCHDOG_SECONDS=144000
nohup bash ~/dockermin/scripts/pod_ops/hf_pusher.sh >>~/hf_pusher.log 2>&1 &
nohup bash ~/dockermin/scripts/pod_ops/post_train.sh "$POD_ID" >>~/post_train.log 2>&1 &
nohup bash ~/dockermin/scripts/pod_ops/launch_v4.sh >>~/launch.log 2>&1 &
SSH
```

### Task 4.2: Watch for the $200 pause checkpoint

`cost_log.md` cumulative is **~$133.49** as of 2026-06-08 (sum of all completed sessions). $200 pause triggers at **+$66.51** spent on the v4 arc.

**Pause logic:**

If `trainer.log` shows `Step <N> |` where `N < 100` at the moment cumulative spend hits $200, **pause and surface to Vlad**:

- Pause command: `prime pods stop "$POD_ID"` (stop, not terminate — preserves disk for resume).
- Surface: "v4 arc at step <N>/200, cumulative $200, GRPO reward trend = <X>/<Y>; continue or kill?"

If `N >= 100` (over halfway), the pause is informational only; proceed.

### Task 4.3: Resume on death

If the pod dies UNKNOWN before step 200:

```bash
LAST_STEP=$(curl -s "https://huggingface.co/api/models/vtemian/dockermin-gemma4-12b-it-lora-v4/tree/main" | jq -r '.[] | .path' | grep -oE 'step_[0-9]+' | sort -V | tail -1 | sed 's/step_//')
POD_ID=$(prime pods create --gpu-type A100_80GB --provider massedcompute --output id)
prime pods ssh "$POD_ID" "DOCKERMIN_REV=feat/v4-config RESUME_STEP=$LAST_STEP bash ~/dockermin/scripts/pod_ops/launch_v4_resume.sh"
```

If the second pod also dies before step 200, **stop and surface** — do not auto-restart a third time without Vlad's call.

---

## Phase 5 — Holdout eval

### Task 5.1: Spin eval pod after training adapter is on HF

```bash
EVAL_POD=$(prime pods create --gpu-type A100_80GB --provider massedcompute --output id)
prime pods ssh "$EVAL_POD" bash <<'SSH'
set -e
DOCKERMIN_REV=feat/v4-config bash ~/dockermin/scripts/pod_ops/install_v4.sh
cd ~/dockermin
python scripts/run_eval.py \
  --baselines dockermin_v4 \
  --holdout vtemian/dockermin-v0 \
  --temperature 0.2 \
  --max-new-tokens 1024 \
  --out ~/dockermin/data/eval/dockermin_v4_holdout.jsonl
hf upload vtemian/dockermin-gemma4-12b-it-lora-v4 ~/dockermin/data/eval/dockermin_v4_holdout.jsonl eval/results.jsonl
SSH
```

**Eval budget:** 1h × $1.23 = **$1.23** (mirror gemma_zs probe).

### Task 5.2: Compute metrics

**Files:**
- Run: `make leaderboard` → renders `docs/leaderboard.md`.
- Append row: `| dockermin_v4 | 37 | <pass>% | <reduction>% | <elapsed_s> | <n_reductions> | 0 |`.

### Task 5.3: Decision note

**Files:**
- Create: `docs/decisions/2026-06-09-v4-result.md` — single page:
  - Headline numbers (pass-rate, reduction|pass, total bytes saved).
  - Decision rule application:
    ```
    success = (pass_rate >= 81.1%) AND (reduction|pass >= 50%)
    ```
  - If SUCCESS: leaderboard updated, PR opened, project step complete.
  - If FAIL: write `NEGATIVE_RESULT.md` per project constraint memory — do not pivot to SFT/distillation/prompt-engineering. Document the failure mode and ship.

### Task 5.4: Update cost log

**Files:**
- Modify: `docs/cost_log.md` — append v4 entries (1 train pod, 1 eval pod, maybe 1 smoke pod).

### Task 5.5: PR + ship

```bash
gh pr create --title "feat(v4): Gemma 4 12B-it GRPO arc" \
  --body "$(cat docs/decisions/2026-06-09-v4-result.md)"
```

---

## Kill switches

| Cost checkpoint | Trigger | Action |
|---|---|---|
| $200 cumulative (= +$66.51 on v4) | Hit anywhere in Phase 4 | Pause pod (`prime pods stop`); surface to Vlad before resume. |
| $400 cumulative | Should NEVER be reached by v4 envelope; if so, hard kill | One more run only; per project memory rule. |
| 40h watchdog | `post_train.sh` `WATCHDOG_SECONDS=144000` | Self-terminates pod, pushes last STABLE ckpt. |
| Smoke fail (Task 3.1) | Trainer can't reach step 1 in 25 min | Stop; do not proceed to full arc. |
| Two-strike resume rule (Task 4.3) | 2nd pod also dies before step 200 | Stop; surface; do not auto-restart. |
| Reward collapse | `orchestrator.log` shows reward mean < 0.10 for 20 consecutive steps after step 50 | Surface to Vlad; pause; investigate gradient pathology (issue #2362 risk). |

---

## What could invalidate this plan

1. **PR #2347 stays unmerged AND vendor-patching it fails.** Fallback: D1 option A (separate trainer venv). Adds ~3 days of yak-shaving on prime-rl transitive deps.
2. **Gradient explosion per issue #2362.** Issue documents 1k–1M pre-clip grad norms on Gemma 4 SFT+LoRA, unresolved. No GRPO repro on file. If it manifests on our GRPO run, the smoke test in Phase 3 catches it before we pay for 200 steps. Fallback: switch to Gemma 4 9B (smaller, but reduction-on-pass may regress vs the 12B probe number). **Important:** this is NOT a pivot to SFT/distillation — still GRPO on a smaller GRPO-amenable base.
3. **vLLM nightly wheel mismatch with pod CUDA.** vLLM nightly publishes cu128/cu129 wheels; the probe ran on cu124 base image. If the wheel doesn't import, fall back to building vLLM from source on the pod (adds ~30 min to install). If THAT fails, fall back to **massedcompute's `1xA100 80GB PCIe (cu128)` image** (different image, same provider).
4. **Cost envelope overrun.** If smoke test reveals step-time > 10 min (>2.3× v2), the 200-step arc projects $90+. Scope cut: drop to `max_steps=150` (75% of plan), still inside $80.
5. **Eval pass-rate >= 81.1% BUT reduction|pass < 50%.** This is the **Branch-B veto firing on v4 too**. NEGATIVE_RESULT.md per project memory; no pivots. Document what the GRPO arc could not lift (presumably Gemma's conservative PHP/Java rows) and ship.
6. **The gemma_zs anchor was an outlier.** Probe was n=37 with MDE ~23 pp. If we re-eval the gemma_zs baseline on a different pod day and get a different number, the "≥ 81.1%" floor moves. Mitigation: the v4 eval pod (Task 5.1) re-runs gemma_zs alongside dockermin_v4 in the SAME pod for an apples-to-apples comparison.

---

## Cost envelope summary

| Phase | Cost (low) | Cost (median) | Cost (high) |
|---|---|---|---|
| Phase 3 smoke | $1.23 | $1.23 | $2.46 (if retry) |
| Phase 4 train arc (200 steps) | $42.00 | $48.00 | $54.00 |
| Phase 4 resume buffer (one death) | $0 | $5.00 | $12.00 |
| Phase 5 eval | $1.23 | $1.23 | $2.46 |
| **Total v4** | **$44.46** | **$55.46** | **$70.92** |

Cumulative at v4 end: $133.49 (current) + $55.46 (median) = **$188.95** — inside the $200 pause checkpoint.

---

## Open questions for Vlad before `/ship-it`

1. **D1**: confirm Option B (bump prime-rl pin) is the right play; if PR #2347 is unmerged at execution start, OK to vendor-patch it?
2. **D2**: confirm vLLM nightly `0.22.1rc1.dev+` is acceptable (pre-release).
3. **D3**: confirm G=8 / max_completion_tokens=1024 / 200 steps; these are the dollar-fit defaults.
4. **D4**: confirm massedcompute primary, crusoe fallback.
5. **Branch-B veto trigger on v4**: confirm the "pass ≥ 81.1% AND reduction|pass ≥ 50%" success rule binds; below either threshold → NEGATIVE_RESULT.md, no pivot.
6. **Scope-cut threshold**: if smoke test reveals step-time > 10 min, OK to drop max_steps from 200 to 150 without re-running this plan?

---

**Plan complete. Save path: `docs/plans/2026-06-09-v4-gemma-grpo.md`. Awaiting `/ship-it` or explicit go-ahead per planning-pipeline rules.**
