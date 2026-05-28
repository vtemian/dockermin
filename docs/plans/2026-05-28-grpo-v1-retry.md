# GRPO v1 Retry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> ## v1.1 AMENDMENTS (2026-05-28, post-multi-agent-audit) — read these first
>
> After v1's Phases A–C shipped, four parallel subagents (empirical reward simulation,
> published-best-practice comparison, v0 forensic analysis, devil's-advocate review)
> independently converged on the same diagnosis: **v1-as-originally-specified would have
> produced another null result** because the smoothing was undersized vs the real
> `command_count` distribution, the pad-the-Dockerfile reward hack was wide open, and
> the deferred batch-size bump was actually the binding constraint. Estimated probability
> of a ≥5pp lift was ~15–25% before amendments; ~60–70% after.
>
> Four within-technique fixes were applied (Phases A1, B1, plus new H/I below) and are
> already on `main`:
>
> 1. **Reward magnitudes recalibrated** (`src/dockermin/reward/gates.py`):
>    `CMD_CREDIT_PER_CMD: 0.005 → 0.02`,
>    `BUILD_FAIL_CMD_CREDIT_MAX: 0.10 → 0.30`,
>    `TEST_FAIL_SCORE: 0.05 → 0.35`,
>    `TEST_FAIL_CMD_CREDIT_MAX: 0.15 → 0.10`.
>    Saturation knee moves from cc=20 (median dataset cc=19, so 47% saturated) to cc=15.
>    Rung ordering preserved: max build_fail (0.30) < min reachable test_fail (0.39) < pass floor (0.50). Commit `d1eed5f`.
>
> 2. **Pad-hack closed** (gates.py + dockermin_reward.py + dockermin_env.py):
>    `_cmd_partial_credit` and `compute_score` now take a `baseline_command_count` arg;
>    when provided, effective cc is capped at `min(cc, baseline_cc + 2)` so the model
>    cannot win the failure-rung credit by emitting no-op `LABEL`/`RUN echo` lines.
>    The env adapter parses the prompt Dockerfile's command_count via `parse_gate` and
>    injects it into `info`. Commit `d1eed5f` (same as #1).
>
> 3. **Effective batch size 4× larger** (`configs/dockermin_full.toml`):
>    `batch_size: 8 → 16`, `group_size: 8 → 16` (256 rollouts/step vs the prior 64;
>    still 8× below the smallest published prime-rl example, but P(all-degenerate group)
>    drops from ~50% to <1%). To keep total compute roughly constant, `max_steps: 500 → 250`.
>    Commit `0889046`.
>
> 4. **Early-warning panel** (`scripts/grpo_early_warning.py`, 491 LOC):
>    Pod-side single-invocation CLI computing 8 metrics (cap saturation, group stdev,
>    command_count slope, unique-completions ratio, trainer entropy, stuck-prompt fraction,
>    adapter L2 drift at step 25, grad-norm median) over a step window. Exits 1 on any
>    RED. **Replaces Phase E3's single stdev check with this 8-metric panel.**
>    Commit `85764be`.
>
> ### Phase E3 SUPERSEDED — use the early-warning script instead
>
> The plan's original Phase E3 (manual `stdev > 0.05` reading on 3 sampled steps) is
> superseded. On the pod, at step 10 AND step 20, run:
>
> ```bash
> ssh ... "cd ~/prime-rl && .venv/bin/python ~/dockermin/scripts/grpo_early_warning.py \
>   --run-dir outputs/run_default --step-window 1,STEP"
> ```
>
> (replace STEP with 10 then 20). If exit code is 1 (any RED), **terminate the pod immediately**
> rather than letting the 14h training run silently produce noise. If all GREEN or only YELLOW,
> proceed. Phase E3 budget guard still applies (16h watchdog backstop).
>
> ### Cost target unchanged
>
> The v1.1 amendments are pure code/config changes with no pod cost. Same ~$40–55 target,
> same ~$58 hard cap. Wallet $50 pre-flight gate at D2 still applies.
>
> ---

**Goal:** Honestly test the GRPO hypothesis on dockermin by fixing the three structural reasons v0 produced a null result (151/200 zero-advantage crash, no periodic checkpoints, cliff-shaped reward that collapses ~40% of rollouts to a single value), then re-evaluating against the same holdout. Outcome is binary: either v1 beats Sonnet zero-shot on the kill-criterion holdout test (the experiment worked), or we ship `NEGATIVE_RESULT.md` per Vlad's pre-commitment.

**Architecture:** Three within-technique changes, no algorithm pivot:
1. **Reward reshape** (`gates.py`): add a per-instruction-count partial-credit on the `build_fail` and `test_fail` rungs so rollouts that BOTH fail at the same gate but differ in Dockerfile completeness no longer score identically. Targets the documented GRPO-killer: build_fail (0.0) absorbs ~40% of rollouts on early training, and 8-roll groups collapse to all-0.0 reward → zero advantage → wasted batch.
2. **Crash-proof orchestrator config**: set `[[orchestrator.filters]] type="zero_advantage" enforce=false` so the orchestrator stops *removing* zero-advantage rollouts (they get sent to the trainer with their actual reward, contributing nothing but never collapsing a batch). The MAX_EMPTY_BATCH_ATTEMPTS=3 crash path at `orchestrator.py:421-433` is never reached. Also add periodic checkpointing (`[ckpt] interval=25 keep_last=4`) so a mid-run failure is recoverable AND we have intermediate adapters to diagnose learning curves.
3. **Longer schedule**: `max_steps = 500` (up from 200; v0 only got 151 effective steps). All other knobs proven in v0 stay: `seq_len=5120`, `max_completion_tokens=2048`, `batch_size=8`, `group_size=8`, `[trainer.model.lora] rank=32 alpha=64`, inference 0.30 / max-model-len 5120.

**Tech Stack:** prime-rl (main branch, pin captured in `docs/decisions/2026-05-22-prime-rl-pin.md`), Qwen2.5-Coder-7B-Instruct + LoRA r32, transformers + peft, vLLM, dockermin reward/annotate, 1×H100 80GB pod (proven recipe).

---

## Context (read before executing)

The v0 result was robustly null (per-triple diagnostic in `docs/journal.md` 2026-05-27 entries): dockermin-v0 vs base Qwen 2.5 Coder 7B → 56.8% vs 56.8% test-pass, 32.6% vs 32.1% reduction (Δ +0.5pp, noise). Sonnet zero-shot beat both (91.9% / 38.9%). The three structural causes — and the subagent evidence for each:

- **GRPO collapsed at step 151/200** with `RuntimeError: All 8 rollouts were filtered out on 3 consecutive attempts`. Source: `prime-rl/src/prime_rl/orchestrator/orchestrator.py:421-433`, `MAX_EMPTY_BATCH_ATTEMPTS=3` at line 81. Filter at `filters.py:97-112` (`ZeroAdvantageFilter`) defaults to `enforce=True`. **Fix:** flip to `enforce=false` via `ZeroAdvantageFilterConfig` (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:475-481`) — official config flag, no code patch needed.
- **No mid-run checkpoint.** v0 used a bare `[ckpt]` (saves only at end). The trainer died with the orchestrator → no end-checkpoint; the only reason we recovered an adapter was the per-step `outputs/run_default/broadcasts/step_<N>/adapter_model.safetensors` written by the filesystem weight-broadcast path. **Fix:** `[ckpt] interval=25 keep_last=4` (full schema at `prime-rl-configs/src/prime_rl/configs/orchestrator.py:350-371`).
- **Reward cliff.** Current scores (`src/dockermin/reward/gates.py`): `parse_fail=-0.1`, `too_few=-0.2`, **`build_fail=0.0`**, `test_fail=0.05`, `pass∈[0.5, 1.0]`. Eval shows ~40% of all rollouts (qwen_zs *and* dockermin) hit `build_fail` → identical 0.0 score → zero advantage in any GRPO group with ≥5 build-fails (≈ 50% of all groups by binomial CDF at p=0.378, k≥5, n=8). **Fix:** add a small per-command-count partial credit on the build_fail and test_fail rungs so two distinct-but-broken Dockerfiles do not score identically.

The dataset (108 train + 37 test on `vtemian/dockermin-v0`) is **kept as-is** for v1 — growing it is a separate lever and a separate plan. We're testing the GRPO setup with the dataset we have, so the v0→v1 delta is attributable to the GRPO changes alone.

## Pre-commitment (do not relax)

- **Within-technique only.** v1 is GRPO with three fixes. NOT SFT, NOT distillation, NOT prompt engineering, NOT activation steering. See `~/.claude/projects/.../memory/project_dockermin_constraints.md` "Committed technique" block.
- **If v1 still misses the kill-criterion** (does not strictly beat Sonnet zero-shot reduction on ≥30% of holdouts), ship `NEGATIVE_RESULT.md` — *do not* pivot to a different technique. v0+v1 together become the honest experimental record.
- **Budget guard.** Hard cap the run at 16h watchdog (~$50 on 1×H100). If the eval pod balance falls below $15 before the v1 retrain even launches, pause and ask Vlad.

## Out of scope (explicitly deferred)

- Dataset growth past 28 bases (separate plan if v1 also fails; the levers and costs are documented in the subagent report).
- `agent_loop` kill-criterion (~$18 Anthropic API) — `sonnet_zs` will remain the Sonnet bar; if v1 fails to beat `sonnet_zs`, `agent_loop` would only widen the gap.
- `group_size` increase (8 → 16). Proven 8 is what v0 used; isolating the effect of the three fixes matters more than adding a fourth variable in v1.
- Resuming v1 from v0's `step_151` broadcast. The orchestrator's `checkpoints/step_151/` (buffer + progress) doesn't exist from v0 (we only have `broadcasts/`), so resume can't reload buffer state. v1 is a fresh-start run.

---

## Phase A — Reward reshape (off-pod, TDD)

Goal: replace the flat `build_fail=0.0` and `test_fail=0.05` rungs with smooth, command-count-aware partial credit. Two failing builds that produced a 4-instruction stub vs a 22-instruction reasonable Dockerfile now score differently → GRPO group has reward variance → non-zero advantage.

### Task A1: Write failing tests for the smoothed reward

**Files:**
- Modify: `tests/test_gates.py` (or `tests/test_reward.py` if `test_gates.py` does not exist — first run `ls tests/` to confirm)

**Step 1: Confirm where the existing `compute_score` tests live**

Run: `grep -l "compute_score\|BUILD_FAIL_SCORE" tests/ -r`
Expected: at least one test file path printed. Use that file for new tests.

**Step 2: Append failing tests to that file**

```python
import pytest
from dockermin.reward.gates import (
    compute_score,
    BUILD_FAIL_SCORE,
    TEST_FAIL_SCORE,
    BUILD_FAIL_CMD_CREDIT_MAX,
    TEST_FAIL_CMD_CREDIT_MAX,
    CMD_CREDIT_PER_CMD,
)


def test_build_fail_partial_credit_scales_with_command_count() -> None:
    """A 5-instruction build-fail must score lower than a 20-instruction build-fail,
    so GRPO groups with mixed-complexity build failures have non-zero reward variance."""
    small = compute_score(
        parse_ok=True, build_ok=False, test_ok=False,
        command_count=5, baseline_size=100, new_size=0, dockerfile_text="",
    )
    big = compute_score(
        parse_ok=True, build_ok=False, test_ok=False,
        command_count=20, baseline_size=100, new_size=0, dockerfile_text="",
    )
    assert small == pytest.approx(BUILD_FAIL_SCORE + 5 * CMD_CREDIT_PER_CMD)
    assert big > small
    assert big <= BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX  # cap honoured


def test_build_fail_partial_credit_saturates_at_cap() -> None:
    """Even a 200-instruction build-fail must not score higher than the cap, otherwise
    a degenerate verbose Dockerfile could out-reward a buildable one."""
    huge = compute_score(
        parse_ok=True, build_ok=False, test_ok=False,
        command_count=200, baseline_size=100, new_size=0, dockerfile_text="",
    )
    assert huge == pytest.approx(BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX)


def test_test_fail_partial_credit_separates_from_build_fail() -> None:
    """A test-failing build must always score higher than a build-failing one with the
    same command count — preserves the gate order even with smoothing."""
    bf = compute_score(
        parse_ok=True, build_ok=False, test_ok=False,
        command_count=10, baseline_size=100, new_size=0, dockerfile_text="",
    )
    tf = compute_score(
        parse_ok=True, build_ok=True, test_ok=False,
        command_count=10, baseline_size=100, new_size=80, dockerfile_text="",
    )
    assert tf > bf


def test_full_pass_unchanged_by_smoothing() -> None:
    """The pass-rung formula (0.5 + 0.5*dense + shape) must not be touched."""
    s = compute_score(
        parse_ok=True, build_ok=True, test_ok=True,
        command_count=5, baseline_size=1000, new_size=500, dockerfile_text="",
    )
    # 50% reduction, no shape -> 0.5 + 0.5*0.5 = 0.75
    assert s == pytest.approx(0.75)


def test_parse_fail_unchanged() -> None:
    """The parse-fail rung is not smoothed (a non-parseable Dockerfile has no useful
    command_count to credit)."""
    s = compute_score(
        parse_ok=False, build_ok=False, test_ok=False,
        command_count=0, baseline_size=100, new_size=0, dockerfile_text="",
    )
    from dockermin.reward.gates import PARSE_FAIL_SCORE
    assert s == PARSE_FAIL_SCORE
```

**Step 3: Run tests, verify they fail with `ImportError` / `AttributeError`**

Run: `make test-pure 2>&1 | grep -E "FAILED|ERROR|cannot import"`
Expected: ImportError on `BUILD_FAIL_CMD_CREDIT_MAX` (the constant does not exist yet). This is the correct RED state.

**Step 4: Commit the failing tests**

```bash
git checkout -b feat/grpo-v1-reward-smoothing
git add tests/  # whichever file you edited
git commit -m "test(reward): smoothed build_fail/test_fail credit by command count"
```

---

### Task A2: Implement the smoothed reward

**Files:**
- Modify: `src/dockermin/reward/gates.py`

**Step 1: Read the current `compute_score` carefully**

Run: `sed -n '74,101p' src/dockermin/reward/gates.py`
Expected: the existing gate-ladder loop using `gate_failures` tuple. You will refactor it into explicit `if` checks (preserving order) so we can inject partial credit per gate.

**Step 2: Add the smoothing constants near the top of the file**

Insert after the existing `TEST_FAIL_SCORE = 0.05` line:

```python
# Smoothing on the failure rungs. Without this, ~40% of early-training rollouts
# collapse to BUILD_FAIL_SCORE (0.0) — and 8-rollout GRPO groups with ≥5
# build-fails have zero advantage → wasted batch. Adding a small command-count
# credit ensures two distinct-but-broken Dockerfiles do not score identically.
# Caps are kept well below TEST_FAIL_SCORE→PASS gradient (0.05 → 0.5) so the
# gate-order incentives (broken < buildable < passing) are preserved.
CMD_CREDIT_PER_CMD = 0.005
BUILD_FAIL_CMD_CREDIT_MAX = 0.10  # saturates at command_count >= 20
TEST_FAIL_CMD_CREDIT_MAX = 0.15  # buildable-but-failing earns more than broken
```

**Step 3: Add a small helper above `compute_score`**

```python
def _cmd_partial_credit(command_count: int, cap: float) -> float:
    """Per-instruction partial credit on the failure rungs, capped."""
    return min(cap, CMD_CREDIT_PER_CMD * max(0, command_count))
```

**Step 4: Refactor `compute_score`'s gate-ladder**

Replace the existing `gate_failures` tuple + loop with explicit per-gate returns. The full new body of `compute_score` (keep the docstring + signature unchanged):

```python
def compute_score(  # noqa: PLR0913
    *,
    parse_ok: bool,
    build_ok: bool,
    test_ok: bool,
    command_count: int,
    baseline_size: int,
    new_size: int,
    dockerfile_text: str,
) -> float:
    """Composite reward. Keyword-only API is fixed by callers and tests."""
    if not parse_ok:
        return PARSE_FAIL_SCORE
    if command_count < MIN_COMMANDS:
        return TOO_FEW_COMMANDS_SCORE
    if not build_ok:
        return BUILD_FAIL_SCORE + _cmd_partial_credit(command_count, BUILD_FAIL_CMD_CREDIT_MAX)
    if not test_ok:
        return TEST_FAIL_SCORE + _cmd_partial_credit(command_count, TEST_FAIL_CMD_CREDIT_MAX)
    reduction = max(0.0, (baseline_size - new_size) / max(1, baseline_size))
    dense = min(1.0, reduction)
    text = dockerfile_text.lower()
    shape = _shape_bonus(text) + _shape_penalty(text) + _tiny_image_penalty(text, baseline_size, new_size)
    return min(1.0, 0.5 + 0.5 * dense + shape)
```

**Step 5: Run the new tests, verify GREEN**

Run: `make test-pure 2>&1 | tail -10`
Expected: ALL tests pass (139 existing + 5 new = 144).

**Step 6: Run the full quality gate**

Run: `make quality 2>&1 | tail -5`
Expected: all green (ruff format-check + lint + mypy + pytest).

**Step 7: Commit**

```bash
git add src/dockermin/reward/gates.py
git commit -m "feat(reward): smoothed build_fail/test_fail by command count to break GRPO zero-advantage plateau"
```

---

### Task A3: Update the deterministic smoke script's expectations

The existing `scripts/smoke_reward_replay.py` (4 cases, real Docker) hard-codes expected score ranges. The `build failure` case still scores 0.0 in the script (the alpine `RUN exit 1` example has command_count=3 → +0.015 credit → 0.015 not 0.0). Update the assertion text so the smoke still passes locally.

**Files:**
- Modify: `scripts/smoke_reward_replay.py`

**Step 1: Read the existing cases**

Run: `sed -n '20,60p' scripts/smoke_reward_replay.py`
Note the `build failure (RUN exits non-zero)` case's `expect=` string.

**Step 2: Update only the build-fail case's `expect` text** so the printed expectation matches the new behaviour:

```python
    Case(
        name="build failure (RUN exits non-zero)",
        completion=_fenced('FROM alpine:3.20\nRUN exit 1\nCMD ["true"]'),
        info={"baseline_size": 100_000_000, "test_cmd": ["sh", "-c", "echo PYOK"], "expected_substring": "PYOK"},
        expect="build-fail score (~0.015, BUILD_FAIL_SCORE + small per-cmd credit)",
    ),
```

**Step 3: Don't run the smoke yet** — it needs Docker locally. The pod will exercise this path. Just confirm syntax:

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/smoke_reward_replay.py').read())"`
Expected: no output (parse OK).

**Step 4: Commit**

```bash
git add scripts/smoke_reward_replay.py
git commit -m "chore(smoke): update build-fail expectation comment for smoothed reward"
```

---

### Task A4: Merge Phase A to main

**Step 1: Quality gate one more time**

Run: `make quality 2>&1 | tail -3`
Expected: green.

**Step 2: Merge & push (separate Bash calls per the commit-guard hook)**

Call 1: `git checkout main`
Call 2: `git merge --ff-only feat/grpo-v1-reward-smoothing && git push origin main && git branch -d feat/grpo-v1-reward-smoothing`

Verify with: `git log --oneline -3`
Expected: top commits include the reward-smoothing test + impl + smoke-doc update.

---

## Phase B — Config + watchdog updates (off-pod)

Goal: enable periodic checkpointing, soften the zero-advantage filter so the orchestrator never crashes, extend `max_steps`, and bump the watchdog cap. All changes are TOML.

### Task B1: Update `configs/dockermin_full.toml`

**Files:**
- Modify: `configs/dockermin_full.toml`

**Step 1: Read the current file**

Run: `cat configs/dockermin_full.toml`
Confirm: `max_steps = 200`, `[ckpt]` present but bare, no `[[orchestrator.filters]]` block.

**Step 2: Apply the three changes**

Change `max_steps = 200` → `max_steps = 500`.

Replace the bare `[ckpt]` block with:

```toml
# Periodic checkpoints so a mid-run failure is recoverable AND we have learning-curve
# snapshots for the post-mortem. interval=25 keeps step_25, 50, 75, ..., 500 if the
# run completes. keep_last=4 caps disk to ~4 LoRA snapshots (~1.3GB total).
[ckpt]
interval = 25
keep_last = 4
```

Add a new `[[orchestrator.filters]]` block (place it after `[orchestrator.train.sampling]`):

```toml
# enforce=false means the orchestrator only TRACKS zero-advantage rollouts in
# metrics (no skip, no batch rejection). This prevents the MAX_EMPTY_BATCH_ATTEMPTS
# crash that ended v0 at step 151/200 — degenerate-reward batches still cost a
# wasted gradient step, but the run survives them.
[[orchestrator.filters]]
type = "zero_advantage"
enforce = false
```

**Step 3: Validate TOML parses**

Run:
```bash
python3 -c "import tomllib; d=tomllib.load(open('configs/dockermin_full.toml','rb')); print('max_steps=', d['max_steps']); print('ckpt=', d['ckpt']); print('filters=', d['orchestrator']['filters'])"
```
Expected:
```
max_steps= 500
ckpt= {'interval': 25, 'keep_last': 4}
filters= [{'type': 'zero_advantage', 'enforce': False}]
```

**Step 4: Commit on a fresh branch**

```bash
git checkout -b feat/grpo-v1-config
git add configs/dockermin_full.toml
git commit -m "feat(grpo-v1): max_steps=500, ckpt every 25, zero_advantage enforce=false"
```

---

### Task B2: Bump the eval-unattended watchdog (not used in training, but keep in sync)

**Files:**
- Modify: `scripts/eval_unattended.sh`

**Step 1: Confirm current watchdog**

Run: `grep WATCHDOG_SECONDS scripts/eval_unattended.sh`
Expected: `WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-21600}"  # 6h hard cap` (or similar after the v0 manual extension).

**Step 2: No change required for v1 retraining** (this script is for eval, not training). Skip this task — note in commit message that the training run will use its own pod-side launch with its own watchdog (set in Phase E).

---

### Task B3: Merge Phase B

Call 1: `git checkout main`
Call 2: `git merge --ff-only feat/grpo-v1-config && git push origin main && git branch -d feat/grpo-v1-config`

---

## Phase C — Audit script fix (off-pod, optional but cheap)

The eval surfaced that `scripts/audit_rollouts.py` has the same pydantic-message-vs-dict bug the reward had (`r.get("completion","").lower()` on a list-of-AssistantMessage). Fix it so the daily reward-hacking audit actually runs during v1.

### Task C1: Test + fix audit_rollouts.py

**Files:**
- Modify: `scripts/audit_rollouts.py`
- Test: a tiny inline assertion (no unit-test file yet; this is a script)

**Step 1: Read the broken extraction line**

Run: `grep -n "completion" scripts/audit_rollouts.py`
Expected: a line doing `r.get("completion", "").lower()` (or similar).

**Step 2: Replace with the same extraction path the reward uses**

Change:
```python
texts = [r.get("completion", "").lower() for r in sample]
```
to:
```python
from dockermin.reward.dockermin_reward import _completion_text
from dockermin.reward.prompts import extract_dockerfile
texts = []
for r in sample:
    extracted = extract_dockerfile(_completion_text(r.get("completion", ""))) or ""
    texts.append(extracted.lower())
```

**Step 3: Smoke-test against a real rollout file**

Run: `find data -name "*rollouts*.jsonl" -type f | head -1` (any existing rollouts dump). If one exists:

`.venv/bin/python scripts/audit_rollouts.py $(dirname "$(find data -name '*rollouts*.jsonl' -type f | head -1)")`

Expected: a JSON stats blob with `from_scratch`, `latest_tag`, `no_cmd`, `mean_lines`. No `AttributeError`.

If no rollout file exists locally, skip the smoke and rely on Phase F validation on the pod.

**Step 4: Quality gate + commit**

Run: `make quality 2>&1 | tail -3`

```bash
git checkout -b fix/audit-rollouts-completion
git add scripts/audit_rollouts.py
git commit -m "fix(audit): extract dockerfile from list[AssistantMessage] completions, mirroring reward path"
```

Merge to main (two-call pattern).

---

## Phase D — Pod session: rent + install

Goal: bring up a 1×H100 pod, install prime-rl + dockermin + DooD using the proven recipe in `docs/decisions/2026-05-22-prime-rl-pin.md`, then dry-run the v1 config. **Do not start the training yet.**

### Task D1: Pre-flight checks (local)

**Step 1: Confirm wallet balance is sufficient for a ~$50 run**

Run: `prime wallet | grep Balance`
Expected: balance ≥ $50. If not, **stop and ask Vlad to add credit** — do not rent.

**Step 2: Confirm latest main has all three Phase A/B/C commits**

Run: `git log --oneline -5`
Expected: top commits include `feat(reward):`, `feat(grpo-v1):`, and `fix(audit):`.

### Task D2: Rent the pod

**Step 1: Check H100 availability + pick cheapest in-stock on-demand**

Run:
```bash
prime availability list --gpu-type H100_80GB --gpu-count 1 --output json | jq -r '.gpu_resources | map(select(.stock_status=="Available" and .is_spot != true)) | sort_by(.price_value) | .[:3][] | [.price_per_hour, .provider, (.vcpus|tostring), .id] | @tsv'
```
Expected: at least one provider listed. Pick the cheapest non-spot (the v0 training used massedcompute $2.35/hr; datacrunch $3.25/hr if that's gone).

**Step 2: Create the pod with explicit `--disk-size` (datacrunch needs it)**

```bash
prime pods create --id <ID> --name dockermin-grpo-v1 --disk-size 500 --yes --plain | tail -8
```
Expected: `Successfully created pod <POD_ID>`.

**Step 3: Append the cost-log row immediately, with start timestamp**

Edit `docs/cost_log.md`, add a row under the existing table:

```
| 2026-MM-DD | GRPO v1 retry (training) | <provider> | 1xH100 80GB | <ISO start> |       |       | $<rate>/hr |        |            |
```

Commit on a docs branch + push (so the row exists even if everything explodes).

**Step 4: Poll to active, capture SSH endpoint**

```bash
for i in $(seq 1 20); do
  S=$(prime pods status <POD_ID> --plain 2>/dev/null | awk -F"  +" '/^SSH /{print $2; exit}')
  [ -n "$S" ] && [ "$S" != "N/A" ] && echo "READY: $S" && break
  sleep 15
done
```
Expected: SSH line printed. Record it. The user is `ubuntu` on massedcompute, `root` on datacrunch (note port-22 explicit on datacrunch).

### Task D3: Install on the pod

Execute the proven recipe from `docs/decisions/2026-05-22-prime-rl-pin.md` ("Install recipe that worked" + "Single-GPU RL run recipe"). The exact sequence:

**Step 1: SSH in, install uv, clone prime-rl + dockermin, init submodules**

```bash
ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o PubkeyAcceptedAlgorithms=+ssh-rsa <USER@IP> '
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
git config --global url."https://github.com/".insteadOf "git@github.com:"
rm -rf ~/prime-rl ~/dockermin
git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl.git ~/prime-rl
git clone --depth 1 https://github.com/vtemian/dockermin.git ~/dockermin
cd ~/prime-rl && git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
nohup uv sync --all-extras > ~/uvsync.log 2>&1 &
'
```
Expected: clones complete, uv sync starts (~15-20 min).

**Step 2: Poll uv sync to completion**

```bash
ssh ... 'cd ~/prime-rl
for i in $(seq 1 22); do
  .venv/bin/python -c "import torch,vllm,verifiers" 2>/dev/null && echo "SYNC DONE" && break
  echo "[$i] $(tail -1 ~/uvsync.log | cut -c1-60)"
  sleep 45
done'
```
Expected: "SYNC DONE".

**Step 3: Install dockermin + eval deps (NO deps to avoid downgrading torch/vllm)**

```bash
ssh ... 'cd ~/prime-rl && export PATH=$HOME/.local/bin:$PATH
uv pip install peft
uv pip install --no-deps -e ~/dockermin -e ~/dockermin/prime_env/dockermin_env
uv pip install "docker==7.1.*" "dockerfile==3.3.1" anthropic openai tenacity
.venv/bin/python -c "import dockermin.eval.baselines, peft, docker, dockerfile; print(\"eval imports OK\")"'
```
Expected: `eval imports OK`.

**Step 4: DooD setup**

```bash
ssh ... 'bash ~/dockermin/scripts/setup_pod_docker.sh 2>&1 | tail -3'
```
Expected: `docker setup ok`.

**Step 5: Verify the v1 reward fix is on the pod**

```bash
ssh ... 'cd ~/dockermin && grep -c "_cmd_partial_credit" src/dockermin/reward/gates.py'
```
Expected: `>= 2` (the helper definition + at least one call site). If 0, the pod's clone is stale — `git pull` to fix.

### Task D4: Dry-run the v1 config

**Step 1: Dry-run**

```bash
ssh ... 'cd ~/prime-rl && export PATH=$HOME/.local/bin:$PATH
rm -rf outputs
uv run --no-sync rl @ ~/dockermin/configs/dockermin_full.toml --dry-run 2>&1 | tail -5'
```
Expected: `Dry run complete.`

If it errors on `[[orchestrator.filters]]`, the filter config block belongs under a different parent — check the orchestrator schema and adjust the TOML accordingly. The known-good location is at the top level of the orchestrator config (`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:475-481` documents the field).

**Step 2: Verify the generated sub-configs reflect the changes**

```bash
ssh ... 'grep -E "max_steps|interval|keep_last|enforce" ~/prime-rl/outputs/configs/orchestrator.toml ~/prime-rl/outputs/configs/trainer.toml | head'
```
Expected: `max_steps = 500`, `interval = 25`, `keep_last = 4`, `enforce = false` all appearing.

---

## Phase E — Pod session: launch training + verify the fixes are biting

Goal: launch the 3-process colocated training using the proven Phase-1 recipe, then watch the first ~10 steps to confirm (a) reward variance is healthier than v0 (fewer all-same-reward groups), (b) no zero-advantage crash, (c) periodic checkpoints landing on disk.

### Task E1: Launch inference

**Step 1: Inference server with the proven flags**

```bash
ssh ... 'cd ~/prime-rl && export PATH=$HOME/.local/bin:$PATH && rm -f ~/inference.log
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 setsid \
  uv run --no-sync inference @ outputs/configs/inference.toml \
  --gpu-memory-utilization 0.30 --model.max-model-len 5120 \
  > ~/inference.log 2>&1 < /dev/null &'
```

**Step 2: Poll readiness**

```bash
ssh ... 'for i in $(seq 1 16); do curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && echo UP && break; sleep 15; done'
```
Expected: `UP`.

### Task E2: Launch trainer + orchestrator

**Step 1: Trainer via torchrun (NOT bare `uv run trainer`)**

```bash
ssh ... 'cd ~/prime-rl && export PATH=$HOME/.local/bin:$PATH && export WANDB_MODE=offline && rm -f ~/trainer.log ~/orchestrator.log
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid \
  uv run --no-sync torchrun --nproc-per-node=1 --rdzv-endpoint=localhost:29512 --rdzv-id=grpov1 \
  -m prime_rl.trainer.rl.train @ outputs/configs/trainer.toml \
  > ~/trainer.log 2>&1 < /dev/null &
DOCKERMIN_MAX_BUILDS=6 setsid \
  uv run --no-sync orchestrator @ outputs/configs/orchestrator.toml \
  > ~/orchestrator.log 2>&1 < /dev/null &'
```

**Step 2: Wait for the first orchestrator step (~3-5 min)**

```bash
ssh ... 'for i in $(seq 1 20); do
  grep -qE "Step 0 \|" ~/orchestrator.log 2>/dev/null && echo "STEP 0 DONE" && break
  grep -qiE "out of memory|ChildFailedError|Fatal" ~/trainer.log ~/orchestrator.log 2>/dev/null && echo "EARLY CRASH" && break
  echo "[$i] $(tail -1 ~/orchestrator.log | tr -d "\r" | cut -c1-70)"
  sleep 30
done'
```
Expected: `STEP 0 DONE`. If EARLY CRASH: read the error, diagnose, terminate the pod.

### Task E3: Verify the fixes are actually biting

This is the critical validation: the *whole point* of v1 is that fewer rollouts collapse to identical rewards. Confirm before letting it run hours.

**Step 1: Sample reward variance from the first 3 steps' saved rollouts**

```bash
ssh ... '.venv/bin/python -c "
import json, glob, statistics
files = sorted(glob.glob(\"~/prime-rl/outputs/run_default/rollouts/step_*/train_rollouts.jsonl\"))[:3]
for f in files:
    rows=[json.loads(l) for l in open(f)]
    rewards=[r[\"reward\"] for r in rows]
    n_unique=len(set(round(r,4) for r in rewards))
    print(f.split(\"/\")[-2], \"n=\", len(rewards), \"unique=\", n_unique, \"stdev=\", round(statistics.pstdev(rewards),4) if len(rewards)>1 else 0)
"'
```
Expected criteria:
- `stdev > 0.05` on at least 2 of the first 3 steps (v0 had many steps with stdev=0 → zero advantage).
- `unique >= 3` per step (v0 had steps with unique=1, i.e., all 8 rollouts at the same reward).

If `stdev ≈ 0` and `unique ≤ 1` for all 3 sampled steps, the smoothing isn't helping enough. **STOP**, terminate the pod, write up the finding, and either revise the smoothing magnitude (Phase A1/A2 redo) or escalate to Vlad before sinking more compute.

**Step 2: Confirm `enforce=false` is honoured (no "filtered out" warnings)**

```bash
ssh ... 'grep -c "filtered out" ~/orchestrator.log'
```
Expected: `0`. (The zero_advantage filter still RUNS — it just reports detections in metrics rather than removing rollouts.)

**Step 3: Confirm the first checkpoint will land at step 25**

```bash
ssh ... 'cat ~/prime-rl/outputs/configs/orchestrator.toml | grep -A2 "\[ckpt\]"'
```
Expected: `interval = 25`, `keep_last = 4`.

### Task E4: Launch the pod-side watcher + extended watchdog

Goal: make the run resilient to my SSH session dropping for 18+ hours. The watcher writes status to a file (cheap to poll later), and a separate 16h watchdog kills the pod if anything hangs.

**Step 1: Reuse the watcher pattern from v0 training**

The proven pattern is in `docs/journal.md` 2026-05-26 entries — a `~/watch.sh` that loops every 60s writing `~/run_status.txt` and a `~/run_done` marker on completion/crash. Re-create it on the pod:

```bash
ssh ... 'cat > ~/watch.sh <<"EOF"
#!/bin/bash
cd ~/prime-rl
PEAK=0
while true; do
  TS=$(grep -oE "Step [0-9]+ \| Time" ~/trainer.log 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
  G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [ "${G:-0}" -gt "$PEAK" ] && PEAK=$G
  ORCH=$(grep -c Reward: ~/orchestrator.log 2>/dev/null)
  CKPTS=$(ls ~/prime-rl/outputs/run_default/checkpoints/ 2>/dev/null | wc -l)
  echo "$(date -u +%H:%M:%S) trainer=${TS:-0}/500 orch=$ORCH gpu=$G peak=$PEAK ckpts=$CKPTS" > ~/run_status.txt
  if grep -qiE "out of memory|ChildFailedError|Fatal error in train|All [0-9]+ rollouts were filtered" ~/trainer.log ~/orchestrator.log 2>/dev/null; then
    echo "CRASH step=$TS peak=$PEAK ckpts=$CKPTS" > ~/run_done; break
  fi
  if grep -qiE "wandb sync outputs" ~/trainer.log 2>/dev/null && [ "${TS:-0}" -ge 495 ]; then
    echo "COMPLETE step=$TS peak=$PEAK ckpts=$CKPTS" > ~/run_done; break
  fi
  sleep 60
done
EOF
rm -f ~/run_done
setsid bash ~/watch.sh > /dev/null 2>&1 < /dev/null &
sleep 2; cat ~/run_status.txt'
```
Expected: a status line printed.

**Step 2: Launch a 16h watchdog with prime-CLI self-terminate**

The watchdog needs the prime CLI authed on the pod (install + scp `~/.prime/config.json` — same pattern as the eval pod in `docs/journal.md` 2026-05-27 entries). Then:

```bash
# Local: stage prime config + HF token on the pod (if not done in Phase D)
scp -i ~/.ssh/id_rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa ~/.prime/config.json <USER@IP>:~/.prime/config.json
scp -i ~/.ssh/id_rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa ~/.cache/huggingface/token <USER@IP>:~/.cache/huggingface/token

# Pod: install prime CLI + test auth + launch 16h watchdog
ssh ... 'uv tool install -U prime --with "typer<0.26" >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
prime pods list --plain >/dev/null 2>&1 && echo "prime auth on pod OK"
setsid bash -c "sleep 57600; export PATH=\$HOME/.local/bin:\$PATH; prime pods terminate <POD_ID> --yes" >/dev/null 2>&1 < /dev/null & echo "16h watchdog launched pid=$!"'
```
Expected: `prime auth on pod OK` and `16h watchdog launched`.

**Step 3: Off-pod handoff message**

At this point the run is genuinely unattended. The pod-side watcher updates `~/run_status.txt` every 60s; my future polls just `cat` that file (no long SSH sessions). The watchdog hard-caps spend.

---

## Phase F — Wait for completion, push v1 adapter, re-evaluate

Goal: once `~/run_done` appears with `COMPLETE` or `CRASH`, decide what to do; on success, push the final-step LoRA to HF as a NEW model id (don't overwrite v0), then re-run the existing self-terminating eval pointed at v1.

### Task F1: Poll for completion (short SSH, robust to disconnects)

```bash
ssh ... 'cat ~/run_done 2>/dev/null && echo "--- last status ---" && cat ~/run_status.txt'
```

Run this whenever you check in. Expected outcomes:
- `COMPLETE step=499 peak=<MiB> ckpts=20` — proceed to F2.
- `CRASH step=<N> peak=<MiB> ckpts=<K>` — investigate; the latest checkpoint at `outputs/run_default/checkpoints/step_<lastN*25>/` is recoverable.
- (no file yet) — still running; wait.

### Task F2: Push the v1 adapter to HF

Use a fresh model id (`vtemian/dockermin-qwen7b-lora-v1` is reserved for v0; v1 retrain becomes `-v2`).

```bash
ssh ... 'cd ~/prime-rl
LATEST=$(ls -d outputs/run_default/broadcasts/step_* 2>/dev/null | sed "s/.*step_//" | sort -n | tail -1)
echo "latest broadcast step: $LATEST"
[ -f outputs/run_default/broadcasts/step_$LATEST/adapter_model.safetensors ] && echo "adapter file present"
.venv/bin/python ~/dockermin/scripts/push_adapter.py outputs/run_default/broadcasts/step_$LATEST 2>&1 | tail -3'
```

**IMPORTANT:** edit `scripts/push_adapter.py` LOCALLY first to push to `vtemian/dockermin-qwen7b-lora-v2` (or take the repo id as a CLI arg). Commit + pull on the pod before running.

### Task F3: Terminate the training pod

```bash
prime pods terminate <POD_ID> --yes
sleep 4
prime pods list --plain | grep -E "Total|grpo-v1"
```
Expected: `Total: 0`.

Update `docs/cost_log.md` row with the end timestamp, hours, actual cost, and cumulative.

### Task F4: Re-run the eval against v2

The `scripts/eval_unattended.sh` runs the proven 6-baseline eval. Point it at v2:

**Step 1: Update `dockermin_model` arg in the eval script**

Edit `scripts/eval_unattended.sh` (or pass via CLI): change the `--dockermin-model` to `vtemian/dockermin-qwen7b-lora-v2`. Commit on a branch + merge to main.

**Step 2: Rent eval pod + run** (same recipe as `docs/journal.md` 2026-05-27 unattended-eval entries). A new L40S 48GB pod at ~$0.82/hr; full eval ~3-4h; ~$3-5.

**Step 3: Pull the leaderboard from HF** (after the eval pod self-terminates).

---

## Phase G — Decision tree

Read both leaderboards (v0 and v2) and apply the pre-committed kill criteria.

### Criterion 1: Did v1 beat base Qwen meaningfully?

Compare v0-vs-qwen vs v2-vs-qwen on the SAME 37-triple holdout:

- v0: dockermin 56.8% pass / 32.6% reduction vs qwen 56.8% / 32.1% — **flat**.
- v2: dockermin <X>% pass / <Y>% reduction vs qwen 56.8% / 32.1%.

If `(Y - 32.1) >= 5` percentage points AND `X > 56.8`, the GRPO setup IS working — proceed to Criterion 2.

If not — **STOP**: GRPO with this dataset + reward shape genuinely doesn't learn. Ship NEGATIVE_RESULT.md per pre-commitment.

### Criterion 2: Does v1 clear the kill criterion vs Sonnet?

Per `project_dockermin_constraints.md` kill criterion #4: "reduction is strictly better than Sonnet 4.6 zero-shot on at least 30% of holdouts".

Compute from `results.jsonl` v2: count of triples where `dockermin_reduction > sonnet_zs_reduction AND both test_passes`. Divide by 37.

- ≥30% — **shipping branch**: blog post, leaderboard, model card, X thread. The agent-loop comparison becomes optional (we already cleared the bar on zero-shot Sonnet).
- <30% — **NEGATIVE_RESULT.md branch**: include both v0 and v2 leaderboards as evidence the GRPO experiment was honestly tested at two budgets. The agent-loop comparison would not change the verdict (Sonnet zero-shot already wins).

### Either way

Push a v0+v2 leaderboard diff to HF (`vtemian/dockermin-qwen7b-lora-v2/eval/leaderboard-vs-v0.md`) so the contribution is reproducible regardless of outcome.

---

## Cost ledger (target vs cap)

| Phase | Action | Pod $ | API $ | Time |
|---|---|---:|---:|---:|
| A-C | Off-pod code/config/audit | 0 | 0 | ~1h |
| D | Rent + install | 0 | 0 | ~20m |
| E | Training launch + 10-step verify | $0.5 | 0 | ~30m |
| (training) | 500 steps × ~130s + overhead | **$35-50** | 0 | ~14-18h |
| F | Push adapter + terminate | <$1 | 0 | ~5m |
| F | Eval pod (L40S) + re-eval | $3-5 | $1 | ~3-4h |
| Total target | | **~$40-55** | ~$1 | ~18-23h |
| **Hard cap** | 16h training watchdog | $58 | — | — |

Budget at v0 close was $48.53 in wallet → must verify ≥$50 before D2 (top up if needed).

---

## Reference files (do not modify in this plan)

- `docs/decisions/2026-05-22-prime-rl-pin.md` — proven install recipe & memory bounds.
- `docs/runbook_pilot.md` — single-GPU colocation pattern (still applies).
- `docs/journal.md` 2026-05-26/27 entries — v0 lessons, eval-pod patterns, the orphaned-trainer SIGKILL gotcha.
- `project_dockermin_constraints.md` (memory) — kill criteria, pre-commitments, what NOT to do.
- Subagent reports from this plan session — captured inline in the Context section.

---

## Sub-skill required for execution

When executing this plan, use the **superpowers:executing-plans** skill: execute in batches with review checkpoints between phases. Each phase ends with a clear go/no-go (especially E3 — if the reward fix isn't biting, do NOT proceed to the multi-hour training).
