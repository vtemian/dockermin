# Gemma 4 Zero-Shot Probe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run `google/gemma-4-12B-it` zero-shot on the dockermin 37-row holdout, push results to HF, and decide via fixed criteria whether to spend $50+ on a v4-Gemma GRPO run.

**Architecture:** Add a new `baseline_gemma_zero_shot` to `src/dockermin/eval/baselines.py` mirroring the existing `baseline_qwen_zero_shot` pattern (transformers, bf16, `apply_chat_template`). The eval runs on a Datacrunch A100 80GB pod with a fresh venv pinned to `transformers>=5.10.2` because Gemma 4's `model_type=gemma4_unified` is not recognized by our project's `transformers==4.46.*` pin. Results push to `vtemian/dockermin-gemma4-12b-it-probe` on a 60s pusher loop. Decision criteria are encoded up front (Section "Decision Tree" below) so the result mechanically dictates the next move.

**Tech Stack:** transformers ≥5.10.2, torch 2.5+, accelerate, peft (unused but install-side present), huggingface_hub, docker 7.x, tenacity, tqdm. Pod: Datacrunch A100 80GB spot. Output: JSONL + a one-paragraph decision note.

**Wallet impact:** $2–6 happy path on Datacrunch; $10 hard stop. Wallet expected to drop from ~$50 to ~$44–46.

---

## Reference Baselines (the comparison anchors)

| Model | Pass rate (n=37) | Mean reduction\|pass | Source |
|---|---|---|---|
| `qwen_zs` (untrained Qwen 2.5 Coder 7B) | 56.8% | 60.6% | prior controlled re-eval |
| `dockermin v2 step_250` | 56.8% | 85.7% | GRPO-trained Qwen, prior re-eval |
| `dockermin v3 abl_c step_244` | 59.5% | 88.3% | prior re-eval |
| `sonnet_zs` | 91.9% | 59.6% | high-water-mark |
| **`gemma_zs` (this probe)** | **TBD** | **TBD** | — |

## Decision Tree (fix BEFORE running)

Primary signal: pass-rate delta vs `qwen_zs` 56.8%.

```
delta_pass = pass_rate(gemma_zs) - 56.8

if delta_pass >= +5pp:
    if reduction_on_pass(gemma_zs) >= 50%:
        Branch A — write v4-Gemma GRPO plan
    else:
        Branch B — neutral; Gemma is too timid, GRPO can't recover ceiling
elif delta_pass <= -5pp:
    Branch C — kill swap, lock Qwen as base
else:  # within ±5pp
    if reduction_on_pass(gemma_zs) >= 70%:
        Branch A — high aggression ceiling salvages the tie
    else:
        Branch B — neutral; spend budget on tag-RAG or holdout growth
```

Tertiary signal (informational, not decision-changing): failure-mode breakdown
- parse failures dominant → Gemma's instruction-following is weaker; bias toward C
- build failures dominant → world-knowledge gap on tags; GRPO unlikely to fix; B
- test failures dominant → too-aggressive shrink; GOOD failure mode for RL; bias toward A

---

## Tasks

### Task 1: Add `baseline_gemma_zero_shot` (TDD)

**Files:**
- Modify: `src/dockermin/eval/baselines.py:152` (insert new baseline immediately after `baseline_qwen_zero_shot`)
- Test: `tests/test_baselines.py` (existing file — add a registry-presence assertion)

**Step 1: Write the failing test**

Open `tests/test_baselines.py` and find the existing `test_available_baselines_includes_expected` (or equivalent). Add `"gemma_zs"` to the expected set.

```python
# tests/test_baselines.py — add or update the registry assertion
def test_available_baselines_includes_gemma_zs() -> None:
    from dockermin.eval.baselines import available_baselines
    assert "gemma_zs" in available_baselines()
```

If a single existing test asserts the full set with a literal, append `"gemma_zs"` to that set there too — the assertion-failure-message will point you at the file:line.

**Step 2: Run test to verify it fails**

```
pytest tests/test_baselines.py -k gemma -v
```

Expected: FAIL — `KeyError` / `assert "gemma_zs" in ...`.

**Step 3: Implement `_hf_gemma` + `baseline_gemma_zero_shot`**

Insert in `src/dockermin/eval/baselines.py` immediately after the existing `baseline_qwen_zero_shot` block (currently ending around L167). The chat template natively supports the `system` role (verified by reading `google/gemma-4-12B-it/chat_template.jinja` — Gemma 4 routes `messages[0]` with `role=="system"` or `"developer"` as its own turn), so no system→user fold is needed.

```python
# ---------------------------------------------------------------------------
# Baseline 1b: Gemma 4 zero-shot (transformers >= 5.10.2 required)
# ---------------------------------------------------------------------------
# Gemma 4 advertises model_type=gemma4_unified and is rejected by transformers
# 4.46 (our project pin). Run this baseline on a separate pod venv pinned to
# transformers>=5.10.2; see docs/plans/2026-06-08-gemma4-zero-shot-probe.md.
_GEMMA_HANDLE: dict[str, Any] = {}


def _hf_gemma(model_id: str) -> tuple[Any, Any]:
    """Lazy-load a Gemma 4 instruct checkpoint via transformers."""
    if _GEMMA_HANDLE.get("id") == model_id:
        return _GEMMA_HANDLE["model"], _GEMMA_HANDLE["tok"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="bfloat16", device_map="auto"
    )
    _GEMMA_HANDLE["model"] = model
    _GEMMA_HANDLE["tok"] = tok
    _GEMMA_HANDLE["id"] = model_id
    return model, tok


def baseline_gemma_zero_shot(
    triple: dict[str, Any],
    model_id: str = "google/gemma-4-12B-it",
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
) -> EvalEntry:
    """Base Gemma 4 instruct with the standard prompt, no fine-tuning."""
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        model, tok = _hf_gemma(model_id)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True
        )
        text = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        new_df = extract_dockerfile(text)
        if new_df is None:
            return _error_entry("gemma_zs", triple, t0, "no fenced dockerfile block")
        return _entry("gemma_zs", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — eval safety-net: one bad row must not crash the loop
        return _error_entry("gemma_zs", triple, t0, f"gemma error: {e!r}")
```

Then add to `_REGISTRY` at `src/dockermin/eval/baselines.py:656`:

```python
_REGISTRY: dict[str, Callable[..., EvalEntry]] = {
    "qwen_zs": baseline_qwen_zero_shot,
    "gemma_zs": baseline_gemma_zero_shot,   # <-- add this line
    "gpt4o": baseline_gpt4o,
    ...
}
```

**Step 4: Wire `gemma_zs` into `run_eval.py`'s temperature-pass branch**

Modify `scripts/run_eval.py:129`:

```python
# before:
if baseline in {"qwen_zs", "dockermin"}:
    kwargs["temperature"] = args.temperature
    kwargs["max_new_tokens"] = args.max_new_tokens

# after:
if baseline in {"qwen_zs", "gemma_zs", "dockermin"}:
    kwargs["temperature"] = args.temperature
    kwargs["max_new_tokens"] = args.max_new_tokens
```

**Step 5: Re-run the test**

```
pytest tests/test_baselines.py -k gemma -v
```

Expected: PASS.

**Step 6: Run the project quality gate**

```
make quality
```

Expected: PASS. If `mypy` complains about the new `_GEMMA_HANDLE: dict[str, Any]` annotation, match the existing `_BASE_HANDLE` line exactly — they should be identical.

**Step 7: Commit on a feature branch**

```
git checkout -b feat/gemma-zero-shot-baseline
git add src/dockermin/eval/baselines.py scripts/run_eval.py tests/test_baselines.py
git commit -m "feat(eval): add gemma_zs baseline for Gemma 4 12B-it zero-shot"
```

---

### Task 2: Sanity-load Gemma 4 12B locally (skip if no local GPU)

Goal: catch tokenizer / `AutoModelForCausalLM` issues before paying for a pod. Skip this task entirely if you have no local GPU large enough — go straight to Task 3.

**Files:** none. One-off shell.

**Step 1: Fresh probe venv**

```
python3.11 -m venv /tmp/gemma-probe-venv
source /tmp/gemma-probe-venv/bin/activate
pip install -U pip
pip install "transformers>=5.10.2" "accelerate>=1.1" huggingface_hub torch
```

**Step 2: Verify the architecture loads**

```
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('google/gemma-4-12B-it')
print('model_type =', cfg.model_type)
print('transformers_version =', cfg.transformers_version)
"
```

Expected: `model_type = gemma4_unified` and a `5.10` or newer transformers_version. If the script raises `ValueError: Unrecognized model in google/gemma-4-12B-it`, the `transformers>=5.10.2` pin did not take effect — recheck the venv activation.

**Step 3: Verify chat template handles a `system` role**

```
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-12B-it')
out = tok.apply_chat_template(
    [{'role':'system','content':'SYS'}, {'role':'user','content':'USER'}],
    tokenize=False, add_generation_prompt=True
)
print(out)
assert 'SYS' in out, 'system content dropped'
assert 'USER' in out, 'user content dropped'
print('OK')
"
```

Expected: stdout contains both `SYS` and `USER`, and prints `OK`. If `SYS` is missing, file a follow-on issue — the plan's chat-template assumption is wrong and Task 3 must add a system→user fold.

If you skipped Task 2, all this validation moves to Task 5 on the pod.

---

### Task 3: Spin up Datacrunch A100 80GB pod

**Files:**
- New: `scripts/pod_ops/launch_gemma_zs_probe.sh`

**Step 1: Read the existing launcher to crib the pattern**

```
cat scripts/pod_ops/launch_v1_1.sh
```

Note the structure: shebang → `set -euo pipefail` → env-var assertions → `prime pods create` → SSH probe loop → install.sh upload → exec.

**Step 2: Write the launcher**

Create `scripts/pod_ops/launch_gemma_zs_probe.sh`:

```bash
#!/bin/bash
# Launch a Datacrunch A100 80GB pod for the Gemma 4 12B-it zero-shot probe.
#
# Outputs:
#   data/eval/gemma_zs_probe.jsonl (pushed back via the 60s loop)
#   vtemian/dockermin-gemma4-12b-it-probe/eval/results.jsonl on HF
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set (read+write scope on vtemian/* repos)}"
HF_REPO_OUT="vtemian/dockermin-gemma4-12b-it-probe"

# Datacrunch A100 80GB spot — known-good provider for this project.
POD_NAME="gemma-zs-probe-$(date +%s)"
prime pods create \
  --name "$POD_NAME" \
  --gpu-type A100_80GB_PCIE \
  --gpu-count 1 \
  --provider datacrunch \
  --image "ubuntu_22_04_cuda_12_4_open_docker" \
  --disk-size 200

echo "POD launched: $POD_NAME"
echo "Wait for ACTIVE, then run scripts/pod_ops/install_gemma_probe.sh on it."
```

**Step 3: Make executable + verify shellcheck**

```
chmod +x scripts/pod_ops/launch_gemma_zs_probe.sh
shellcheck scripts/pod_ops/launch_gemma_zs_probe.sh   # if installed
```

Expected: no warnings.

**Step 4: Commit**

```
git add scripts/pod_ops/launch_gemma_zs_probe.sh
git commit -m "chore(pod-ops): add gemma_zs probe launcher"
```

---

### Task 4: Write the pod-side install + run script

**Files:**
- New: `scripts/pod_ops/install_gemma_probe.sh`
- New: `scripts/pod_ops/run_gemma_probe.sh`

**Step 1: install_gemma_probe.sh**

```bash
#!/bin/bash
# On-pod installer for the Gemma 4 12B-it probe.
# Idempotent: re-running re-uses the venv but re-installs the editable dockermin.
set -euo pipefail

cd /workspace
[ -d dockermin ] || git clone https://github.com/vtemian/dockermin.git
cd dockermin
git fetch origin
git checkout feat/gemma-zero-shot-baseline   # the branch from Task 1
git pull

# Fresh venv — DO NOT reuse the prime-rl/.venv which is pinned to transformers 4.46.
python3.11 -m venv /workspace/gemma-probe-venv
# shellcheck disable=SC1091
source /workspace/gemma-probe-venv/bin/activate
pip install -U pip wheel

# Bumped transformers + the eval pipeline's runtime deps. --no-deps on the
# editable install prevents transformers from being downgraded back to 4.46.
pip install \
  "transformers>=5.10.2" \
  "torch==2.5.*" \
  "accelerate>=1.1" \
  "huggingface_hub" \
  "docker==7.1.*" \
  "tenacity" \
  "tqdm" \
  "datasets" \
  "openai" \
  "anthropic"
pip install -e . --no-deps

# HF login (token from env)
huggingface-cli login --token "${HF_TOKEN:?}" --add-to-git-credential

# Verify gemma4 model_type is recognized before paying for the weights download.
python -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('google/gemma-4-12B-it')
assert cfg.model_type == 'gemma4_unified', f'unexpected model_type={cfg.model_type}'
print('config OK')
"

echo "install done."
```

**Step 2: run_gemma_probe.sh**

```bash
#!/bin/bash
# Run the Gemma 4 12B-it zero-shot eval on the 37-row holdout.
# Pushes results to HF on a 60s loop so a pod death loses at most 60s of work.
set -uo pipefail
exec > >(tee -a "$HOME/probe.log") 2>&1

POD_ID="${1:?usage: run_gemma_probe.sh <POD_ID>}"
HF_REPO_OUT="vtemian/dockermin-gemma4-12b-it-probe"

cd /workspace/dockermin
# shellcheck disable=SC1091
source /workspace/gemma-probe-venv/bin/activate

terminate() { prime pods terminate "$POD_ID" --yes 2>&1 | head -1 || true; }
trap terminate EXIT
( sleep 14400; echo "[$(date -u)] WATCHDOG 4h"; terminate ) &

# Background pusher: push results.jsonl to HF every 60s.
nohup bash -c '
while true; do
  if [ -f /workspace/dockermin/data/eval/gemma_zs_probe.jsonl ]; then
    python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj=\"/workspace/dockermin/data/eval/gemma_zs_probe.jsonl\",
    path_in_repo=\"eval/results.jsonl\",
    repo_id=\"'"$HF_REPO_OUT"'\",
    repo_type=\"model\",
)
print(\"pushed\")
" || true
  fi
  sleep 60
done
' &

# Pre-warm the docker cache so build-gate results are not artifacts of
# cold-cache pull time. Pull every base FROM tag in the 37-row holdout.
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
print(f"pre-warming {len(bases)} unique base images")
for b in sorted(bases):
    try:
        r = subprocess.run(["docker", "pull", b], capture_output=True, timeout=300)
        print(f"  {'OK' if r.returncode == 0 else 'SKIP'} {b}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {b}")
PYEOF

# Also pull common rewrite targets so cross-pod cache state is consistent
for img in python:3.9-slim python:3.12-slim python:3.12-slim-bookworm \
           eclipse-temurin:25-jdk-resolute eclipse-temurin:25-jdk-noble \
           php:8.5-apache-trixie node:14-slim; do
  docker pull "$img" >/dev/null 2>&1 && echo "OK $img" || echo "SKIP $img"
done

# RUN THE PROBE
mkdir -p data/eval
python scripts/run_eval.py \
  --baselines gemma_zs \
  --holdout vtemian/dockermin-v0 \
  --temperature 0.2 \
  --max-new-tokens 1024 \
  --out data/eval/gemma_zs_probe.jsonl

# Final push (in case the 60s loop missed the last row)
python - <<'PYEOF'
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj="/workspace/dockermin/data/eval/gemma_zs_probe.jsonl",
    path_in_repo="eval/results.jsonl",
    repo_id="vtemian/dockermin-gemma4-12b-it-probe",
    repo_type="model",
)
print("final push OK")
PYEOF

echo "[$(date -u)] === GEMMA_ZS PROBE DONE — terminating pod ==="
```

**Step 3: Mark both executable + commit**

```
chmod +x scripts/pod_ops/install_gemma_probe.sh scripts/pod_ops/run_gemma_probe.sh
git add scripts/pod_ops/install_gemma_probe.sh scripts/pod_ops/run_gemma_probe.sh
git commit -m "chore(pod-ops): add gemma_zs probe install + run scripts"
```

---

### Task 5: Create the HF result repo

**Files:** none (HF-side operation).

**Step 1: Create the repo via HF CLI**

```
huggingface-cli repo create dockermin-gemma4-12b-it-probe --type model
```

Expected: `Your repo now lives at https://huggingface.co/vtemian/dockermin-gemma4-12b-it-probe`.

**Step 2: Smoke an empty upload to verify token scope**

```
echo '{"baseline":"smoke","triple_id":"smoke","new_size_bytes":null,"test_passes":false,"elapsed_s":0.0,"error":"smoke"}' \
  | huggingface-cli upload vtemian/dockermin-gemma4-12b-it-probe /dev/stdin eval/.smoke.jsonl
```

If this 401s, regenerate the HF token with write scope before launching the pod. If it succeeds, delete the smoke file from the HF web UI; it doesn't matter much, the pusher loop will overwrite `eval/results.jsonl`.

---

### Task 6: Push the feature branch + launch the pod

**Files:** none.

**Step 1: Push the branch**

```
git push -u origin feat/gemma-zero-shot-baseline
```

The on-pod installer (Task 4) clones from `origin` and checks out this branch — it must be pushed for the pod to find it.

**Step 2: Launch**

```
export HF_TOKEN=hf_...
bash scripts/pod_ops/launch_gemma_zs_probe.sh
```

Expected: stdout reports `POD launched: gemma-zs-probe-<timestamp>` and a pod id. Wait for ACTIVE — typically 2–4 min on Datacrunch.

**Step 3: SSH in + run install**

```
prime pods list | grep gemma-zs-probe
# copy the ssh command, run it
ssh root@<ip> -p 22
# on the pod:
export HF_TOKEN=hf_...
curl -fsSL https://raw.githubusercontent.com/vtemian/dockermin/feat/gemma-zero-shot-baseline/scripts/pod_ops/install_gemma_probe.sh | bash
```

Expected: install completes with `config OK` on the last line.

**Step 4: Kick off the probe in the background**

```
# Still on the pod:
nohup bash /workspace/dockermin/scripts/pod_ops/run_gemma_probe.sh <POD_ID> > /workspace/probe.out 2>&1 &
disown
exit
```

The watchdog inside `run_gemma_probe.sh` will terminate the pod after the eval completes or after 4 hours, whichever comes first.

---

### Task 7: Monitor + collect

**Files:** none.

**Step 1: Poll HF for partial results every few minutes**

```
huggingface-cli download vtemian/dockermin-gemma4-12b-it-probe \
  eval/results.jsonl --local-dir /tmp/gemma_probe
wc -l /tmp/gemma_probe/eval/results.jsonl
```

Expected: row count climbs from 0 toward 37 over ~40 min. If it stalls at 0 for >10 min after the pod went ACTIVE, SSH in and read `/workspace/probe.out`.

**Step 2: Verify the pod self-terminated**

```
prime pods list | grep gemma-zs-probe
```

Expected: empty (the pod is gone). If it's still ACTIVE after `wc -l` shows 37 rows, terminate manually: `prime pods terminate <POD_ID> --yes`.

**Step 3: Pull the final JSONL into the repo**

```
mkdir -p data/eval
cp /tmp/gemma_probe/eval/results.jsonl data/eval/gemma_zs_probe.jsonl
```

Sanity check the row count and the schema:

```
wc -l data/eval/gemma_zs_probe.jsonl   # expect 37
head -1 data/eval/gemma_zs_probe.jsonl | python -m json.tool
```

Expected: 37 rows, each with `baseline: "gemma_zs"`, `triple_id`, `new_size_bytes`, `test_passes`, etc.

---

### Task 8: Render comparison + apply the decision tree

**Files:**
- New: `docs/decisions/2026-06-08-gemma-zs-probe-result.md`
- Modify: `docs/leaderboard.md` (regenerated by `scripts/leaderboard.py`)

**Step 1: Compute the headline numbers**

```
python - <<'PYEOF'
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("data/eval/gemma_zs_probe.jsonl").read_text().splitlines()]
n = len(rows)
passes = sum(1 for r in rows if r.get("test_passes"))
pass_rate = passes / n * 100
sizes = [(r["original_size"], r["new_size_bytes"]) for r in rows
         if r.get("test_passes") and r.get("new_size_bytes")]
red = [(o - n) / o * 100 for (o, n) in sizes] if sizes else []
mean_red = sum(red) / len(red) if red else 0.0
parse_fail = sum(1 for r in rows if "no fenced dockerfile block" in (r.get("error") or ""))
build_fail = sum(1 for r in rows if r.get("error") and "build" in r["error"].lower())
test_fail = passes < n and (n - passes - parse_fail - build_fail)
print(f"n={n}")
print(f"pass_rate={pass_rate:.1f}%")
print(f"mean_reduction|pass={mean_red:.1f}%")
print(f"parse_fail={parse_fail}  build_fail={build_fail}  test_fail≈{test_fail}")
PYEOF
```

Expected: a printout with the four numbers needed for the decision tree.

**Step 2: Update the leaderboard**

```
make leaderboard
```

Expected: `docs/leaderboard.md` regenerates with a `gemma_zs` row.

**Step 3: Apply the decision tree**

Look up the printed `pass_rate` and `mean_reduction|pass` in the Decision Tree table at the top of this plan. Identify Branch A/B/C.

**Step 4: Write the decision note**

Create `docs/decisions/2026-06-08-gemma-zs-probe-result.md`:

```markdown
# Gemma 4 12B-it Zero-Shot Probe — Result and Next Move

**Date:** 2026-06-08
**Cost:** $<actual> (Datacrunch A100 80GB, ~<minutes> min wall)
**Probe results JSONL:** vtemian/dockermin-gemma4-12b-it-probe/eval/results.jsonl

## Numbers (n=37 holdout, T=0.2, max_new_tokens=1024)

| Model | Pass rate | Mean reduction\|pass |
|---|---|---|
| qwen_zs | 56.8% | 60.6% |
| dockermin v2 step_250 | 56.8% | 85.7% |
| **gemma_zs** | **<X>%** | **<Y>%** |

## Failure-mode breakdown
- parse_fail: <count>
- build_fail: <count>
- test_fail: <count>

## Branch fired
**Branch <A|B|C>** — <one-line rationale per the decision tree>.

## Next move
- Branch A: draft `docs/plans/2026-06-09-v4-gemma-grpo.md`.
- Branch B: spike tag-RAG over Docker Hub manifest (separate plan).
- Branch C: lock Qwen 2.5 Coder 7B Instruct as the base; write up NEGATIVE_RESULT.md addendum noting Gemma 4 was tested zero-shot and rejected.
```

Fill in the `<X>`, `<Y>`, branch letter, and rationale. **Do not** start the follow-on work in this PR.

**Step 5: Commit + open PR**

```
git add docs/decisions/2026-06-08-gemma-zs-probe-result.md docs/leaderboard.md \
        data/eval/gemma_zs_probe.jsonl
git commit -m "docs(probe): gemma_zs result + branch <A|B|C> decision"
git push
gh pr create --fill --base main
```

---

## Verification Checklist (before declaring done)

- [ ] `make quality` passes locally on the feature branch.
- [ ] `tests/test_baselines.py` includes the `gemma_zs` registry assertion.
- [ ] `data/eval/gemma_zs_probe.jsonl` has exactly 37 rows.
- [ ] HF repo `vtemian/dockermin-gemma4-12b-it-probe` has `eval/results.jsonl` matching the local file.
- [ ] `docs/leaderboard.md` includes a `gemma_zs` row.
- [ ] `docs/decisions/2026-06-08-gemma-zs-probe-result.md` exists with branch decision filled in.
- [ ] The pod has terminated (`prime pods list` empty).
- [ ] Total wallet spend ≤ $10.

---

## Failure modes + responses

| Failure | Response |
|---|---|
| `transformers>=5.10.2` install pulls in incompatible torch | Pin `torch==2.5.*` explicitly (already in install script). If still broken, fall back to `torch==2.4.*`. |
| OOM loading Gemma 4 12B on A100 80GB | Unexpected (~24 GiB weights). Fall back to `google/gemma-4-E4B-it` by setting `--dockermin-model google/gemma-4-E4B-it` — wait, the env-var threads through `baseline_gemma_zero_shot`'s `model_id` default, not a CLI flag. Patch `run_eval.py` to forward `--gemma-model` if you actually hit this. |
| Docker registry mirror returns 5xx during pre-warm | The pre-warm tolerates per-image failures (`SKIP` markers). Eval continues; affected rows will fail at build-gate and count as build_fail. |
| Pod dies mid-eval before all 37 rows are written | The 60s pusher loop has the partial JSONL on HF. Resume by launching a second pod, running install, then re-running ONLY the missing rows (manual filter on `triple_id`). Cost: ~$3 more. |
| Gemma chat template silently drops system role (Task 2 caught this is FALSE for Gemma 4, but verify on first row) | Patch `format_messages` in `src/dockermin/reward/prompts.py` to fold `SYSTEM_PROMPT` into the user content. Re-push branch, restart pod. Cost: ~$3 more. |

---

## Out of Scope (do not implement now)

- Running Gemma 4 E4B (skip unless 12B fails — see decision in agent reports).
- Adding a `baseline_gemma_zero_shot` to the `quality` test fixture set in a way that requires `transformers>=5.10.2` in the project's venv. The new baseline lives in the registry; the test only asserts presence in `available_baselines()`. Mypy / ruff don't import transformers' Gemma 4 paths.
- A full v4-Gemma GRPO training plan. That's Branch A's follow-on, a SEPARATE plan.
- Any tag-RAG implementation. That's Branch B's follow-on.

---

## Notes for the executor

- The default `model_id` in `baseline_gemma_zero_shot` is hardcoded to `google/gemma-4-12B-it`. Do NOT parameterize via env var in this plan — that's premature flexibility. If we end up with multiple Gemma variants, plumb a `--gemma-model` arg through `run_eval.py` then.
- The existing `baseline_qwen_zero_shot` is the model to mirror. Read it carefully (`src/dockermin/eval/baselines.py:130-167`) before writing the Gemma version — match its style exactly, including the safety-net comment and the `_entry`/`_error_entry` calls.
- The probe MUST run with the same `temperature=0.2, max_new_tokens=1024` as the prior qwen_zs and dockermin-v2 evals. Any drift makes the comparison meaningless.
- The decision tree in this plan is binding. Do not retroactively soften the thresholds after seeing the numbers.
