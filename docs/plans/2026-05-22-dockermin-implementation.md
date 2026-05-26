# Dockermin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a GRPO-fine-tuned LoRA on Qwen 2.5 Coder 7B Instruct that rewrites working Dockerfiles to be smaller while still passing functional tests. Deliverables: LoRA adapter on HF, dataset on HF, GitHub repo with CLI, benchmark + leaderboard vs 7 baselines, blog writeup OR `NEGATIVE_RESULT.md`.

**Architecture:** prime-rl (TOML config layer over `verifiers`) runs on Prime Intellect on-demand H100 pods. vLLM serves rollouts with dynamic LoRA load/unload (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`). Reward is a CURE-style composite: hard gates (parse, build, test) then dense size-reduction signal then shaping bonuses gated on test pass. Eval runs against 7 baselines including Claude Sonnet 4.6 in a Claude Code agent loop. Docker rollouts run on the same pod via DooD (mounted host docker socket); BuildKit `docker-container` driver with local cache export amortizes apt/pip/npm across 16 concurrent builds.

**Tech Stack:**
- Python 3.11
- `prime-rl` (pin to commit after PR #1392 lands, or `main` with non-NCCL broadcast workaround)
- `verifiers` (transitive via prime-rl; expect >=0.1.5)
- `vllm` 0.7.3 (LoRA V1, runtime adapter load/unload)
- `peft` 0.13, `transformers` 4.46, `torch` 2.5
- `docker` 7.1 SDK (containers.run/remove/prune only) + `docker buildx` subprocess (builds)
- `dockerfile` 3.3.1 (parse, deprecated but works)
- Prime Intellect on-demand pods (1xH100 80GB dev, 8xH100 80GB full run)
- HF Hub for dataset + adapter, wandb for telemetry

**Time budget:** 3 weekends. Hard kill criteria documented in `project_dockermin_constraints.md` memory.
**Cost cap:** $400 (triggers $200 pause, $400 cap, $500 stop).

---

## Plan-wide conventions

- **TDD where applicable.** Reward function, dataset annotators, prompt parser: tests first. Infra setup (renting GPUs, installing CLIs, docker config): imperative checklist.
- **Commit per task.** Frequent commits with imperative subject. Use `git push` at end of each phase.
- **Verification before completion.** Each task ends with an explicit checkpoint command + expected output. Do not mark a task done if the checkpoint fails.
- **Reference skills:** @superpowers:test-driven-development for reward function, @superpowers:systematic-debugging if any pilot run fails, @superpowers:verification-before-completion before claiming any milestone done.
- **Code style:** ruff defaults (line-length 100), no emdash in code/prose, docstrings only when WHY is non-obvious.

---

# Phase 0: Local prep (already partially done Friday evening)

## Task 0.1: Fix `pyproject.toml` dep pins per research findings

**Why:** verifiers==0.1.4 is the wrong pin (no LoRA hotswap path, no `@reward` decorator). prime-rl is the actual entrypoint.

**Files:**
- Modify: `pyproject.toml`

**Step 1: Edit `pyproject.toml`**

Replace the `dependencies` block with:
```toml
dependencies = [
  # prime-rl pulls verifiers, vllm, transformers, peft transitively.
  # Pinning prime-rl explicitly + leaving verifiers floating to whatever prime-rl needs.
  "prime-rl",                # exact commit pin set during Task 1.3 once we confirm PR #1392 status
  "verifiers>=0.1.5",        # 0.1.4 lacks LoRA hotswap glue; prime-rl needs newer
  "vllm==0.7.3",
  "transformers==4.46.*",
  "peft==0.13.*",
  "torch==2.5.*",
  "accelerate==1.1.*",
  "datasets==3.1.*",
  "dockerfile==3.3.1",       # deprecated upstream but works
  "docker==7.1.*",
  "huggingface_hub",
  "wandb",
  "anthropic",               # synthetic variant generation + Sonnet baseline
  "openai",                  # GPT-4o baseline
  "tqdm",
  "tenacity",                # retry on Docker daemon flakes
]
```

**Step 2: Commit**
```bash
git add pyproject.toml
git commit -m "pyproject: switch to prime-rl entrypoint, drop hard verifiers 0.1.4 pin"
git push
```

**Checkpoint:** `grep "prime-rl" pyproject.toml` returns the dep line.

---

## Task 0.2: Add `src/dockermin/` package skeleton

**Why:** All implementation code lives in `src/dockermin/`. Create the directories now so file paths in later tasks are unambiguous.

**Files (create empty placeholders, one line `"""Module docstring."""` each):**
- Create: `src/dockermin/dataset/__init__.py`
- Create: `src/dockermin/dataset/annotate.py`
- Create: `src/dockermin/dataset/scrape.py`
- Create: `src/dockermin/reward/__init__.py`
- Create: `src/dockermin/reward/dockermin_reward.py`
- Create: `src/dockermin/reward/gates.py`
- Create: `src/dockermin/reward/prompts.py`
- Create: `src/dockermin/eval/__init__.py`
- Create: `src/dockermin/eval/baselines.py`
- Create: `src/dockermin/eval/agent_loop.py`
- Create: `src/dockermin/cli.py` (already exists, leave)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_annotate.py`
- Create: `tests/test_gates.py`
- Create: `tests/test_reward.py`
- Create: `tests/test_prompts.py`

**Step 1: Create each file with one-line docstring matching its purpose.**

**Step 2: Commit**
```bash
git add src/ tests/
git commit -m "scaffold: package layout for dataset, reward, eval, cli"
git push
```

**Checkpoint:** `find src tests -name '*.py' | wc -l` returns 18 (16 new + 2 pre-existing: `src/dockermin/__init__.py`, `src/dockermin/cli.py`).

---

# Phase 1: Weekend 1 Saturday - stack smoke test

## Task 1.1: Prime Intellect account, CLI install, auth

**Why:** Cannot rent a pod without it.

**Files:** none.

**Steps (Vlad action - all on local machine):**
1. Create account at https://app.primeintellect.ai
2. Generate API key at https://app.primeintellect.ai/dashboard/tokens
3. Install CLI: `uv tool install -U prime` (or `pip install prime` into the venv)
4. `prime login` (opens browser) or `prime config set-api-key <KEY>`
5. Copy key into `dockermin/.env` as `PRIME_INTELLECT_API_KEY=...`
6. Verify: `prime availability --gpu-type H100_80GB --gpu-count 1` returns provider list with prices.

**Checkpoint:** `prime availability --gpu-type H100_80GB --gpu-count 1 | head` shows at least one provider with an hourly rate.

**Time estimate:** 10 min.

**No commit (no code).**

---

## Task 1.2: Rent 1xH100 80GB on-demand pod

**Why:** Need a GPU to run the smoke tests. Picking the cheapest available provider keeps the smoke-test phase under $10.

**Files:**
- Create: `docs/cost_log.md` already exists, append row.

**Steps:**
1. From `prime availability` output, pick cheapest provider id.
2. `prime pods create --gpu-type H100_80GB --gpu-count 1 --provider <id> --image <image_id>` (default image is fine; needs CUDA, will configure inside).
3. `prime pods list` to get pod id.
4. `prime pods status <pod_id>` until status is `running`, copy `sshConnection`.
5. Append to `docs/cost_log.md`: row with date, "1.2 rent dev pod", provider, "1xH100", start timestamp, end blank, hours blank, $/hr, cost blank, cumulative blank.
6. `ssh root@<ip> -p <port>` to verify access.

**Checkpoint:** `ssh root@<ip> -p <port> 'nvidia-smi | head -20'` returns one H100 entry.

**Hard exit:** if `prime pods status` is stuck >5 min in `provisioning`, switch provider and retry. Do not pay for a queued pod.

**Time estimate:** 15 min, ~$0.40 spend by checkpoint.

**No commit (cost log updated by hand or at end of phase).**

---

## Task 1.3: prime-rl install on pod + PR #1392 status check

**Why:** prime-rl IS the entrypoint. Need to know if PR #1392 (LoRA + NCCL `adapter_only` bug) is merged before we commit to it.

**Files:**
- Create: `docs/decisions/2026-05-22-prime-rl-pin.md`

**Steps on pod:**
1. `git clone https://github.com/PrimeIntellect-ai/prime-rl.git && cd prime-rl`
2. `gh pr view 1392 --json mergedAt,state` from a host with `gh` (local laptop). Capture the result.
3. If merged: `git checkout main && git log -1 --format="%H %s"` (or pin to `git rev-parse HEAD`). Record this commit in `docs/decisions/2026-05-22-prime-rl-pin.md` (back on the local repo).
4. If NOT merged: `git log --all --grep "adapter_only\|nccl\|1392" --oneline | head` to find any related fix. Worst case, pin to a commit before LoRA auto-config was introduced and pass explicit non-NCCL broadcast args. Record the decision.
5. On pod: `curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/<commit>/scripts/install.sh | bash` (or `uv sync --all-extras` from the cloned tree).
6. Verify: `uv run rl --help` lists subcommands including model/lora args.

**Checkpoint:** `uv run rl --help 2>&1 | grep -i lora` shows `--model.experimental.lora` and `--model.max_lora_rank` flags.

**Step 7: Local commit (back on laptop)**
```bash
git add docs/decisions/2026-05-22-prime-rl-pin.md
git commit -m "decision: pin prime-rl to <commit> after PR #1392 status check"
git push
```

**Time estimate:** 30 min including any install hiccups.

---

## Task 1.4: Run prime-rl `alphabet_sort` LoRA smoke test (~1h on 1xH100)

**Why:** Cheap end-to-end proof that prime-rl + verifiers + vLLM + LoRA all wire together on this pod. Reward should trend up across 100 steps. If this does not work, the framework is broken on this pod and the weekend is at risk.

**Files:** none in repo, all on pod.

**Steps on pod:**
1. `prime env install primeintellect/alphabet-sort`
2. `wandb login` with key from `.env`
3. `bash scripts/tmux.sh` (prime-rl launcher script that brings up orchestrator + inference + trainer panes)
4. In a tmux pane, run: `uv run rl @ examples/alphabet_sort/rl.toml --model.experimental.lora --model.max_lora_rank 32 --wandb.project dockermin --wandb.name alphabet-sort-smoke`
5. Watch the wandb panel. Expected: reward starts near 0, climbs above 0.3 within 30-50 steps.
6. Let it run for 100 steps (~1h on 1xH100).

**Checkpoint:** wandb shows reward > 0.5 at step 100. Loss is finite. No CUDA OOM. vLLM server stayed alive.

**Hard exit:** if reward is flat or NaN at step 30, kill the run. Do not let it burn an hour. Open Prime Intellect Discord, file the issue with logs.

**Time estimate:** 70 min wall-clock (~$1.50 spend).

**No commit (smoke test, no code changes).** Update `docs/cost_log.md` at end of session.

---

## Task 1.5: LoRA hotswap correctness smoke test

**Why:** Highest-risk-not-yet-verified piece. If vLLM 0.7.3 + Qwen 2.5 Coder 7B + LoRA rank 32 hotswap is broken, we lose async rollout overlap. Need to know BEFORE we build the dataset.

**Files:**
- Create: `scripts/smoke_lora_hotswap.py` (on pod, also commit to repo)
- Create: `scripts/__init__.py` (empty)

**Step 1: Write the smoke test script**

```python
# scripts/smoke_lora_hotswap.py
"""Verify vLLM 0.7.3 LoRA hotswap on Qwen 2.5 Coder 7B before committing to GRPO pipeline."""
from __future__ import annotations
import os, sys, tempfile, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
PROMPT = "Write a Python function that returns the nth Fibonacci number."

def train_tiny_lora(seed: int, out_dir: Path) -> None:
    """Train a 50-step LoRA on a single random batch so adapters differ."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    text = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n" * 4
    enc = tok(text, return_tensors="pt").to("cuda")
    for _ in range(50):
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward(); opt.step(); opt.zero_grad()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    del model; torch.cuda.empty_cache()

def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="lora_smoke_"))
    a_dir, b_dir = workdir / "a", workdir / "b"
    print(f"Training LoRA A in {a_dir}")
    train_tiny_lora(seed=1, out_dir=a_dir)
    print(f"Training LoRA B in {b_dir}")
    train_tiny_lora(seed=2, out_dir=b_dir)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=BASE, enable_lora=True, max_loras=4, max_lora_rank=32,
              dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=128, seed=0)

    t0 = time.perf_counter()
    base_out = llm.generate([PROMPT], sp)[0].outputs[0].text
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    a1 = llm.generate([PROMPT], sp, lora_request=LoRARequest("a", 1, str(a_dir)))[0].outputs[0].text
    t_a1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    b1 = llm.generate([PROMPT], sp, lora_request=LoRARequest("b", 2, str(b_dir)))[0].outputs[0].text
    t_b1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    a2 = llm.generate([PROMPT], sp, lora_request=LoRARequest("a", 1, str(a_dir)))[0].outputs[0].text
    t_a2 = time.perf_counter() - t0

    print(f"latency base={t_base:.3f}s a1={t_a1:.3f}s b1={t_b1:.3f}s a2={t_a2:.3f}s")
    if base == a1:
        print("FAIL: base output == LoRA A output (adapter not applied)"); return 1
    if a1 == b1:
        print("FAIL: LoRA A output == LoRA B output (swap not effective)"); return 1
    if a1 != a2:
        print("FAIL: same LoRA + greedy + same seed produced different output (nondeterminism)"); return 1
    print("PASS: base != A != B, A reproducible after swap")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: rsync the script to the pod and run**
```bash
# local laptop
rsync -av scripts/ root@<pod_ip>:/root/dockermin-scripts/

# on pod
cd /root/dockermin-scripts
python smoke_lora_hotswap.py
```

**Checkpoint:** Script prints `PASS: base != A != B, A reproducible after swap`. Latencies for a2 (warm swap) should be lower than b1 (first load of B).

**Step 3: Commit script back to repo (laptop)**
```bash
git add scripts/smoke_lora_hotswap.py scripts/__init__.py
git commit -m "scripts: vLLM 0.7.3 LoRA hotswap correctness smoke test"
git push
```

**Hard exit:** if any FAIL print appears, the LoRA path is broken. File issue against vllm 0.7.3 with the repro, then decide: either fall back to verifiers without hotswap (slower, more expensive), or wait for fix. Do NOT proceed to Phase 2 until this passes.

**Time estimate:** 25 min (loads model twice, trains 2 tiny LoRAs, runs 4 inferences).

---

## Task 1.6: Terminate pod, log cost

**Why:** Idle pod keeps billing. Hygiene that protects the budget.

**Steps:**
1. `prime pods terminate <pod_id>` from laptop.
2. `prime pods list` confirms status `terminated`.
3. Update `docs/cost_log.md`: fill end timestamp, hours, cost, cumulative.
4. Update `docs/journal.md`: 3-5 sentences on what worked, what surprised, what we now know.

**Step 4: Commit**
```bash
git add docs/cost_log.md docs/journal.md
git commit -m "weekend1 sat: stack smoke + LoRA hotswap passed, log cost"
git push
```

**Checkpoint:** `prime pods list | grep <pod_id>` shows `terminated` and `docs/cost_log.md` cumulative is filled in.

---

# Phase 2: Weekend 1 Sunday - dataset curation

## Task 2.1: TDD - annotate() parse-gate behaves

**Why:** Annotate is the foundation. Parse-gate is the cheapest gate to test (no docker required). TDD it first.

**Files:**
- Modify: `tests/test_annotate.py`
- Modify: `src/dockermin/dataset/annotate.py`

**Step 1: Write the failing test**
```python
# tests/test_annotate.py
"""Tests for the per-Dockerfile annotate() pipeline."""
import pytest
from dockermin.dataset.annotate import parse_gate, ParseResult

def test_parse_gate_accepts_valid_dockerfile():
    df = "FROM python:3.12-slim\nRUN pip install flask\nCMD [\"python\",\"-m\",\"flask\",\"run\"]"
    result = parse_gate(df)
    assert isinstance(result, ParseResult)
    assert result.ok is True
    assert result.command_count == 3

def test_parse_gate_rejects_garbage():
    result = parse_gate("this is not a Dockerfile")
    assert result.ok is False
    assert "parse" in result.error.lower()

def test_parse_gate_rejects_too_short():
    result = parse_gate("FROM scratch")
    assert result.ok is False
    assert "too short" in result.error.lower() or "minimum" in result.error.lower()
```

**Step 2: Run, expect failure**
```bash
.venv/bin/pytest tests/test_annotate.py -v
```
Expected: 3 failures with `ImportError: cannot import name 'parse_gate'`.

**Step 3: Minimal implementation**
```python
# src/dockermin/dataset/annotate.py
"""Per-Dockerfile annotation pipeline: parse, build, test."""
from __future__ import annotations
from dataclasses import dataclass
import dockerfile

MIN_COMMANDS = 2

@dataclass(frozen=True)
class ParseResult:
    ok: bool
    command_count: int = 0
    error: str = ""

def parse_gate(df_text: str) -> ParseResult:
    """Validate Dockerfile parses and has at least MIN_COMMANDS instructions."""
    try:
        cmds = dockerfile.parse_string(df_text)
    except dockerfile.GoParseError as e:
        return ParseResult(ok=False, error=f"parse error: {e}")
    if len(cmds) < MIN_COMMANDS:
        return ParseResult(ok=False, command_count=len(cmds),
                           error=f"too short: {len(cmds)} < minimum {MIN_COMMANDS}")
    return ParseResult(ok=True, command_count=len(cmds))
```

**Step 4: Run, expect pass**
```bash
.venv/bin/pytest tests/test_annotate.py -v
```
Expected: 3 passed.

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "annotate: parse_gate with min instruction count"
git push
```

---

## Task 2.2: TDD - annotate() build_gate via docker SDK

**Why:** Build gate proves the Dockerfile produces an image. Limit to 300s timeout, capture size.

**Files:**
- Modify: `tests/test_annotate.py`
- Modify: `src/dockermin/dataset/annotate.py`

**Step 1: Write the failing test (uses real docker daemon - skip if not available)**
```python
# tests/test_annotate.py - APPEND

import os, pytest, docker
from dockermin.dataset.annotate import build_gate, BuildResult

DOCKER_AVAILABLE = False
try:
    docker.from_env().ping()
    DOCKER_AVAILABLE = True
except Exception:
    pass

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_succeeds_on_minimal_alpine():
    df = "FROM alpine:3.20\nRUN echo hello > /msg\nCMD [\"cat\",\"/msg\"]\n"
    result = build_gate(df, timeout_s=120)
    assert isinstance(result, BuildResult)
    assert result.ok is True
    assert result.size_bytes > 0
    assert result.size_bytes < 20_000_000  # alpine + tiny file should be <20MB
    assert result.tag.startswith("dockermin/curate:")

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_fails_on_broken_command():
    df = "FROM alpine:3.20\nRUN exit 1\n"
    result = build_gate(df, timeout_s=60)
    assert result.ok is False
    assert "build" in result.error.lower() or "exit" in result.error.lower()
```

**Step 2: Run, expect failure (ImportError).**

**Step 3: Implementation**
```python
# src/dockermin/dataset/annotate.py - APPEND
import hashlib, io, time
import docker
from docker.errors import BuildError, APIError

@dataclass(frozen=True)
class BuildResult:
    ok: bool
    tag: str = ""
    size_bytes: int = 0
    build_seconds: float = 0.0
    error: str = ""

def _docker_client():
    return docker.from_env(timeout=600)

def build_gate(df_text: str, timeout_s: int = 300) -> BuildResult:
    """Build the Dockerfile and return image size. Uses the classic builder via SDK (no BuildKit)."""
    client = _docker_client()
    digest = hashlib.sha256(df_text.encode()).hexdigest()[:12]
    tag = f"dockermin/curate:{digest}"
    t0 = time.perf_counter()
    try:
        image, log_stream = client.images.build(
            fileobj=io.BytesIO(df_text.encode()),
            tag=tag, rm=True, forcerm=True, timeout=timeout_s,
        )
        # Drain the log stream so the http connection isn't held.
        for _ in log_stream:
            pass
    except (BuildError, APIError) as e:
        return BuildResult(ok=False, error=f"build error: {e}")
    elapsed = time.perf_counter() - t0
    size = client.images.get(tag).attrs["Size"]
    return BuildResult(ok=True, tag=tag, size_bytes=size, build_seconds=elapsed)
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "annotate: build_gate via docker SDK, returns size"
git push
```

**Note:** The build hot-path in training will use `docker buildx build` subprocess (BuildKit + cache mounts). This `build_gate` for curation can stay on the SDK; curation throughput is not the bottleneck.

---

## Task 2.3: TDD - annotate() test_gate

**Why:** Last gate before a triple is accepted. Runs the test_cmd inside the built image and matches against expected substring.

**Files:**
- Modify: `tests/test_annotate.py`
- Modify: `src/dockermin/dataset/annotate.py`

**Step 1: Write failing test**
```python
# tests/test_annotate.py - APPEND
from dockermin.dataset.annotate import test_gate, TestResult

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_passes_when_substring_present():
    df = "FROM alpine:3.20\nRUN echo readyok > /msg\nCMD [\"cat\",\"/msg\"]\n"
    build = build_gate(df)
    assert build.ok
    result = test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert isinstance(result, TestResult)
    assert result.ok is True
    assert "readyok" in result.output

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_fails_when_substring_absent():
    df = "FROM alpine:3.20\nRUN echo nope > /msg\nCMD [\"cat\",\"/msg\"]\n"
    build = build_gate(df)
    assert build.ok
    result = test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert result.ok is False
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/dataset/annotate.py - APPEND
@dataclass(frozen=True)
class TestResult:
    ok: bool
    output: str = ""
    exit_code: int | None = None
    error: str = ""

def test_gate(tag: str, cmd: list[str], expected_substring: str, timeout_s: int = 30) -> TestResult:
    """Run cmd inside the image, capture combined output, match expected substring."""
    client = _docker_client()
    try:
        container = client.containers.run(
            tag, command=cmd, detach=True,
            network_mode="bridge",
            mem_limit="1g", memswap_limit="1g",
            nano_cpus=2_000_000_000,
            pids_limit=512,
        )
    except APIError as e:
        return TestResult(ok=False, error=f"start error: {e}")
    try:
        status = container.wait(timeout=timeout_s)
        exit_code = status.get("StatusCode")
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        combined = stdout + "\n" + stderr
        if exit_code != 0:
            return TestResult(ok=False, output=combined, exit_code=exit_code,
                              error=f"non-zero exit {exit_code}")
        if expected_substring not in combined:
            return TestResult(ok=False, output=combined, exit_code=exit_code,
                              error="expected substring not found")
        return TestResult(ok=True, output=combined, exit_code=exit_code)
    finally:
        try: container.remove(force=True)
        except Exception: pass
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "annotate: test_gate runs cmd, matches substring with limits"
git push
```

---

## Task 2.4: annotate() composition + a top-level `annotate_one()`

**Files:**
- Modify: `tests/test_annotate.py`
- Modify: `src/dockermin/dataset/annotate.py`

**Step 1: Write failing test**
```python
from dockermin.dataset.annotate import annotate_one, AnnotateResult

@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_annotate_one_happy_path_flask_smoke():
    df = """FROM python:3.12-slim
RUN pip install --no-cache-dir flask==3.0.0
CMD ["python","-c","import flask,sys;print('ok',flask.__version__)"]
"""
    result = annotate_one(df, ["python","-c","import flask,sys;print('ok',flask.__version__)"], "ok 3.0.0")
    assert isinstance(result, AnnotateResult)
    assert result.ok is True
    assert result.baseline_size > 50_000_000
    assert result.baseline_build_s > 0
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/dataset/annotate.py - APPEND
@dataclass(frozen=True)
class AnnotateResult:
    ok: bool
    baseline_size: int = 0
    baseline_build_s: float = 0.0
    tag: str = ""
    error: str = ""

def annotate_one(df_text: str, test_cmd: list[str], expected_substring: str,
                 build_timeout_s: int = 300, test_timeout_s: int = 30) -> AnnotateResult:
    p = parse_gate(df_text)
    if not p.ok: return AnnotateResult(ok=False, error=p.error)
    b = build_gate(df_text, timeout_s=build_timeout_s)
    if not b.ok: return AnnotateResult(ok=False, error=b.error)
    t = test_gate(b.tag, test_cmd, expected_substring, timeout_s=test_timeout_s)
    if not t.ok: return AnnotateResult(ok=False, error=t.error)
    return AnnotateResult(ok=True, baseline_size=b.size_bytes,
                          baseline_build_s=b.build_seconds, tag=b.tag)
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "annotate: annotate_one composes parse + build + test gates"
git push
```

---

## Task 2.5: One-Dockerfile Flask smoke (Vlad runs manually)

**Why:** Real end-to-end before scaling.

**Steps (local, requires Docker Desktop or similar running on laptop):**
1. `cd /Users/whitemonk/projects/ai/dockermin`
2. `.venv/bin/python -c "from dockermin.dataset.annotate import annotate_one; r = annotate_one(open('tests/fixtures/flask.Dockerfile').read(), ['python','-c','import flask,sys;print(\"ok\",flask.__version__)'], 'ok'); print(r)"`
3. Create `tests/fixtures/flask.Dockerfile` first with the Dockerfile from Task 2.4 test.
4. Then manually edit a smaller version: `tests/fixtures/flask-slim.Dockerfile` using `python:3.12-alpine` + `pip install --no-cache-dir flask==3.0.0` + skip dev deps.
5. Run annotate on both, confirm second is materially smaller.

**Checkpoint:** Slim version produces a smaller `baseline_size` and still returns `ok=True`. This is the sanity check that the gates accept legitimate optimizations.

**Step 6: Commit fixtures**
```bash
git add tests/fixtures/
git commit -m "fixtures: flask Dockerfile pair for sanity-checking annotate"
git push
```

---

## Task 2.6: Scrape candidate Dockerfiles

**Why:** Need raw material before curation. Sources from plan §2.

**Files:**
- Modify: `src/dockermin/dataset/scrape.py`
- Create: `scripts/run_scrape.py`

**Step 1: Implementation**

```python
# src/dockermin/dataset/scrape.py
"""Scrape candidate Dockerfiles from official-images, awesome-compose, and GitHub code search."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess, json
from typing import Iterable

@dataclass(frozen=True)
class Candidate:
    source: str           # e.g. "official-images:python", "awesome-compose:flask-redis", "gh:owner/repo:path"
    url: str
    dockerfile: str
    ecosystem: str        # inferred from base image
    license: str | None   # best-effort

def _gh_search_dockerfiles(query: str, max_results: int = 50) -> list[dict]:
    """Use gh CLI to search code. Returns list of code search hits."""
    out = subprocess.check_output([
        "gh","api","-X","GET","search/code",
        "-f", f"q={query}",
        "-F", "per_page=100",
    ], text=True)
    return json.loads(out).get("items", [])[:max_results]

def fetch_official_images(limit: int = 100) -> Iterable[Candidate]:
    """Walk docker-library/official-images library/ for image dirs, follow GitRepo+Dockerfile pointers."""
    # Minimal: clone or use gh api to read library/ directory listings.
    raise NotImplementedError("implement using `gh api repos/docker-library/official-images/contents/library`")

def fetch_awesome_compose(limit: int = 50) -> Iterable[Candidate]:
    """List subdirs of docker/awesome-compose, pull each Dockerfile."""
    raise NotImplementedError

def fetch_github_search(limit: int = 100) -> Iterable[Candidate]:
    """gh search code 'filename:Dockerfile stars:>500' filter <200 LoC."""
    raise NotImplementedError
```

Then in `scripts/run_scrape.py`:
```python
"""Run all scrapers, write candidates to data/raw/candidates.jsonl. CPU-only."""
from pathlib import Path
import json
from dockermin.dataset.scrape import (
    fetch_official_images, fetch_awesome_compose, fetch_github_search,
)

OUT = Path("data/raw/candidates.jsonl")

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for c in fetch_official_images(limit=100):
            f.write(json.dumps(c.__dict__) + "\n")
        for c in fetch_awesome_compose(limit=50):
            f.write(json.dumps(c.__dict__) + "\n")
        for c in fetch_github_search(limit=150):
            f.write(json.dumps(c.__dict__) + "\n")

if __name__ == "__main__":
    main()
```

**Step 2: Fill in each fetcher.** This is real work and the bulk of Sunday morning. Pseudo-code outline:

- `fetch_official_images`: `gh api repos/docker-library/official-images/contents/library` returns list of image-name directories. Each has a manifest file pointing at a repo + commit + Dockerfile path. Parse the manifest, fetch the Dockerfile via raw github URL.
- `fetch_awesome_compose`: `gh api repos/docker/awesome-compose/contents` for top-level dirs. Each dir has `compose.yaml` and one or more service Dockerfiles.
- `fetch_github_search`: `gh api search/code -f q='filename:Dockerfile stars:>500 size:<5000'`. Score by stars, dedupe by content hash.

**Step 3: Test scraper locally**
```bash
.venv/bin/python scripts/run_scrape.py
wc -l data/raw/candidates.jsonl
```
**Checkpoint:** >= 200 unique entries.

**Step 4: Commit**
```bash
git add src/dockermin/dataset/scrape.py scripts/run_scrape.py
git commit -m "dataset: scrape candidates from official-images, awesome-compose, github"
git push
```

---

## Task 2.7: Default test_cmd inference per ecosystem

**Why:** Manual test_cmd writing for 200 Dockerfiles is the bottleneck. Auto-infer where possible.

**Files:**
- Modify: `src/dockermin/dataset/annotate.py`
- Modify: `tests/test_annotate.py`

**Step 1: Write failing test**
```python
from dockermin.dataset.annotate import infer_test_cmd

def test_infer_test_cmd_python():
    df = "FROM python:3.12-slim\nRUN pip install flask\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd[0] == "python"
    assert "ok" in expected.lower()

def test_infer_test_cmd_node():
    df = "FROM node:20-alpine\nRUN npm install express\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd[0] == "node"

def test_infer_test_cmd_falls_back_to_none_on_unknown():
    df = "FROM scratch\nCOPY app /app\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd is None and expected is None
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/dataset/annotate.py - APPEND
def infer_test_cmd(df_text: str) -> tuple[list[str] | None, str | None]:
    """Best-effort default test_cmd per ecosystem. Returns (None, None) if unknown."""
    text = df_text.lower()
    if "from python" in text or "pip install" in text:
        return (["python","-c","import sys;print('ok',sys.version_info[:2])"], "ok")
    if "from node" in text or "npm install" in text:
        return (["node","-e","console.log('ok',process.version)"], "ok")
    if "from golang" in text or "go build" in text:
        return (["/app/server","--version"], "")
    if "from openjdk" in text or "from eclipse-temurin" in text:
        return (["java","-version"], "")
    return (None, None)
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "annotate: infer default test_cmd from base image hints"
git push
```

---

## Task 2.8: Bulk-annotate candidates → curated set

**Files:**
- Create: `scripts/run_annotate.py`

**Step 1: Implementation**
```python
"""Bulk-annotate candidates.jsonl into curated.jsonl, dropping non-passing."""
from __future__ import annotations
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dockermin.dataset.annotate import annotate_one, infer_test_cmd

IN = Path("data/raw/candidates.jsonl")
OUT = Path("data/curated/triples.jsonl")
TARGET = 200
MAX_WORKERS = 8   # concurrent docker builds locally is limited; tune

def process(rec: dict) -> dict | None:
    df = rec["dockerfile"]
    cmd, expected = infer_test_cmd(df)
    if cmd is None: return None
    r = annotate_one(df, cmd, expected, build_timeout_s=300, test_timeout_s=30)
    if not r.ok: return None
    return {
        "id": rec.get("url","unknown"),
        "dockerfile": df,
        "test_cmd": cmd, "expected_substring": expected,
        "baseline_size": r.baseline_size, "baseline_build_s": r.baseline_build_s,
        "ecosystem": rec.get("ecosystem","unknown"),
        "source_url": rec.get("url",""), "license": rec.get("license"),
    }

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = 0; kept = 0
    with IN.open() as fin, OUT.open("w") as fout, ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = []
        for line in fin:
            seen += 1
            rec = json.loads(line)
            futures.append(ex.submit(process, rec))
            if kept >= TARGET: break
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                fout.write(json.dumps(res)+"\n"); fout.flush()
                kept += 1
                print(f"kept {kept}/{TARGET} (seen {seen})")
                if kept >= TARGET: break

if __name__ == "__main__":
    main()
```

**Step 2: Run**
```bash
.venv/bin/python scripts/run_annotate.py
```
Wall-clock estimate: 200 candidates × median 60s build × 1/8 parallel = ~25 min. Realistically 1-3h with build failures.

**Checkpoint:** `wc -l data/curated/triples.jsonl` is >= 100 (target 200, accept 100 floor per plan kill criterion).

**Step 3: Commit**
```bash
git add scripts/run_annotate.py
git commit -m "dataset: bulk-annotate to curated triples.jsonl"
git push
```

(Do NOT commit `data/` - it's gitignored. Dataset goes to HF.)

---

## Task 2.9: Push v0 dataset to HF Hub

**Files:**
- Create: `scripts/push_to_hf.py`

**Step 1: Implementation**
```python
"""Push curated/triples.jsonl to HF Hub as vtemian/dockermin-v0."""
from __future__ import annotations
import json
from pathlib import Path
from datasets import Dataset

IN = Path("data/curated/triples.jsonl")

def main() -> None:
    records = [json.loads(line) for line in IN.read_text().splitlines() if line.strip()]
    ds = Dataset.from_list(records)
    ds.push_to_hub(
        "vtemian/dockermin-v0",
        private=False,
        commit_message=f"v0: {len(records)} curated Dockerfile triples (parse+build+test verified)",
    )
    print(f"pushed {len(records)} records")

if __name__ == "__main__":
    main()
```

**Step 2: Eyeball 20 random entries (manual)**
```bash
.venv/bin/python -c "
import json, random
from pathlib import Path
recs = [json.loads(l) for l in Path('data/curated/triples.jsonl').read_text().splitlines()]
for r in random.sample(recs, min(20, len(recs))):
    print('---'); print(r['source_url']); print(r['dockerfile'][:200]); print('size', r['baseline_size'])
"
```
**Checkpoint:** No obvious garbage. Mix of ecosystems. Test_cmd matches base image.

**Step 3: Push and verify**
```bash
.venv/bin/python scripts/push_to_hf.py
```
Check at https://huggingface.co/datasets/vtemian/dockermin-v0.

**Step 4: Commit**
```bash
git add scripts/push_to_hf.py
git commit -m "dataset: push v0 to HF as vtemian/dockermin-v0"
git push
```

---

# Phase 3: Weekend 2 Saturday - reward function + pilot

## Task 3.1: TDD - prompt template + extractor

**Why:** Reward depends on extracting a Dockerfile from model output.

**Files:**
- Modify: `src/dockermin/reward/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write failing test**
```python
from dockermin.reward.prompts import (
    SYSTEM_PROMPT, USER_TEMPLATE, format_messages, extract_dockerfile,
)

def test_extract_dockerfile_from_fenced_block():
    text = """Sure, here is the optimized Dockerfile:

```dockerfile
FROM python:3.12-alpine
RUN pip install flask
```

That should be ~50MB smaller.
"""
    df = extract_dockerfile(text)
    assert df is not None
    assert df.startswith("FROM python:3.12-alpine")
    assert "pip install flask" in df

def test_extract_dockerfile_handles_bare_fence():
    text = "```\nFROM alpine\nRUN echo hi\n```"
    df = extract_dockerfile(text)
    assert df is not None and "FROM alpine" in df

def test_extract_dockerfile_returns_none_on_no_fence():
    assert extract_dockerfile("just prose, no code") is None

def test_format_messages_includes_dockerfile_and_test_cmd():
    msgs = format_messages("FROM python\n", ["python","-c","print('ok')"], "ok")
    assert any("system" == m["role"] for m in msgs)
    assert any("python" in m["content"] for m in msgs)
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/reward/prompts.py
"""Prompt template and Dockerfile extraction for Dockermin GRPO."""
from __future__ import annotations
import re

SYSTEM_PROMPT = (
    "You are a Dockerfile optimization engineer. Rewrite the given Dockerfile to be "
    "smaller while keeping it functionally equivalent. The rewritten image MUST still "
    "pass the provided test command and produce the expected output substring. Output "
    "ONLY the new Dockerfile in a single fenced code block tagged ```dockerfile. Do not "
    "include any explanation, prose, or additional code blocks."
)

USER_TEMPLATE = (
    "Optimize this Dockerfile.\n\n"
    "Original Dockerfile:\n```dockerfile\n{dockerfile}\n```\n\n"
    "Test command (run inside the built image): {test_cmd}\n"
    "Expected output substring: {expected}\n\n"
    "Output the optimized Dockerfile only."
)

_FENCE = re.compile(r"```(?:dockerfile|Dockerfile)?\s*\n(.*?)\n```", re.DOTALL)

def extract_dockerfile(text: str) -> str | None:
    """Return the first fenced ```dockerfile (or bare ```) block's body, or None."""
    m = _FENCE.search(text)
    return m.group(1).strip() if m else None

def format_messages(dockerfile: str, test_cmd: list[str], expected: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            dockerfile=dockerfile.strip(),
            test_cmd=" ".join(test_cmd),
            expected=expected,
        )},
    ]
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/reward/prompts.py tests/test_prompts.py
git commit -m "reward: prompt template + dockerfile fence extractor"
git push
```

---

## Task 3.2: TDD - reward gates (no docker, mocked sizes)

**Why:** Reward composition logic is testable without a real docker daemon if we inject the gates as functions.

**Files:**
- Modify: `src/dockermin/reward/gates.py`
- Modify: `tests/test_gates.py`

**Step 1: Write failing test**
```python
from dockermin.reward.gates import compute_score

def test_compute_score_parse_fail_returns_minus_point_1():
    s = compute_score(parse_ok=False, build_ok=False, test_ok=False,
                      command_count=0, baseline_size=100, new_size=0, dockerfile_text="")
    assert s == pytest.approx(-0.1)

def test_compute_score_too_few_commands_returns_minus_point_2():
    s = compute_score(parse_ok=True, build_ok=False, test_ok=False,
                      command_count=1, baseline_size=100, new_size=0, dockerfile_text="FROM scratch")
    assert s == pytest.approx(-0.2)

def test_compute_score_build_fail_returns_zero():
    s = compute_score(parse_ok=True, build_ok=False, test_ok=False,
                      command_count=3, baseline_size=100, new_size=0, dockerfile_text="...")
    assert s == pytest.approx(0.0)

def test_compute_score_build_pass_test_fail_returns_point_05():
    s = compute_score(parse_ok=True, build_ok=True, test_ok=False,
                      command_count=3, baseline_size=100, new_size=80, dockerfile_text="...")
    assert s == pytest.approx(0.05)

def test_compute_score_test_pass_full_reduction_returns_around_one():
    s = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                      command_count=3, baseline_size=100, new_size=10,
                      dockerfile_text="FROM gcr.io/distroless/python3-debian12\nCMD [\"x\"]")
    assert 0.95 <= s <= 1.10  # 0.5 base + 0.5 * 0.9 reduction + small distroless bonus

def test_shape_bonus_gated_on_test_pass():
    # Build passes but test fails: NO shape bonus from alpine string
    s = compute_score(parse_ok=True, build_ok=True, test_ok=False,
                      command_count=3, baseline_size=100, new_size=20,
                      dockerfile_text="FROM alpine\nRUN whatever")
    assert s == pytest.approx(0.05)  # NO alpine shape bonus when test failed

def test_latest_tag_penalty_applied_when_test_passes():
    s_with = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                           command_count=3, baseline_size=100, new_size=80,
                           dockerfile_text="FROM python:latest\nCMD [\"x\"]")
    s_without = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                              command_count=3, baseline_size=100, new_size=80,
                              dockerfile_text="FROM python:3.12-slim\nCMD [\"x\"]")
    assert s_without > s_with
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/reward/gates.py
"""Reward scoring: gates -> dense -> shape. Pure function for testability."""
from __future__ import annotations
import re

def compute_score(*, parse_ok: bool, build_ok: bool, test_ok: bool,
                  command_count: int, baseline_size: int, new_size: int,
                  dockerfile_text: str) -> float:
    if not parse_ok:
        return -0.1
    if command_count < 2:
        return -0.2
    if not build_ok:
        return 0.0
    if not test_ok:
        return 0.05
    reduction = max(0.0, (baseline_size - new_size) / max(1, baseline_size))
    dense = min(1.0, reduction)
    text = dockerfile_text.lower()
    shape = 0.0
    if re.search(r"from\s+\S*distroless", text): shape += 0.05
    if re.search(r"from\s+\S*alpine", text):    shape += 0.03
    if text.count("from ") >= 2:                shape += 0.05  # multi-stage
    if "rm -rf /var/lib/apt/lists" in text:     shape += 0.02
    if "--no-install-recommends" in text:       shape += 0.02
    if ":latest" in text or re.search(r"^from\s+\S+\s*$", text, re.M):
        shape -= 0.05
    if "from scratch" in text and " copy " not in text:
        shape -= 0.10
    return 0.5 + 0.5 * dense + shape
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/reward/gates.py tests/test_gates.py
git commit -m "reward: pure compute_score gates+dense+shape with test-pass gating"
git push
```

---

## Task 3.3: TDD - reward wiring (sync docker side)

**Files:**
- Modify: `src/dockermin/reward/dockermin_reward.py`
- Modify: `tests/test_reward.py`

**Step 1: Write failing test**
```python
from dockermin.reward.dockermin_reward import dockermin_reward

def test_dockermin_reward_garbage_completion_returns_negative():
    # No fence -> extract returns None -> parse_gate fails on "" -> score = -0.1
    completion = [{"role":"assistant","content":"just prose"}]
    info = {"baseline_size": 100, "test_cmd": ["true"], "expected_substring": ""}
    score = dockermin_reward(completion=completion, info=info)
    assert score == pytest.approx(-0.1)
```

**Step 2: Run, expect failure.**

**Step 3: Implementation**
```python
# src/dockermin/reward/dockermin_reward.py
"""Top-level reward function. Signature matches verifiers Rubric inspection."""
from __future__ import annotations
from .prompts import extract_dockerfile
from .gates import compute_score
from dockermin.dataset.annotate import parse_gate, build_gate, test_gate

def _completion_text(completion) -> str:
    """Verifiers passes completion either as str or list[message]."""
    if isinstance(completion, str): return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict): return last.get("content","")
    return ""

def dockermin_reward(completion, info, **kwargs) -> float:
    """Composite reward. Signature accepts arbitrary kwargs per verifiers Rubric convention."""
    text = _completion_text(completion)
    new_df = extract_dockerfile(text) or ""
    p = parse_gate(new_df)
    if not p.ok:
        return compute_score(parse_ok=False, build_ok=False, test_ok=False,
                             command_count=0, baseline_size=info["baseline_size"],
                             new_size=0, dockerfile_text=new_df)
    b = build_gate(new_df, timeout_s=300)
    if not b.ok:
        return compute_score(parse_ok=True, build_ok=False, test_ok=False,
                             command_count=p.command_count,
                             baseline_size=info["baseline_size"],
                             new_size=0, dockerfile_text=new_df)
    t = test_gate(b.tag, info["test_cmd"], info.get("expected_substring",""), timeout_s=30)
    return compute_score(
        parse_ok=True, build_ok=True, test_ok=t.ok,
        command_count=p.command_count,
        baseline_size=info["baseline_size"], new_size=b.size_bytes,
        dockerfile_text=new_df,
    )
```

**Step 4: Run, expect pass.**

**Step 5: Commit**
```bash
git add src/dockermin/reward/dockermin_reward.py tests/test_reward.py
git commit -m "reward: dockermin_reward composes extract + gates + compute_score"
git push
```

**Note:** This is the sync version. Wrap with `asyncio.to_thread` + `Semaphore(16)` when prime-rl calls it concurrently. See Task 3.5.

---

## Task 3.4: Local zero-shot smoke - 10 base-Qwen completions through reward

**Why:** Confirm the prompt template + extractor + reward flow works on real Qwen output before paying for GRPO.

**Files:**
- Create: `scripts/smoke_reward.py`

**Step 1: Implementation**
```python
"""Run base Qwen 2.5 Coder 7B Instruct on 10 dataset entries, compute reward."""
from __future__ import annotations
import json
from pathlib import Path
from anthropic import Anthropic   # cheap proxy - use Claude as a stand-in if no local Qwen yet
from dockermin.reward.prompts import format_messages
from dockermin.reward.dockermin_reward import dockermin_reward

def main() -> None:
    triples = [json.loads(l) for l in Path("data/curated/triples.jsonl").read_text().splitlines()]
    triples = triples[:10]
    client = Anthropic()
    for t in triples:
        msgs = format_messages(t["dockerfile"], t["test_cmd"], t["expected_substring"])
        # Use Claude Haiku 4.5 as fast cheap stand-in for "any code model emits a Dockerfile" check
        r = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=1024,
            system=msgs[0]["content"], messages=[msgs[1]],
        )
        completion = [{"role":"assistant","content": r.content[0].text}]
        score = dockermin_reward(completion=completion, info={
            "baseline_size": t["baseline_size"],
            "test_cmd": t["test_cmd"],
            "expected_substring": t["expected_substring"],
        })
        print(f"id={t['id'][-30:]:30s} score={score:+.3f} size={t['baseline_size']/1e6:.1f}MB")

if __name__ == "__main__":
    main()
```

**Step 2: Run**
```bash
.venv/bin/python scripts/smoke_reward.py
```

**Checkpoint:** All 10 scores are finite (no exceptions). At least 3 are > 0.5 (i.e., model produced a working smaller Dockerfile). Eyeball for reward hacking - any `:latest`, `FROM scratch`, missing CMD.

**Step 3: Commit**
```bash
git add scripts/smoke_reward.py
git commit -m "smoke: dockermin_reward on 10 zero-shot completions"
git push
```

---

## Task 3.5: Build a prime-rl environment package for Dockermin

**Why:** prime-rl expects an "environment" - the dataset + reward + parser wrapped in a verifiers Environment. Reference: `examples/alphabet_sort/` in prime-rl.

**Files:**
- Create: `prime_env/dockermin_env/__init__.py`
- Create: `prime_env/dockermin_env/dockermin_env.py`
- Create: `prime_env/dockermin_env/pyproject.toml`

**Step 1: Write environment module**
```python
# prime_env/dockermin_env/dockermin_env.py
"""Dockermin verifiers Environment for prime-rl. Single-turn: prompt -> Dockerfile -> reward."""
from __future__ import annotations
import verifiers as vf
from datasets import load_dataset
from dockermin.reward.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from dockermin.reward.dockermin_reward import dockermin_reward

def load_environment(**kwargs) -> vf.Environment:
    ds = load_dataset("vtemian/dockermin-v0", split="train")
    def fmt(ex):
        return {
            "prompt": [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":USER_TEMPLATE.format(
                    dockerfile=ex["dockerfile"],
                    test_cmd=" ".join(ex["test_cmd"]),
                    expected=ex["expected_substring"],
                )},
            ],
            "info": {
                "baseline_size": ex["baseline_size"],
                "test_cmd": ex["test_cmd"],
                "expected_substring": ex["expected_substring"],
            },
            "answer": "",  # not used for free-form reward
        }
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in ("prompt","info","answer")])
    return vf.SingleTurnEnv(
        dataset=ds,
        rubric=vf.Rubric(funcs=[dockermin_reward], weights=[1.0]),
    )
```

```toml
# prime_env/dockermin_env/pyproject.toml
[project]
name = "dockermin-env"
version = "0.0.1"
dependencies = ["verifiers", "datasets", "dockermin"]

[project.entry-points."verifiers.environments"]
dockermin = "dockermin_env.dockermin_env:load_environment"
```

**Step 2: Install locally on pod**
```bash
cd prime_env/dockermin_env && uv pip install -e .
vf-install dockermin   # makes it discoverable by prime-rl
```

**Step 3: Commit**
```bash
git add prime_env/
git commit -m "env: dockermin verifiers Environment for prime-rl"
git push
```

---

## Task 3.6: prime-rl TOML config for Dockermin

**Files:**
- Create: `configs/dockermin_pilot.toml`
- Create: `configs/dockermin_full.toml`

**Step 1: Pilot config (50 steps on subset)**
```toml
# configs/dockermin_pilot.toml
model = "Qwen/Qwen2.5-Coder-7B-Instruct"
max_steps = 50
batch_size = 8
rollouts_per_example = 8
learning_rate = 5e-6
lora_alpha = 64
oversampling_factor = 1.5
max_async_level = 2
trajectory_strategy = "interleaved"

[sampling]
max_tokens = 1024
temperature = 1.0

[[env]]
id = "dockermin"

[model.experimental]
lora = true
[model]
max_lora_rank = 32

[wandb]
project = "dockermin"
name = "pilot-50step"
entity = "vladtemian"

[checkpoints]
interval = 25
keep_cloud = 2

[infrastructure]
compute_size = "S"
```

**Step 2: Full config (200 steps)**
```toml
# configs/dockermin_full.toml
# Same as pilot, with:
max_steps = 200
batch_size = 32
[infrastructure]
compute_size = "L"
[wandb]
name = "full-200step"
```

**Step 3: Commit**
```bash
git add configs/
git commit -m "configs: prime-rl pilot + full TOMLs"
git push
```

---

## Task 3.7: Rent 8xH100 pod and run 50-step pilot

**Steps:**
1. `prime availability --gpu-type H100_80GB --gpu-count 8`
2. Pick cheapest provider.
3. `prime pods create --gpu-type H100_80GB --gpu-count 8 --provider <id>`
4. SSH in, clone prime-rl + dockermin repos, install all deps.
5. Set up DooD: `apt-get install -y docker.io docker-buildx-plugin` then verify `docker run hello-world` works inside the training container (the pod usually has a host-mounted docker socket; confirm).
6. Set up egress proxy (Task 3.8 next).
7. `vf-install dockermin` (from `prime_env/dockermin_env`).
8. `bash scripts/tmux.sh` (prime-rl launcher).
9. `uv run rl @ configs/dockermin_pilot.toml`

**Watch wandb:** reward curve should trend up across 50 steps. Mean reward at step 25 should be > step 5.

**Hard exit:** if reward is flat or NaN at step 20, kill the run. Drop temperature, raise beta, simplify reward (drop shape bonuses), re-run.

**Time / cost:** 50 steps × ~2 min/step = 100 min wall-clock on 8xH100 at ~$12-20/hr = $20-35 spend.

**No commit (no code change in this task).** Update `cost_log.md` + `journal.md` after.

---

## Task 3.8: DooD + egress proxy setup on training pod

**Files:**
- Create: `scripts/setup_pod_docker.sh`
- Create: `scripts/setup_pod_proxy.sh`

**Step 1: Docker setup script**
```bash
#!/bin/bash
# scripts/setup_pod_docker.sh
set -euo pipefail

# Install docker if not present
if ! command -v docker >/dev/null; then
  apt-get update && apt-get install -y docker.io docker-buildx-plugin
fi

# Verify daemon is up
docker ps >/dev/null || { echo "docker daemon not reachable"; exit 1; }

# buildx builder with cache export
docker buildx create --name dockermin \
  --driver docker-container \
  --driver-opt network=host \
  --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=10000000 \
  --buildkitd-flags '--allow-insecure-entitlement=network.host --oci-worker-gc-keepstorage=20480' \
  --bootstrap --use

# daemon.json tuning
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<EOF
{
  "max-concurrent-downloads": 16,
  "max-concurrent-uploads": 16,
  "storage-driver": "overlay2"
}
EOF
systemctl restart docker || service docker restart

# Periodic prune cron
cat > /etc/cron.d/dockermin-prune <<'EOF'
*/30 * * * * root docker system prune -af --filter "until=2h" >/dev/null 2>&1
*/30 * * * * root docker buildx prune -af --filter "until=2h" --keep-storage=20GB --builder dockermin >/dev/null 2>&1
EOF

echo "docker setup ok"
```

**Step 2: Proxy setup script**
```bash
#!/bin/bash
# scripts/setup_pod_proxy.sh - logged HTTP proxy allowlist
set -euo pipefail

apt-get install -y tinyproxy
cat > /etc/tinyproxy/tinyproxy.conf <<'EOF'
Port 8888
Listen 127.0.0.1
LogFile "/var/log/tinyproxy/proxy.log"
LogLevel Info
MaxClients 200
Allow 127.0.0.1
# Allowlist
FilterURLs Yes
Filter "/etc/tinyproxy/filter"
EOF

cat > /etc/tinyproxy/filter <<'EOF'
^https?://pypi\.org
^https?://files\.pythonhosted\.org
^https?://registry\.npmjs\.org
^https?://registry-1\.docker\.io
^https?://auth\.docker\.io
^https?://production\.cloudflare\.docker\.com
^https?://archive\.ubuntu\.com
^https?://security\.ubuntu\.com
^https?://deb\.debian\.org
^https?://security\.debian\.org
^https?://github\.com
^https?://codeload\.github\.com
^https?://objects\.githubusercontent\.com
EOF

systemctl restart tinyproxy
echo "proxy on 127.0.0.1:8888 with allowlist"
```

**Step 3: Run both on the pod**
```bash
bash scripts/setup_pod_docker.sh
bash scripts/setup_pod_proxy.sh
```

**Step 4: Commit**
```bash
git add scripts/setup_pod_docker.sh scripts/setup_pod_proxy.sh
git commit -m "scripts: DooD + tinyproxy allowlist setup for training pod"
git push
```

**Checkpoint:** `docker run --rm -e http_proxy=http://127.0.0.1:8888 -e https_proxy=http://127.0.0.1:8888 alpine wget -qO- https://github.com` returns HTML. Same command against `https://random-cdn.example.com` returns proxy denial.

---

# Phase 4: Weekend 2 Sunday - full training run

## Task 4.1: Pilot decision - go/no-go

**Decision criteria (review pilot wandb at start of Sunday):**

| Signal | Threshold | Action |
|---|---|---|
| Mean reward at step 50 | >= 0.5 | GO |
| % rollouts that pass gates (parse + build + test) at step 50 | >= 60% | GO |
| Reward stable (no NaN, no collapse) | yes | GO |
| Daily reward-hacking audit % `:latest` | < 10% | GO |

If GO: proceed to Task 4.2. If NO-GO: per kill criterion 2, simplify reward (drop shape bonuses, use only gates + size) and rerun pilot.

**No commit.** Update `journal.md` with the decision.

---

## Task 4.2: Full 200-step run

**Steps:**
1. From the same 8xH100 pod (already up from Saturday), `uv run rl @ configs/dockermin_full.toml`
2. Run for 200 steps. Expected wall-clock: 6-10 hours.
3. Every 25 steps, prime-rl auto-saves checkpoint. Verify in wandb.

**Active monitoring:**
- wandb reward curve (should keep climbing or plateau, not collapse)
- pod GPU utilization (should be > 70% on inference + training GPUs)
- Daily reward-hacking audit script (Task 4.3)

**Cost:** 8-10 GPU-hours × $12-20/hr = $100-200 for the run.

**Hard exit triggers:**
- Reward collapses (drops > 50% from peak): kill, save last good checkpoint, declare null result candidate
- KL divergence explodes (> 1.0): lower learning rate, restart from last checkpoint
- Build/daemon errors > 30% of rollouts: pause, investigate egress / cache / daemon health

**No commit during run.** Add a journal entry every 2 hours of run time.

---

## Task 4.3: Daily reward-hacking audit script

**Files:**
- Create: `scripts/audit_rollouts.py`

**Step 1: Implementation**
```python
"""Pull 20 random rollouts from the latest checkpoint, eyeball reward-hacking patterns."""
from __future__ import annotations
import random, re, json
from pathlib import Path

def audit(rollouts_dir: Path) -> dict:
    files = list(rollouts_dir.glob("*.jsonl"))
    rollouts = []
    for f in files:
        for line in f.read_text().splitlines():
            try: rollouts.append(json.loads(line))
            except: continue
    sample = random.sample(rollouts, min(20, len(rollouts)))
    stats = {
        "n": len(sample),
        "from_scratch": sum(1 for r in sample if "from scratch" in r.get("completion","").lower()),
        "latest_tag": sum(1 for r in sample if ":latest" in r.get("completion","")),
        "no_cmd": sum(1 for r in sample if "CMD" not in r.get("completion","") and "ENTRYPOINT" not in r.get("completion","")),
        "mean_lines": sum(len(r.get("completion","").splitlines()) for r in sample) / max(1, len(sample)),
    }
    return stats

if __name__ == "__main__":
    import sys
    stats = audit(Path(sys.argv[1] if len(sys.argv)>1 else "rollouts/"))
    print(json.dumps(stats, indent=2))
```

**Step 2: Run after every checkpoint (every 25 steps)**
```bash
.venv/bin/python scripts/audit_rollouts.py rollouts/latest/
```

**Checkpoint:** `from_scratch < 10%`, `latest_tag < 5%`, `no_cmd < 5%`, `mean_lines > 4`.

**Step 3: Commit**
```bash
git add scripts/audit_rollouts.py
git commit -m "scripts: reward-hacking audit on rollouts"
git push
```

---

## Task 4.4: Best checkpoint -> HF

**Files:**
- Create: `scripts/push_adapter.py`

**Step 1: Implementation**
```python
"""Push best LoRA checkpoint to HF as vtemian/dockermin-qwen7b-lora-v1."""
from __future__ import annotations
import sys
from huggingface_hub import HfApi

def main(ckpt_dir: str) -> None:
    api = HfApi()
    api.create_repo("vtemian/dockermin-qwen7b-lora-v1", repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=ckpt_dir, repo_id="vtemian/dockermin-qwen7b-lora-v1",
                      repo_type="model", commit_message="initial LoRA checkpoint")

if __name__ == "__main__":
    main(sys.argv[1])
```

**Step 2: Run with best checkpoint dir (select from wandb-tagged step with highest holdout reward)**
```bash
.venv/bin/python scripts/push_adapter.py checkpoints/step_175/
```

**Step 3: Commit**
```bash
git add scripts/push_adapter.py
git commit -m "scripts: push LoRA to HF"
git push
```

**Step 4: Terminate the pod**
```bash
prime pods terminate <pod_id>
```
Update cost_log.md. This is the biggest billed session of the project.

---

# Phase 5: Weekend 3 Saturday - eval + CLI

## Task 5.1: Eval harness skeleton

**Files:**
- Modify: `src/dockermin/eval/baselines.py`
- Create: `scripts/run_eval.py`

**Step 1: Eval orchestrator**
```python
# src/dockermin/eval/baselines.py
"""Run each baseline over the holdout, return per-Dockerfile (size, test_passes, time)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class EvalEntry:
    baseline: str
    triple_id: str
    new_size_bytes: int | None
    test_passes: bool
    elapsed_s: float
    error: str = ""
```

The `run_eval.py` script iterates triples + baselines and writes results to `data/eval/results.jsonl`.

**Step 2: Commit**
```bash
git add src/dockermin/eval/baselines.py scripts/run_eval.py
git commit -m "eval: harness skeleton"
git push
```

---

## Task 5.2: Baseline 1 - Qwen 2.5 Coder 7B zero-shot

Single function: load base model via vLLM on the pod, run the prompt over holdout, collect Dockerfile, run annotate-style gates.

Skipping copy-paste here for brevity - same pattern as Task 3.4 smoke but with all 150 holdouts and real Qwen.

---

## Task 5.3: Baseline 2 - GPT-4o zero-shot
## Task 5.4: Baseline 3 - Sonnet 4.6 zero-shot
## Task 5.5: Baseline 4 - hadolint mechanical apply
## Task 5.6: Baseline 5 - SlimToolkit `slim build`
## Task 5.7: Baseline 6 - manual best-practice rewriter (apply 5-rule transform)
## Task 5.8: Baseline 7 - Claude Sonnet 4.6 in Claude Code agent loop

For Baseline 7, the critical one:

**Files:**
- Create: `src/dockermin/eval/agent_loop.py`

```python
"""Claude Code headless agent-loop baseline. Sonnet 4.6, 5 turns, docker tool via bash allowlist."""
from __future__ import annotations
import json, subprocess, tempfile, shutil
from pathlib import Path

def run_agent_loop(dockerfile: str, test_cmd: list[str], expected: str, max_turns: int = 5) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="agentloop_"))
    (workdir / "Dockerfile").write_text(dockerfile)
    prompt = (
        f"Optimize ./Dockerfile to be smaller. Build it with docker build. "
        f"Verify test_cmd passes inside the built image: {' '.join(test_cmd)} should output a string containing {expected!r}. "
        f"Iterate up to {max_turns} times. Stop when both: image is smaller than original AND test passes."
    )
    try:
        result = subprocess.run([
            "claude","-p", prompt,
            "--output-format","json",
            "--model","claude-sonnet-4-6",
            "--max-turns", str(max_turns),
            "--max-budget-usd","0.50",
            "--permission-mode","bypassPermissions",
            "--allowedTools","Bash(docker build *)","Bash(docker run *)","Read","Edit","Write",
            "--cwd", str(workdir),
        ], capture_output=True, text=True, timeout=600)
        meta = json.loads(result.stdout) if result.returncode == 0 else {}
        final_df = (workdir / "Dockerfile").read_text()
        return {"final_dockerfile": final_df, "cost_usd": meta.get("total_cost_usd",0),
                "turns": meta.get("num_turns",0), "exit_code": result.returncode}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
```

**Step 5: Commit (each baseline its own commit)**
```bash
git add src/dockermin/eval/agent_loop.py
git commit -m "eval: claude code agent-loop baseline 7"
git push
```

---

## Task 5.9: Run all baselines + dockermin LoRA over 150 holdouts

```bash
.venv/bin/python scripts/run_eval.py --baselines qwen_zs gpt4o sonnet_zs hadolint slim manual agent_loop dockermin --out data/eval/results.jsonl
```

Wall-clock estimate: ~3-6h depending on Claude API rate limits. Cost: agent loop ~$40, GPT-4o ~$15, Sonnet zero-shot ~$5, others free.

**Checkpoint:** `data/eval/results.jsonl` has rows for each (baseline, triple) pair. Counts make sense (~1200 rows for 8 baselines × 150 triples).

---

## Task 5.10: Generate leaderboard

**Files:**
- Create: `scripts/leaderboard.py`
- Create: `docs/leaderboard.md`

```python
"""Aggregate eval/results.jsonl into a leaderboard markdown table."""
# group by baseline: mean reduction | conditional on test pass | test pass rate | mean build time | mean CVE delta
```

**Step:** Run and review the table. **The headline number we care about:** dockermin reduction conditional on test pass, vs agent loop reduction conditional on test pass.

If dockermin >= agent_loop on a meaningful slice: we have a result.
If dockermin < agent_loop everywhere: per the additional kill criterion, declare null result.

**Step 6: Commit**
```bash
git add scripts/leaderboard.py docs/leaderboard.md
git commit -m "eval: leaderboard with primary + secondary metrics"
git push
```

---

## Task 5.11: `dockermin` CLI

**Files:**
- Modify: `src/dockermin/cli.py`

```python
"""dockermin <dockerfile_path> --test '<cmd>' --expect <substring> --model <hf_id>"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from .reward.prompts import format_messages, extract_dockerfile

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dockerfile", type=Path)
    p.add_argument("--test", required=True)
    p.add_argument("--expect", required=True)
    p.add_argument("--model", default="vtemian/dockermin-qwen7b-lora-v1")
    args = p.parse_args()
    df = args.dockerfile.read_text()
    msgs = format_messages(df, args.test.split(), args.expect)
    # Inference: load base Qwen + LoRA via vllm or HF transformers. For CLI ship the simpler HF path.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", torch_dtype="bfloat16", device_map="auto")
    model = PeftModel.from_pretrained(base, args.model)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=1024, temperature=0.2, do_sample=True)
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    new_df = extract_dockerfile(text)
    if new_df is None:
        print("ERROR: model output had no fenced dockerfile block", file=sys.stderr)
        return 1
    print(new_df)
    return 0
```

**Step:** Commit.

---

# Phase 6: Weekend 3 Sunday - ship

## Task 6.1: Write blog post OR NEGATIVE_RESULT.md

**Files:**
- Create: `docs/blog.md` (positive) OR `NEGATIVE_RESULT.md` (null result)

Per the decision from Task 5.10. Either path ships the dataset + benchmark as contribution.

## Task 6.2: Update repo README with leaderboard badge + dataset link

## Task 6.3: HF dataset card + model card polish

## Task 6.4: X thread tagging @willccbb + Prime Intellect

Optional. Status-shaped but legitimate for a learning artifact ship.

---

# Risk register (active during execution)

| Risk | Where addressed | Status |
|---|---|---|
| Build latency dominates rollout | BuildKit cache + buildx + tinyproxy | mitigated, verify in Task 3.7 |
| prime-rl PR #1392 LoRA+NCCL bug | Task 1.3 pin decision | tracked |
| vLLM 0.7.3 LoRA hotswap broken on Qwen 7B | Task 1.5 smoke | mitigated, verify in 1.5 |
| Test command leakage / reward hacking | Task 4.3 daily audit | accepted v0, fix v1 |
| docker daemon concurrency bottleneck | daemon.json + buildx | mitigated, verify in 3.7 |
| Agent loop baseline wins | Task 5.8 + 5.10 | acknowledged kill criterion |
| Dataset curation > 1 weekend | drop to 100 triples | per plan kill criterion 1 |
| Reward unstable | drop shape bonuses | per plan kill criterion 2 |
| Cost overrun | $200/$400/$500 triggers | per plan §8 |

---

# What ships at end of weekend 3

- HF adapter: `vtemian/dockermin-qwen7b-lora-v1`
- HF dataset: `vtemian/dockermin-v0`
- GitHub repo: `github.com/vtemian/dockermin` (already created)
- Benchmark suite + leaderboard (`docs/leaderboard.md`)
- `dockermin` CLI (`pip install` from repo)
- Blog post OR `NEGATIVE_RESULT.md`
- Updated journal capturing what was learned
