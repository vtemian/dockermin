# Dockermin Audit-Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the blockers a 4-agent implementation audit found, in severity order, so a real GRPO run produces an honest result instead of a model that games the metric. The headline fix removes a reward-hacking leak; the rest unblock training start, correct reward execution, and make the eval honest.

**Architecture:** Five phases mapped to the audit's severity tiers. Tier 1 (reward integrity) and the async-reward correctness are TDD - the reward is pinned to exact values and a wrong fix silently corrupts training. Dataset, pinning, pod-readiness, and eval-honesty are imperative tasks with explicit verification. No new runtime deps.

**Tech Stack:** unchanged - Python 3.11, verifiers/prime-rl, ruff+mypy gates (must stay green), pytest. All changes keep `make quality` clean.

**Source audits (this session):** dataset, reward, training-wiring, eval. **Design research (this session):** reward-hack mitigation, verifiers reward-execution model.

---

## Ground truth gathered (do not re-derive)

- **Reward leak:** `prompts.py:USER_TEMPLATE` copies `test_cmd`+`expected_substring` into the model-visible prompt. But verifiers passes `info` to the Rubric and NEVER to the policy - the model-invisible channel already exists. Removing the copy kills the leak with zero schema change.
- **Verifiers execution:** rollouts run on ONE asyncio event loop; a sync reward blocks all concurrent rollouts. Reward funcs may be `async def` (verifiers `await`s them via `maybe_await`). Verifiers `score_group` does `asyncio.gather` with NO semaphore; it swallows reward exceptions to `0.0`. Reward runs in the **orchestrator** process (docker must be there).
- **Dataset:** `triples.jsonl` = 16 baselines; `triples_with_variants.jsonl` = 93 rows (11 distinct bases + 82 variants), all parse clean, schemas compatible. `push_to_hf.py` pushes the wrong (16-row) file. Real ceiling: 11-16 functionally-distinct bases - variants give volume, not base-diversity. Split MUST group by `base_id`.
- **prime-rl:** unpinned in `pyproject.toml:13`; needs a pin >= commit `91182b7` (PR #1392, NCCL+LoRA fix).
- **Pod:** `setup_pod_docker.sh` assumes systemd (PI pods are containers w/ host socket); proxy allowlist missing `huggingface.co`; no docker preflight; `build_gate` builder/cache never wired on pod.
- **Eval:** holdout falls back to "last 150 train rows" (not disjoint); `agent_loop` cost parsed only on `returncode==0` (under-reports); leaderboard silently drops triples missing `baseline_size`, no per-baseline N guard.

---

# Phase 1: Reward integrity (Tier 1 - stop the model learning to cheat)

## Task 1.1: Remove the test leak from the prompt (TDD)

**Files:**
- Modify: `src/dockermin/reward/prompts.py`
- Modify: `prime_env/dockermin_env/dockermin_env.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write the failing test** in `tests/test_prompts.py`:
```python
def test_prompt_does_not_leak_expected_substring() -> None:
    """The model must NOT see the test_cmd or expected output - that is the
    reward-hack vector (FROM scratch + RUN echo <expected>)."""
    msgs = format_messages("FROM python:3.12\nRUN pip install flask\n")
    blob = " ".join(m["content"] for m in msgs)
    assert "expected" not in blob.lower()
    assert "test_cmd" not in blob.lower()
    assert "ok 3.0.0" not in blob  # a sample expected value must never appear
```

**Step 2: Run, expect failure** (format_messages currently takes test_cmd+expected and interpolates them):
```bash
.venv/bin/pytest tests/test_prompts.py::test_prompt_does_not_leak_expected_substring -v
```
Expected: FAIL (TypeError on signature, or assertion - "expected" present).

**Step 3: Implement.** In `prompts.py`:
- Change `format_messages(dockerfile)` to take ONLY the dockerfile (drop `test_cmd`, `expected` params).
- New `USER_TEMPLATE`:
```python
USER_TEMPLATE = (
    "Optimize this Dockerfile to be smaller while keeping it functionally "
    "equivalent - same runtime, same installed packages, same entrypoint "
    "behaviour.\n\n"
    "Original Dockerfile:\n```dockerfile\n{dockerfile}\n```\n\n"
    "Output the optimized Dockerfile only, in a single fenced ```dockerfile block."
)
```
- Trim `SYSTEM_PROMPT`'s "MUST still pass the provided test command and produce the expected output substring" -> "MUST remain functionally equivalent (same packages importable, same entrypoint)."

In `dockermin_env.py` `fmt()`: stop passing test_cmd/expected into `format_messages`; keep `info` exactly as-is (test_cmd + expected_substring + baseline_size stay in info for the reward).

**Step 4: Run, expect pass** + full suite:
```bash
.venv/bin/pytest tests/ -q
```
Expected: the new test passes; existing prompt tests updated for the new signature; 98+ pass.

**Step 5: Commit**
```bash
git add src/dockermin/reward/prompts.py prime_env/dockermin_env/dockermin_env.py tests/test_prompts.py
git commit -m "fix(reward): stop leaking test_cmd/expected into the prompt (reward-hack vector)"
```

---

## Task 1.2: Anti-cheat regression test + harsher empty-image handling (TDD)

**Files:**
- Modify: `src/dockermin/reward/gates.py`
- Modify: `tests/test_gates.py`
- Modify: `src/dockermin/reward/dockermin_reward.py` (only if a gate-level change needs plumbing)

**Step 1: Write the failing test** in `tests/test_gates.py` - encode that the known cheats score low:
```python
def test_from_scratch_echo_cheat_scores_low() -> None:
    """FROM scratch + RUN echo (the classic cheat) must not out-score an
    honest shrink, even if it somehow passed the test gate."""
    cheat = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                          command_count=2, baseline_size=100, new_size=1,
                          dockerfile_text="FROM scratch\nRUN echo ok")
    honest = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                          command_count=4, baseline_size=100, new_size=80,
                          dockerfile_text="FROM python:3.12-slim\nRUN pip install --no-cache-dir flask\nCMD [\"python\",\"-m\",\"flask\"]")
    assert cheat < honest

def test_suspiciously_tiny_image_soft_penalised() -> None:
    """An image <2% of baseline with no multi-stage COPY is likely empty -
    soft penalty, not a hard reject (legit static distroless binaries exist)."""
    s = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                     command_count=2, baseline_size=100, new_size=1,
                     dockerfile_text="FROM scratch\nRUN echo ok")
    s_legit = compute_score(parse_ok=True, build_ok=True, test_ok=True,
                     command_count=3, baseline_size=100, new_size=1,
                     dockerfile_text="FROM scratch\nCOPY --from=build /app /app\nENTRYPOINT [\"/app\"]")
    assert s < s_legit
```

**Step 2: Run, expect failure.**

**Step 3: Implement** in `gates.py` `_shape_penalty` (keep it a pure function, keep existing exact-value tests green):
- Tighten the `from scratch` rule: currently -0.10 when no `\bcopy\b`. Add a "suspiciously tiny" soft penalty: if `new_size < baseline_size * 0.02` AND no `copy --from` (no real multi-stage artifact), subtract an additional penalty (e.g. -0.15). Legit `COPY --from=` static binaries are exempt.
- Do NOT try to detect `RUN echo <expected>` by matching expected - after Task 1.1 the reward no longer holds `expected` in a model-visible place, and matching it here is brittle. Rely on the tiny-image heuristic + the hardened probes (Task 1.3) instead.

**Step 4: Run** - new tests pass, ALL existing exact-value gate tests still pass (the reward math for normal cases must be unchanged):
```bash
.venv/bin/pytest tests/test_gates.py -v
```

**Step 5: Commit**
```bash
git add src/dockermin/reward/gates.py tests/test_gates.py
git commit -m "fix(reward): soft-penalise suspiciously tiny images, regression-test the echo cheat"
```

---

## Task 1.3: Harden functional probes (TDD where pure)

**Files:**
- Modify: `src/dockermin/dataset/annotate.py` (`infer_test_cmd` / `_TEST_CMD_BY_ECOSYSTEM`)
- Modify: `tests/test_annotate.py`

**Step 1: Write failing tests** for the new probe shapes:
```python
def test_infer_test_cmd_python_imports_real_package() -> None:
    cmd, expected = infer_test_cmd("FROM python:3.12-slim\nRUN pip install flask\n")
    # probe must import a real package + print a value that requires the
    # interpreter to actually run - not a constant the model can echo
    assert cmd[0] == "python"
    assert "import" in " ".join(cmd)

def test_infer_test_cmd_node_requires_module() -> None:
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install express\n")
    assert cmd[0] == "node"
    assert "require" in " ".join(cmd)
```

**Step 2: Run, expect failure.**

**Step 3: Implement.** Replace the weak probes with ones that need the real runtime:
- python: `["python", "-c", "import sys; import importlib; importlib.import_module('flask'); print('PYOK', sys.version_info[0])"]` expecting `"PYOK"` (still a constant, but now requires `import flask` to not crash - a `FROM scratch` image has no python, exits non-zero, fails the gate). Where the installed package is unknown, fall back to importing the stdlib + a build-marker file check.
- node: `["node", "-e", "require('express'); console.log('NODEOK', process.version)"]` expecting `"NODEOK"`.
- java: keep `java -version` expecting `openjdk` (already requires the JRE).

The key property: the probe runs INSIDE the image, so it requires the real interpreter/runtime to exist. `RUN echo` at build time cannot satisfy a runtime probe that imports a package.

**Step 4: Run + note** these change `expected_substring` values, so the dataset must be re-annotated (Phase 3 handles the regen). Tests pass.

**Step 5: Commit**
```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "fix(dataset): runtime-import probes resistant to RUN echo cheats"
```

---

# Phase 2: Reward execution correctness (async + failure isolation)

## Task 2.1: Make dockermin_reward async, threaded, semaphore-capped, failure-isolated (TDD)

**Files:**
- Modify: `src/dockermin/reward/dockermin_reward.py`
- Modify: `tests/test_reward.py`

**Why:** verifiers runs all rollouts on one event loop; the current sync 300s build blocks every concurrent rollout. And verifiers swallows reward exceptions to 0.0, so a malformed `info` row silently looks like a bad Dockerfile and pollutes the gradient.

**Step 1: Write failing tests** (use `pytest.mark.asyncio` - add `pytest-asyncio` to dev deps, or test via `asyncio.run`):
```python
import asyncio

def test_reward_is_async_and_handles_malformed_info() -> None:
    """Malformed info must yield a defined score, not raise (verifiers would
    swallow a raise to 0.0, indistinguishable from a real build failure)."""
    score = asyncio.run(dockermin_reward(
        completion=[{"role": "assistant", "content": "no fence here"}],
        info={},  # missing baseline_size/test_cmd
    ))
    assert score == 0.0

def test_reward_garbage_completion_still_negative() -> None:
    score = asyncio.run(dockermin_reward(
        completion=[{"role": "assistant", "content": "prose only"}],
        info={"baseline_size": 100, "test_cmd": ["true"], "expected_substring": "x"},
    ))
    assert score == pytest.approx(-0.1)  # parse gate fails on empty df
```

**Step 2: Run, expect failure** (current reward is sync; `asyncio.run` on a sync func that returns float actually works, so the malformed-info test is the real driver - it currently raises KeyError).

**Step 3: Implement** per the researched design:
```python
import asyncio
import os

_BUILD_SEM = asyncio.Semaphore(int(os.getenv("DOCKERMIN_MAX_BUILDS", "6")))

async def dockermin_reward(completion: ..., info: ..., **_kwargs: object) -> float:
    try:
        text = _completion_text(completion)
        new_df = extract_dockerfile(text) or ""
        baseline_size = info["baseline_size"]
        test_cmd = info["test_cmd"]
        expected = info.get("expected_substring", "")
    except (KeyError, TypeError):
        return 0.0  # malformed sample: defined score, not a swallowed raise

    p = parse_gate(new_df)  # pure, microseconds - stays inline
    if not p.ok:
        return compute_score(parse_ok=False, build_ok=False, test_ok=False,
                            command_count=0, baseline_size=baseline_size,
                            new_size=0, dockerfile_text=new_df)
    async with _BUILD_SEM:
        try:
            b = await asyncio.to_thread(build_gate, new_df, 300)
            if not b.ok:
                return compute_score(parse_ok=True, build_ok=False, test_ok=False,
                                    command_count=p.command_count, baseline_size=baseline_size,
                                    new_size=0, dockerfile_text=new_df)
            t = await asyncio.to_thread(run_test_gate, b.tag, test_cmd, expected, 30)
        except Exception:  # noqa: BLE001 - docker hiccup -> score as build failure, never crash the loop
            return compute_score(parse_ok=True, build_ok=False, test_ok=False,
                                command_count=p.command_count, baseline_size=baseline_size,
                                new_size=0, dockerfile_text=new_df)
    return compute_score(parse_ok=True, build_ok=True, test_ok=t.ok,
                        command_count=p.command_count, baseline_size=baseline_size,
                        new_size=b.size_bytes, dockerfile_text=new_df)
```
Notes: semaphore acquired in the async layer BEFORE `to_thread` (asyncio.Semaphore is not thread-safe). `build_gate`/`run_test_gate` in annotate.py stay sync (also used by the offline annotator). N=6 default, env-tunable.

**Step 4: Run** the reward tests + full suite. Add `pytest-asyncio` to `[project.optional-dependencies] dev` if using the marker; the `asyncio.run` approach needs no plugin.

**Step 5: Commit**
```bash
git add src/dockermin/reward/dockermin_reward.py tests/test_reward.py pyproject.toml
git commit -m "fix(reward): async + to_thread + Semaphore(6) + failure isolation (unblock event loop)"
```

---

# Phase 3: Dataset (push the right data, honest split)

## Task 3.1: Fix push_to_hf to push the variant set + HF_TOKEN guard

**Files:**
- Modify: `scripts/push_to_hf.py`

**Step 1:** Change `IN` from `data/curated/triples.jsonl` to `data/curated/triples_with_variants.jsonl` (93 rows). Add an explicit token check at the top:
```python
import os
if not (os.getenv("HF_TOKEN") or Path.home().joinpath(".cache/huggingface/token").exists()):
    raise SystemExit("HF_TOKEN not set and not logged in. Run `huggingface-cli login` or set HF_TOKEN.")
```

**Step 2:** Do NOT push yet (needs the split from Task 3.2). Commit the file fix + token guard.
```bash
git add scripts/push_to_hf.py
git commit -m "fix(dataset): push the variant set (93 rows) not triples.jsonl (16); guard HF_TOKEN"
```

---

## Task 3.2: Grouped train/test split by base_id (TDD)

**Files:**
- Create: `src/dockermin/dataset/split.py`
- Create: `tests/test_split.py`
- Modify: `scripts/push_to_hf.py`

**Why:** variants share `base_id` with their base; a random split leaks (a test variant whose base is in train is not a holdout). Split by base.

**Step 1: Write failing test** in `tests/test_split.py`:
```python
from dockermin.dataset.split import grouped_train_test_split

def test_split_keeps_all_variants_of_a_base_on_one_side() -> None:
    rows = [
        {"id": "b1", "base_id": None}, {"id": "b1-v0", "base_id": "b1"},
        {"id": "b2", "base_id": None}, {"id": "b2-v0", "base_id": "b2"},
    ]
    train, test = grouped_train_test_split(rows, test_frac=0.5, seed=0)
    train_bases = {r.get("base_id") or r["id"] for r in train}
    test_bases = {r.get("base_id") or r["id"] for r in test}
    assert train_bases.isdisjoint(test_bases)  # no base on both sides
```

**Step 2: Run, expect failure.**

**Step 3: Implement** `grouped_train_test_split(rows, test_frac, seed)`: group rows by `base_id or id`, shuffle the GROUP keys with the seed, assign whole groups to train/test by `test_frac`. Returns `(train_rows, test_rows)`.

**Step 4:** Wire into `push_to_hf.py`: build `DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})` and `push_to_hub(...)`. With only 11-16 bases, use `test_frac` that yields >= 3 test bases (document the small-N caveat in the dataset card).

**Step 5: Run + commit**
```bash
git add src/dockermin/dataset/split.py tests/test_split.py scripts/push_to_hf.py
git commit -m "feat(dataset): grouped-by-base train/test split so holdout doesn't leak"
```

---

## Task 3.3: Train env uses train split; eval uses test split

**Files:**
- Modify: `prime_env/dockermin_env/dockermin_env.py`
- Modify: `scripts/run_eval.py`

**Step 1:** `dockermin_env.load_environment` -> `load_dataset("vtemian/dockermin-v0", split="train")` explicitly. Optionally pass `eval_dataset=load_dataset(..., split="test")` to `SingleTurnEnv` so checkpoint evals use the holdout.

**Step 2:** `run_eval._load_holdout` -> load `split="test"` and REMOVE the "last 150 train rows" fallback; if no test split exists, raise a clear error (don't silently eval on training data).

**Step 3:** mypy/ruff clean, commit.
```bash
git add prime_env/dockermin_env/dockermin_env.py scripts/run_eval.py
git commit -m "fix(eval): train env uses train split, eval uses disjoint test split (no leak fallback)"
```

---

## Task 3.4: Document the dataset-diversity reality

**Files:**
- Modify: `docs/journal.md` (or a `docs/dataset_card.md`)

**Step 1:** Write the honest constraint: 11-16 functionally-distinct bases + 82 variants = high row count but low base diversity. The kill-criterion "100 floor" is met in rows (~93) but NOT in distinct bases. State the implication: a pilot run is meaningful as a methodology test, but generalization claims need more bases (more scraping). This guards against over-claiming in the writeup. Commit.

---

# Phase 4: prime-rl pin + pod readiness

## Task 4.1: Pin prime-rl to a commit >= 91182b7

**Files:**
- Modify: `pyproject.toml`

**Step 1:** Replace bare `"prime-rl"` with a pinned form. Confirm the exact post-#1392 commit/tag via `gh api repos/PrimeIntellect-ai/prime-rl/commits/main` or the PR merge SHA, then:
```toml
"prime-rl @ git+https://github.com/PrimeIntellect-ai/prime-rl@<sha-or-tag>",
```
**Step 2:** Validate toml parses; commit. (Cannot install locally - torch/vllm are pod-only - so this is verified on the pod at install time.)
```bash
git add pyproject.toml
git commit -m "chore(deps): pin prime-rl to post-#1392 commit (NCCL+LoRA fix)"
```

---

## Task 4.2: Pod docker setup - conditional systemd + preflight

**Files:**
- Modify: `scripts/setup_pod_docker.sh`

**Step 1:** Guard the daemon.json write + restart behind a systemd check:
```bash
if command -v systemctl >/dev/null && systemctl list-units >/dev/null 2>&1; then
  # write daemon.json + restart
else
  echo "no systemd (containerised pod) - using host-mounted docker socket as-is"
fi
```
**Step 2:** Add a preflight that FAILS LOUD if docker isn't usable:
```bash
docker run --rm hello-world >/dev/null 2>&1 || { echo "FATAL: docker not available on this pod (reward needs it)"; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "FATAL: buildx missing"; exit 1; }
```
**Step 3:** `bash -n` syntax check; commit.

---

## Task 4.3: Proxy allowlist - add HuggingFace

**Files:**
- Modify: `scripts/setup_pod_proxy.sh`

**Step 1:** Add to the tinyproxy filter: `^https?://(.*\.)?huggingface\.co` and the LFS CDN `^https?://cdn-lfs.*\.huggingface\.co` so model weights + dataset download aren't blocked. `bash -n`; commit.

---

## Task 4.4: Wire builder + cache_dir on the pod path

**Files:**
- Modify: `scripts/run_annotate.py` (and/or document for the reward path)

**Step 1:** `build_gate` accepts `builder` + `cache_dir` but nothing passes them. Add `--builder` and `--cache-dir` args to `run_annotate.py` (default None for laptop) so on the pod we pass the `dockermin` buildx builder + `/scratch/bkcache`. Document that the reward path (dockermin_reward -> build_gate) should set these via env on the pod, or accept default-builder (no cache) for the pilot. Commit.

---

# Phase 5: Eval honesty

## Task 5.1: Capture agent-loop cost even on non-zero exit

**Files:**
- Modify: `src/dockermin/eval/agent_loop.py`

**Step 1:** `run_agent_loop` parses `meta` only when `returncode == 0`. The budget cap trips a non-zero exit, so cost/turns silently read 0 - under-reporting the agent's true cost (the number that argues "RL is cheaper"). Parse the JSON `result` regardless of exit code (claude `-p --output-format json` still emits the result object on budget-exceed), guard with try/except json.JSONDecodeError. Commit.

---

## Task 5.2: Leaderboard - guard the baseline_size join + assert equal N (TDD)

**Files:**
- Modify: `scripts/leaderboard.py`
- Create: `tests/test_leaderboard.py`

**Step 1: Write failing tests** for: (a) a triple missing `baseline_size` is reported (not silently dropped); (b) a warning/assert fires if baselines saw different triple counts.

**Step 2: Implement:** when a triple lacks `baseline_size`, count it as a logged skip with a visible warning + a skipped-count column; after aggregation, assert all baselines have the same N (or print a loud MISMATCH banner). Commit.

---

# Verification (run after each phase + at the end)

```bash
.venv/bin/ruff check src tests scripts prime_env
.venv/bin/ruff format --check src tests scripts prime_env
.venv/bin/mypy src tests
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-fail-under=45
```
All must stay green. Push after each phase so CI (`quality.yml`) verifies.

---

# Risk register

| Risk | Mitigation |
|---|---|
| Hiding the test slows early GRPO convergence | Intended trade (functional equivalence > probe-passing). Flag in pilot review; watch reward curve. **Needs Vlad sign-off.** |
| async reward + to_thread fights prime-rl's executor pool | Semaphore N=6 <= executor size; raise via set_concurrency() on pod if threads queue |
| 11-base dataset too small for generalization | Documented (Task 3.4); pilot is a methodology test, not a generalization claim |
| Re-annotation needed after probe change (Task 1.3) | Re-run annotate on the pod before pushing the dataset; the probe change alters expected_substring values |
| prime-rl pin SHA wrong | Verify against the actual PR #1392 merge commit before pinning |

# What this plan does NOT do (YAGNI / out of scope)

- CURE-style co-evolved tester (too heavy for an 11-base learning project)
- Re-scraping for more bases (separate effort; documented as the real path to generalization)
- The differential train-vs-eval probe with a 2nd dataset column (deferred; the hidden-test fix #1 + hardened probes #1.3 cover the leak; revisit if pilot shows hacking)

# Done means

- Reward no longer leaks the test to the model; the echo/scratch cheats score below honest shrinks (regression-tested)
- Reward is async, semaphore-capped, failure-isolated; sync docker no longer blocks the event loop
- `push_to_hf` ships the 93-row variant set with a grouped train/test split; eval uses the disjoint test split
- prime-rl pinned; pod scripts survive a containerised pod + don't block HuggingFace; docker preflight fails loud
- agent-loop cost honest; leaderboard doesn't silently drop triples
- `make quality` green; CI green
