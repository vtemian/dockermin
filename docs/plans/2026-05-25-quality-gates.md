# Dockermin Quality Gates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring dockermin up to (and slightly past) the sisif code-quality bar: enforced ruff + mypy-strict gates, CI that fails on any gate, test coverage on the currently-untested glue, and a CLAUDE.md/ARCHITECTURE.md that encode the conventions so future agents don't re-accrue debt.

**Architecture:** Three enforcement layers, each independently blocking: (1) `pyproject.toml` ruff + mypy config = the machine-readable rules; (2) `.pre-commit-config.yaml` = local pre-commit blocking; (3) GitHub Actions `quality.yml` = CI blocking on PR/push (closes sisif's own hole where mypy never runs in CI). Plus two prose layers that humans/agents read: `CLAUDE.md` (rules) and `ARCHITECTURE.md` (map). The existing source is brought to zero ruff + zero mypy errors behavior-preservingly, then locked in by the gates.

**Tech Stack:** ruff 0.15.x, mypy (strict), pytest + pytest-cov, pre-commit, GitHub Actions, Python 3.11. No new runtime deps.

**Source of truth for the bar:** sisif at `/Users/whitemonk/projects/ai/sisif` (`pyproject.toml`, `CLAUDE.md`, `CODE_STYLE.md`, `Makefile`, `.pre-commit-config.yaml`).

---

## Starting state (measured 2026-05-25)

- ruff: **3 errors** remaining (down from 158; bulk already cleaned in uncommitted working tree)
- mypy strict: **22 errors**, concentrated in `src/dockermin/eval/baselines.py` (no-any-return, type-arg, call-overload)
- pytest: **35 passing** (25 original + 10 new `tests/test_ops.py`)
- Untested modules: `eval/baselines.py`, `eval/agent_loop.py`, `dataset/scrape.py`, all `scripts/*.py`
- Missing: `CLAUDE.md`, `ARCHITECTURE.md`, CI quality job, mypy in pre-commit, coverage measurement

**Important:** the working tree already contains the bulk lint/type cleanup (uncommitted, behavior-preserving, tests green). Phase 1 verifies and commits it rather than redoing it. If any task finds the cleanup wrong, fix per this plan's standards.

---

## Decisions baked into this plan

1. **Stricter than sisif where sisif waived "for Django":** ENABLE `EM101`/`EM102` (raw/f-string in exceptions) and keep `PLR0913` with an explicit `max-args = 6` instead of globally ignoring it. Drop sisif's gradual-debt security waivers (`S113`, `S308`, `S324`, `S608`, `B904`) - we have no such debt.
2. **Close sisif's CI hole:** mypy + ruff + tests ALL run in CI, not just pre-commit.
3. **YAGNI on scaffolding** (per structure analysis): root CLAUDE.md only (no per-module), single `pyproject.toml` deps (no pip-tools trifecta), one `docs/` tree (no `thoughts/` ceremony).
4. **scripts/ are CLI entrypoints:** keep the per-file-ignore for T20/S603/S607/ANN/complexity. They are glue, not library.
5. **Coverage floor starts at a realistic number** (measure first), ratchets up - not 100% on day one.

---

# Phase 1: Finish lint + type cleanup, lock it in

## Task 1.1: Resolve the 3 remaining ruff errors

**Files:**
- Modify: whatever `ruff check` reports (likely `src/dockermin/eval/baselines.py`, `tests/test_annotate.py`)

**Step 1: See the exact 3**
```bash
.venv/bin/ruff check src tests scripts prime_env
```

**Step 2: Fix each behavior-preservingly.** Prefer a real fix over `# noqa`. Only use `# noqa: <CODE>` with a trailing reason comment when the rule genuinely should not apply (e.g. a deliberate safety-net `except Exception` in an eval baseline).

**Step 3: Verify clean**
```bash
.venv/bin/ruff check src tests scripts prime_env
```
Expected: `All checks passed!`

**Step 4: Verify tests still green**
```bash
.venv/bin/pytest tests/ -q
```
Expected: `35 passed`

**Step 5: Commit**
```bash
git add -u && git add src/dockermin/ops.py tests/test_ops.py
git commit -m "refactor: ruff-clean all modules (158 -> 0), extract ops.py (dedup acquire_lock + jsonl)"
```

---

## Task 1.2: Resolve the 22 mypy errors in baselines.py

**Files:**
- Modify: `src/dockermin/eval/baselines.py`

**Context:** errors are `no-any-return` (untyped SDK calls), `type-arg` (bare `dict`/`re.Match`), `call-overload` (the `_openai_chat`/`_anthropic_messages` wrappers pass `**kwargs: object` which the strict SDK overloads reject).

**Step 1: See them**
```bash
.venv/bin/mypy src/dockermin/eval/baselines.py
```

**Step 2: Fix by category:**
- `type-arg`: `dict` -> `dict[str, Any]`, `re.Match` -> `re.Match[str]`.
- `no-any-return`: where a function returns the result of an untyped SDK call, either annotate the local (`text: str = resp.choices[0].message.content or ""`) or `cast(str, ...)`.
- `call-overload` on the SDK wrappers: replace `**kwargs: object` with explicit named params (`model: str, messages: list[...], temperature: float, max_tokens: int`). This IS a signature change - it is correct and intended; the wrappers currently lie about their interface. Keep behavior identical (same args forwarded).

**Step 3: Verify mypy clean on the file**
```bash
.venv/bin/mypy src/dockermin/eval/baselines.py
```
Expected: no errors in this file.

**Step 4: Full mypy + tests**
```bash
.venv/bin/mypy src && .venv/bin/pytest tests/ -q
```
Expected: `Success: no issues found` (or only documented overrides), `35 passed`.

**Step 5: Commit**
```bash
git add src/dockermin/eval/baselines.py
git commit -m "fix(eval): mypy-strict clean baselines.py (typed SDK wrappers, dict/Match args)"
```

---

## Task 1.3: Tighten pyproject ruff to be stricter than sisif where justified

**Files:**
- Modify: `pyproject.toml` (`[tool.ruff.lint]`)

**Step 1: Remove `EM101`, `EM102` from the `ignore` list** (we want raw-string-in-exception flagged). Run `ruff check` - fix any new EM violations by extracting the message to a local variable first:
```python
msg = f"build timeout after {timeout_s}s"
raise BuildError(msg)
```

**Step 2: Replace the global `PLR0913` handling with an explicit arg cap.** In `[tool.ruff.lint.pylint]` add `max-args = 6`. Remove any blanket `PLR0913` ignore. For the legitimately-wide signatures (`annotate_one`, `compute_score`), keep the per-line `# noqa: PLR0913` with reason (callers + tests fix the shape).

**Step 3: Confirm no gradual-debt security waivers crept in.** The `ignore` list should contain ONLY: `S101`, `S311`, `SIM105`, `ANN204`, `TRY003`, `RET504`. Nothing else.

**Step 4: Run the gate**
```bash
.venv/bin/ruff check src tests scripts prime_env && .venv/bin/pytest tests/ -q
```
Expected: clean + 35 passed.

**Step 5: Commit**
```bash
git add pyproject.toml src/
git commit -m "chore(lint): enable EM rules, explicit max-args=6, drop debt waivers"
```

---

# Phase 2: CI enforcement (close sisif's hole)

## Task 2.1: Add mypy to pre-commit

**Files:**
- Modify: `.pre-commit-config.yaml`

**Step 1:** Confirm the mypy hook block exists (it was added with the gate setup). Verify it points at `src` and `tests` and uses `--config-file=pyproject.toml`.

**Step 2: Install and run the hooks once**
```bash
.venv/bin/pip install -q pre-commit
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```
Expected: ruff, ruff-format, mypy all pass (or auto-fix then pass on re-run).

**Step 3: Commit** (include any formatting the hook applied)
```bash
git add -A && git commit -m "chore: pre-commit hooks (ruff, ruff-format, mypy) installed and green"
```

---

## Task 2.2: GitHub Actions quality job

**Files:**
- Create: `.github/workflows/quality.yml`

**Step 1: Write the workflow** (runs the FULL gate on push + PR - this is what sisif lacks for mypy):
```yaml
name: quality
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
          restore-keys: ${{ runner.os }}-pip-
      - name: Install (light - no torch/vllm)
        run: |
          python -m pip install --upgrade pip
          pip install ruff mypy pytest dockerfile docker tqdm tenacity huggingface_hub anthropic openai datasets
          pip install --no-deps -e .
      - name: Ruff lint
        run: ruff check src tests scripts prime_env
      - name: Ruff format check
        run: ruff format --check src tests scripts prime_env
      - name: Mypy strict
        run: mypy src tests
      - name: Pytest (skip docker-gated)
        run: pytest tests/ -v -m "not docker"
```

**Step 2: Validate the YAML**
```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/quality.yml')); print('valid')"
```

**Step 3:** Delete the old `.github/workflows/ci.yml` if it duplicates this (it ran pytest only with `-k "not docker"`). Fold its intent into `quality.yml`.

**Step 4: Commit + push so CI actually runs**
```bash
git add .github/workflows/ && git commit -m "ci: full quality gate (ruff + format-check + mypy + pytest) on push/PR"
git push
```

**Step 5: Verify the run**
```bash
gh run list --limit 1
gh run watch  # or check the actions tab
```
Expected: the `quality` job passes green. If mypy or ruff fails in CI but passed locally, reconcile (usually a dep or version mismatch).

---

# Phase 3: Test coverage on the untested glue

## Task 3.1: Add pytest-cov and measure the baseline

**Files:**
- Modify: `pyproject.toml` (dev extras + `[tool.coverage]`)

**Step 1: Add `pytest-cov` to `[project.optional-dependencies] dev` and a coverage config:**
```toml
[tool.coverage.run]
source = ["src/dockermin"]
omit = ["*/tests/*"]

[tool.coverage.report]
show_missing = true
skip_covered = false
```

**Step 2: Measure**
```bash
.venv/bin/pip install -q pytest-cov
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-report=term-missing
```

**Step 3: Record the baseline %** in the commit message. Set the coverage floor in CI to the measured baseline minus a small margin (ratchet, don't aspire). Add `--cov-fail-under=<baseline>` to the `quality.yml` pytest step.

**Step 4: Commit**
```bash
git add pyproject.toml .github/workflows/quality.yml
git commit -m "test: add coverage measurement + CI floor at <baseline>%"
```

---

## Task 3.2: TDD tests for `dataset/scrape.py` pure helpers

**Why:** scrape.py was the second-most-violating file and has zero tests. The pure parsing helpers (`_parse_official_manifest`/`_split_manifest_blocks`, `_infer_ecosystem`, `_strip_github_repo`, the candidate-builders) are deterministic and trivially testable without network.

**Files:**
- Create: `tests/test_scrape.py`

**Step 1: Write failing tests for the pure helpers.** Example:
```python
"""Tests for dataset.scrape pure parsing helpers (no network)."""
from dockermin.dataset.scrape import _infer_ecosystem, _parse_official_manifest

def test_infer_ecosystem_known_base():
    assert _infer_ecosystem("python") == "python"
    assert _infer_ecosystem("eclipse-temurin") == "java"

def test_infer_ecosystem_unknown_defaults():
    assert _infer_ecosystem("totally-made-up") == "unknown"

def test_parse_official_manifest_extracts_git_coords():
    manifest = "Tags: 3.12\nGitRepo: https://github.com/x/y.git\nGitCommit: abc123\nDirectory: 3.12\n"
    stanzas = _parse_official_manifest(manifest)
    assert stanzas
    assert stanzas[0]["GitCommit"] == "abc123"
```
(Read the actual current helper names/signatures first - the fix-agent renamed/extracted several.)

**Step 2: Run, expect fail (import or assertion)**
```bash
.venv/bin/pytest tests/test_scrape.py -v
```

**Step 3:** These are tests for EXISTING behavior, so they should pass once imports resolve. If a test fails, the test encodes wrong expectations - fix the test to match real (correct) behavior, OR if it reveals a real bug, fix the code (TDD: write the test for the bug first).

**Step 4: Verify + coverage delta**
```bash
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-report=term-missing | grep scrape
```

**Step 5: Commit**
```bash
git add tests/test_scrape.py
git commit -m "test(scrape): cover pure parsing helpers (ecosystem, manifest)"
```

---

## Task 3.3: TDD tests for `eval/baselines.py` deterministic baselines

**Why:** the mechanical baselines (`baseline_hadolint`, `baseline_manual_best_practice`, `_collapse_consecutive_runs`, `_swap_base_image`, `_fix_dl3009`/`_fix_dl3060`) are pure string transforms - testable without docker/models.

**Files:**
- Create: `tests/test_baselines.py`

**Step 1: Write failing tests for the pure transforms:**
```python
"""Tests for eval.baselines pure Dockerfile transforms."""
from dockermin.eval.baselines import _collapse_consecutive_runs, _swap_base_image

def test_collapse_consecutive_runs_merges_adjacent():
    df = "FROM x\nRUN a\nRUN b\nCMD [\"x\"]\n"
    out = _collapse_consecutive_runs(df)
    assert out.count("RUN") == 1
    assert "a" in out and "b" in out

def test_swap_base_image_known_mapping():
    df = "FROM python:3.12\nRUN pip install x\n"
    out = _swap_base_image(df)
    assert "python:3.12-slim" in out
```
(Read the actual function names/signatures first - they were extracted during cleanup.)

**Step 2-4:** Run (expect pass for existing behavior; fix test if expectations wrong, fix code if real bug found), verify coverage delta.

**Step 5: Commit**
```bash
git add tests/test_baselines.py
git commit -m "test(eval): cover mechanical baseline transforms (hadolint, manual, run-collapse)"
```

---

## Task 3.4: Raise the CI coverage floor

**Files:**
- Modify: `.github/workflows/quality.yml`

**Step 1: Re-measure after 3.2 + 3.3**
```bash
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-report=term
```

**Step 2: Bump `--cov-fail-under` to the new (higher) baseline.** Commit + push, confirm CI still green.

---

# Phase 4: Documentation guardrails

## Task 4.1: Write CLAUDE.md (the rules)

**Files:**
- Create: `CLAUDE.md` (repo root)

**Step 1: Write it.** Trim sisif's to the language-agnostic core. Sections:
1. **Project overview** - one paragraph (GRPO Dockerfile minimizer).
2. **Commands** - `make quality`, `make test`, the script entrypoints.
3. **Pre-commit gate (MANDATORY)** - "Run `make quality` before every commit. Do not commit if ruff, mypy, or pytest fail. Never commit to main; branch per change; conventional commits."
4. **Code Quality Standards** with BAD/GOOD examples, copied-and-trimmed from sisif:
   - No Nesting (max one indent level; guard clauses; extract helpers)
   - Names Are Contracts (`fetch_*`->list, `find_*`->Optional, `get_*`->raises; no generic Manager/Helper/Data)
   - Fail Fast (specific exception tuples, never bare `except Exception` except documented safety nets; `logger.exception` in handlers)
   - No Narration (comment WHY not WHAT)
   - Plain functions over classes (classes only for dataclasses + exceptions)
   - Type hints required on all signatures (mypy strict enforces)
   - `logger = logging.getLogger(__name__)`, never `print` in library code (scripts excepted)
5. **The machine gates** - point at pyproject ruff config + complexity limits (max-complexity 5, branches 4, returns 4, statements 15, args 6) so the prose and the linter agree.

DROP: all web/UI rules (emojis, inline styles, HTMX, Tailwind, i18n, Django structure).

**Step 2: Commit**
```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md encoding quality rules (no-nesting, names-are-contracts, fail-fast)"
```

---

## Task 4.2: Write ARCHITECTURE.md

**Files:**
- Create: `ARCHITECTURE.md`

**Step 1: Write it.** Per the structure analysis, the high-value, ~1-hour version:
1. **Overview** - one paragraph.
2. **Annotated directory tree** - every dir in `src/dockermin/`, `scripts/`, `tests/`, `configs/`, `prime_env/` with a one-line purpose.
3. **Module-responsibility table** - `module -> what it owns -> key public functions`.
4. **Two numbered data-flow walkthroughs:**
   - Dataset build: scrape -> annotate (parse/build/test gates) -> synthetic variants -> HF push
   - Train + eval: prime-rl reward (gates -> compute_score) -> LoRA -> eval baselines -> leaderboard
5. **Key decisions table** - prime-rl over raw verifiers, buildx subprocess over docker SDK, etc. (pull from `docs/research_findings_2026-05-22.md`).

**Step 2: Commit**
```bash
git add ARCHITECTURE.md
git commit -m "docs: ARCHITECTURE.md (annotated tree, module table, train/eval + dataset flows)"
```

---

# Phase 5: Convention-compliance audit (the rules ruff can't check)

## Task 5.1: Audit naming contracts + no-nesting across src/

**Files:**
- Modify: any `src/dockermin/**/*.py` violating the prose rules

**Step 1: Audit each public function name against the contract:**
- `fetch_*` must return a collection (never None)
- `find_*` may return None
- `get_*` raises if absent
- No generic `*_helper`, `*_manager`, `process_*`, `handle_*` without a domain noun

Grep for offenders:
```bash
grep -rnE "def (get|fetch|find)_" src/dockermin/
```
Rename mismatches (e.g. a `get_` that returns None should be `find_`). Update callers + tests. Keep tests green after each rename.

**Step 2: Audit nesting depth.** ruff's complexity rules catch most, but eyeball for functions with >1 indent level inside the body that slipped under the complexity cap. Flatten with guard clauses.

**Step 3: After each change, run the gate**
```bash
make quality
```

**Step 4: Commit per logical group**
```bash
git commit -m "refactor: align names with fetch/find/get contracts"
```

---

## Task 5.2: Final full-gate verification

**Step 1: Run the complete gate exactly as CI will:**
```bash
.venv/bin/ruff check src tests scripts prime_env
.venv/bin/ruff format --check src tests scripts prime_env
.venv/bin/mypy src tests
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-report=term
```
Expected: all clean, coverage >= floor.

**Step 2: Confirm CI is green on the latest push**
```bash
gh run list --limit 1
```

**Step 3: Update the journal** (`docs/journal.md`) with the before/after: 158 ruff -> 0, mypy strict 0, coverage X%, gates enforced in pre-commit + CI.

**Step 4: Final commit**
```bash
git add docs/journal.md
git commit -m "docs: journal - quality gates adopted, 158 violations -> 0, mypy strict + CI enforced"
git push
```

---

# Risk register

| Risk | Mitigation |
|---|---|
| mypy strict on ML SDK calls is noisy | `ignore_missing_imports` overrides for torch/vllm/etc already in pyproject; use `cast`/local annotations at SDK boundaries, not blanket `type: ignore` |
| Narrowing `except Exception` changes behavior | Only narrow where the raisable set is known; keep documented `# noqa: BLE001` safety nets in eval baselines (one bad Dockerfile must not crash the eval loop) |
| Coverage floor too aggressive | Start at measured baseline, ratchet up; never block on aspirational numbers |
| CI install pulls torch (huge/slow) | Light install in CI (no torch/vllm); GPU-only code paths are import-guarded and not exercised by `-m "not docker"` tests |
| Convention renames break callers | Rename + update callers + run `make quality` per change; small commits |

# What we are NOT doing (YAGNI)

- Per-module CLAUDE.md files (15 modules don't justify it)
- pip-tools base/dev/prod `.in`/`.txt` trifecta (single pyproject + dev extras)
- `thoughts/shared/` design-vs-plan ceremony (one `docs/` tree + journal)
- Django-specific gates (django-stubs, lint-templates, migrations handling)
- 100% coverage mandate (ratchet from baseline)

# Done means

- `ruff check` + `ruff format --check`: clean on src/tests/scripts/prime_env
- `mypy src tests`: strict, zero errors
- `pytest -m "not docker"`: green, coverage >= floor
- pre-commit installed and green; `quality.yml` green on push/PR
- `CLAUDE.md` + `ARCHITECTURE.md` present and accurate
- Names obey fetch/find/get contracts; no bare excepts except documented safety nets
