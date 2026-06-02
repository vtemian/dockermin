# GRPO v3: Manifest Pre-Build Gate + Dataset Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut dockermin's 43 % holdout failure rate without sacrificing v2's 75.3 % mean-reduction win.

**Architecture:** Two parallel attacks. **Fix A** — add a `manifest_gate` rung between `parse_gate` and `build_gate` in the reward ladder so a hallucinated `FROM` tag scores `-0.05` instead of slipping into the `BUILD_FAIL` rung. Subagent failure-classification proved 9 / 16 v2 failures (56 %) are pure hallucinated tags. **Fix B** — expand training set from ~108 → 500-1000 examples with broader base-image diversity (rust, go, dotnet, distroless, chainguard, ruby), freeze the v0 holdout via an explicit id-list, and re-train with `max_steps=400-500` per prime-rl precedent. Eval matrix re-runs against v2 step_250 @ T=0.2 as the comparison baseline.

**Tech Stack:** Python 3.11+ (`uv` venv), `prime-rl @ main`, `verifiers ≥ 0.1.5`, Qwen 2.5 Coder 7B Instruct + LoRA r=32 α=64, `docker buildx`, `docker manifest inspect` (subprocess), HuggingFace Hub for adapter/eval persistence.

---

## Subagent investigation summary (for executor context)

The four parallel investigations established:

1. **Failure taxonomy (Agent 3):** of v2's 16 holdout failures, **9 are pure hallucinated FROM tags** (the manifest-addressable set: `eclipse-temurin:25-jdk-slim`, `python:3.12-bookworm-slim`, `buildpack-deps:trixie-slim`); 2 are syntax errors inside `RUN` chains; 2 are removed-too-much; 1 stale apt repo; 1 transcription error; 1 output-format. Fix A directly addresses 56 % of failures. Fix B (dataset diversity) is needed for the remaining 44 %.

2. **Reward gate code-path (Agent 1):** the new rung slots into `gates.py:120-128`'s tuple-loop as the third entry (`(not manifest_ok, MANIFEST_FAIL_SCORE)`). `MANIFEST_FAIL_SCORE = -0.05` keeps the invariant `parse_fail (-0.1) < too_few (-0.2)... < manifest_fail (-0.05) < build_fail [0..0.30] < test_fail [0.35..0.45] < pass [0.5..1.0]`. Implementation: subprocess `docker manifest inspect` (subprocess is the project's existing precedent at `annotate.py:92-111` for `docker buildx`), with `docker image inspect` short-circuit for locally-cached tags. Multi-stage policy: check all unique FROM images (skip `scratch` and stage aliases).

3. **Dataset gap (Agent 2):** train has 19 unique source URLs / 7 base-image names; test has 9 source URLs / 6 base-image names. The bottleneck is **tag-variant coverage** — train has zero `python:3.9*` rows but test has 12; zero `eclipse-temurin:*-noble` rows but test has 6. Missing ecosystems entirely: rust, go, dotnet, distroless, chainguard. Scraper extension points are at `scrape.py:570-576` (code-search queries) and `scripts/run_scrape.py:18-31` (fetcher list). Variant generator at `synthetic_variants.py` works unchanged.

4. **Hyperparams (Agent 4):** v2's `max_steps=250` is below prime-rl's 500-step precedent for hard tasks (math, search). Recommended: 400-500 steps, eval-every-50, `batch_size=32` (up from 16; prime-rl ref runs at 256-512), **keep `group_size=16`** (no precedent for 32), **do not touch KL** (no evidence). Manifest gate is a single-scalar ladder rung — per-group advantage normalization stays valid; no second normalization needed.

---

## Quality gate (mandatory before every commit)

Per `CLAUDE.md` § "Pre-Commit Gate":

```bash
make quality   # ruff + mypy strict + pytest -m "not docker"
```

A green local gate is the contract for green CI. Branch per task: `git checkout -b feat/v3-<scope>`.

---

## Phase 1 — Fix A: manifest pre-build gate (TDD)

### Task 1: `MANIFEST_FAIL_SCORE` constant + `manifest_ok` parameter in `compute_score`

**Files:**
- Modify: `src/dockermin/reward/gates.py:7-24` (constants section) and `:100-128` (`compute_score`)
- Test: `tests/test_gates.py`

**Step 1: Write the failing test**

Append to `tests/test_gates.py`:

```python
def test_compute_score_manifest_fail_returns_minus_005() -> None:
    """Hallucinated FROM tag (manifest_ok=False) scores MANIFEST_FAIL_SCORE,
    strictly between PARSE_FAIL and BUILD_FAIL with no command-count credit."""
    score = compute_score(
        parse_ok=True,
        manifest_ok=False,
        build_ok=False,
        test_ok=False,
        command_count=10,
        baseline_size=1_000_000,
        new_size=500_000,
        dockerfile_text="FROM hallucinated:tag\nRUN echo hi",
        baseline_command_count=8,
    )
    assert score == MANIFEST_FAIL_SCORE
    assert MANIFEST_FAIL_SCORE == -0.05  # contract — bump on purpose only


def test_compute_score_manifest_invariant_holds() -> None:
    """max(manifest_fail) < min(build_fail with no credit) < min(test_fail) < pass floor."""
    assert MANIFEST_FAIL_SCORE < BUILD_FAIL_SCORE
    assert MANIFEST_FAIL_SCORE > PARSE_FAIL_SCORE  # manifest_fail is "more progress" than parse_fail
    assert BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX < TEST_FAIL_SCORE
    assert TEST_FAIL_SCORE + TEST_FAIL_CMD_CREDIT_MAX < 0.5  # pass floor
```

Imports (add at top of `tests/test_gates.py` if missing):

```python
from dockermin.reward.gates import (
    BUILD_FAIL_CMD_CREDIT_MAX,
    BUILD_FAIL_SCORE,
    MANIFEST_FAIL_SCORE,  # NEW
    PARSE_FAIL_SCORE,
    TEST_FAIL_CMD_CREDIT_MAX,
    TEST_FAIL_SCORE,
    compute_score,
)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_gates.py::test_compute_score_manifest_fail_returns_minus_005 -v
```

Expected: `ImportError: cannot import name 'MANIFEST_FAIL_SCORE'`.

**Step 3: Write minimal implementation**

In `src/dockermin/reward/gates.py`:

1. After `BUILD_FAIL_SCORE = 0.0` at `:11`, add the new constant:

```python
# Hallucinated FROM tag — Dockerfile parses, command_count >= MIN_COMMANDS, but
# at least one FROM references an image that does not resolve in the registry
# (and isn't cached locally). Scored strictly worse than build_fail's floor so
# a hallucinated rollout never out-scores a real-tag build attempt; strictly
# better than parse_fail so the model gets credit for at least emitting a
# parseable Dockerfile. No command-count credit — padding cannot rescue a
# hallucinated base.
MANIFEST_FAIL_SCORE = -0.05
```

2. Update the rung invariant comment at `:19`:

```python
# Invariant: parse_fail (-0.1) < too_few (-0.2) < manifest_fail (-0.05)
# < max(build_fail) (0.30) < min(test_fail) (0.35) < pass floor (0.5).
# Verified by test_compute_score_manifest_invariant_holds in tests/test_gates.py.
```

3. Add `manifest_ok: bool` to `compute_score`'s keyword-only parameters at `:100-110`:

```python
def compute_score(  # noqa: PLR0913
    *,
    parse_ok: bool,
    manifest_ok: bool,
    build_ok: bool,
    test_ok: bool,
    command_count: int,
    baseline_size: int,
    new_size: int,
    dockerfile_text: str,
    baseline_command_count: int | None = None,
) -> float:
```

4. Insert the new rung in the tuple at `:120-128`:

```python
gate_failures = (
    (not parse_ok, PARSE_FAIL_SCORE),
    (command_count < MIN_COMMANDS, TOO_FEW_COMMANDS_SCORE),
    (not manifest_ok, MANIFEST_FAIL_SCORE),
    (not build_ok, BUILD_FAIL_SCORE + bf_credit),
    (not test_ok, TEST_FAIL_SCORE + tf_credit),
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_gates.py -v -k "manifest" --no-header
```

Expected: both new tests PASS.

**Step 5: Commit**

```bash
git checkout -b feat/v3-manifest-fail-score
git add src/dockermin/reward/gates.py tests/test_gates.py
git commit -m "feat(reward): add MANIFEST_FAIL_SCORE rung between parse and build gates"
```

---

### Task 2: Update all existing `compute_score` callsites to pass `manifest_ok`

**Files:**
- Modify: `src/dockermin/reward/dockermin_reward.py` — every `compute_score(...)` call (4 sites, at approximately `:93-102`, `:107-116`, `:119-128`, `:129-138`)
- Test: existing reward tests in `tests/test_reward.py` must continue to pass with a new `manifest_ok=True` parameter threaded through

**Step 1: Write the failing test**

Add to `tests/test_reward.py`:

```python
@pytest.mark.asyncio
async def test_reward_passes_manifest_ok_true_when_parse_ok() -> None:
    """When parse_ok is True and the manifest_gate has not yet been wired in,
    the orchestrator must explicitly pass manifest_ok=True to compute_score
    so the score path is well-defined."""
    # Sentinel: the call must include manifest_ok or it will TypeError.
    # This is a smoke test for the callsite update; full manifest semantics
    # are tested in Task 4.
    from dockermin.reward.dockermin_reward import dockermin_reward
    completion = "```dockerfile\nFROM python:3.12-slim\nRUN echo hi\nCMD [\"python\", \"-c\", \"print('hi')\"]\n```"
    info = {
        "baseline_size": 50_000_000,
        "test_cmd": ["python", "-c", "print('hi')"],
        "expected_substring": "hi",
        "baseline_command_count": 3,
    }
    # Should not raise — verifies callsites pass manifest_ok
    reward = await dockermin_reward(completion=completion, info=info)
    assert isinstance(reward, float)
```

**Step 2: Run test**

```bash
pytest tests/test_reward.py::test_reward_passes_manifest_ok_true_when_parse_ok -v
```

Expected: `TypeError: compute_score() missing 1 required keyword-only argument: 'manifest_ok'`.

**Step 3: Implement**

In `src/dockermin/reward/dockermin_reward.py`, at every `compute_score(...)` callsite, add `manifest_ok=True` as a placeholder (the real manifest_gate call is added in Task 4). Example for the parse-fail branch:

```python
return compute_score(
    parse_ok=False,
    manifest_ok=True,  # gate not yet wired; placeholder, replaced in Task 4
    build_ok=False,
    test_ok=False,
    command_count=0,
    baseline_size=info["baseline_size"],
    new_size=0,
    dockerfile_text=new_df,
    baseline_command_count=info.get("baseline_command_count"),
)
```

Repeat for the too_few branch, build_fail branch, test_fail branch, and pass branch (all four sites in `dockermin_reward`). The `gate_too_few` and the docker-hiccup safety-net path also get `manifest_ok=True`.

**Step 4: Run test**

```bash
pytest tests/test_reward.py -v -m "not docker"
```

Expected: all pass, including the new one (callsite no longer raises TypeError).

**Step 5: Commit**

```bash
git add src/dockermin/reward/dockermin_reward.py tests/test_reward.py
git commit -m "feat(reward): thread manifest_ok placeholder through compute_score callsites"
```

---

### Task 3: `find_from_images` helper + `ManifestResult` dataclass in `annotate.py`

**Files:**
- Modify: `src/dockermin/dataset/annotate.py` — add helper near `parse_gate` (`:63-71`) and the new dataclass near `BuildResult` (`:32-38`)
- Test: `tests/test_annotate.py`

**Step 1: Write the failing test**

Append to `tests/test_annotate.py`:

```python
def test_find_from_images_single_from() -> None:
    df = "FROM python:3.12-slim\nRUN echo hi\nCMD [\"python\"]"
    assert find_from_images(df) == ["python:3.12-slim"]


def test_find_from_images_multistage_deduped() -> None:
    """Multi-stage: returns each unique registry image once; stage aliases ignored."""
    df = (
        "FROM golang:1.22 AS builder\n"
        "RUN go build -o app\n"
        "FROM gcr.io/distroless/base\n"
        "COPY --from=builder /app /app\n"
        "FROM golang:1.22 AS tester  # duplicate base, should dedup\n"
        "CMD [\"/app\"]\n"
    )
    assert sorted(find_from_images(df)) == ["gcr.io/distroless/base", "golang:1.22"]


def test_find_from_images_skips_scratch_and_aliases() -> None:
    df = (
        "FROM scratch AS base\n"
        "COPY app /app\n"
        "FROM base AS final\n"  # 'base' is a stage alias, not a registry image
        "CMD [\"/app\"]\n"
    )
    assert find_from_images(df) == []


def test_find_from_images_malformed_returns_empty() -> None:
    """Unparseable Dockerfile returns empty list rather than raising."""
    assert find_from_images("this is not a dockerfile") == []
```

Import update at top of file:

```python
from dockermin.dataset.annotate import (
    build_gate,
    find_from_images,  # NEW
    parse_gate,
    run_test_gate,
    ManifestResult,  # NEW
)
```

**Step 2: Run test**

```bash
pytest tests/test_annotate.py -v -k "find_from_images"
```

Expected: `ImportError: cannot import name 'find_from_images'`.

**Step 3: Implement**

In `src/dockermin/dataset/annotate.py`:

1. Add `ManifestResult` near `BuildResult` at `:32`:

```python
@dataclass(frozen=True)
class ManifestResult:
    """Outcome of the manifest pre-build gate.

    ``ok`` is False if any FROM image fails both the local-image-inspect and
    the registry manifest-inspect probes. ``missing`` lists the offending image
    refs (deduped, preserves first-seen order) for logging/observability.
    """

    ok: bool
    missing: tuple[str, ...] = ()
    error: str = ""
```

2. Add `find_from_images` near `parse_gate` at `:63`:

```python
def find_from_images(df_text: str) -> list[str]:
    """Return the unique FROM image refs in dockerfile order, skipping
    ``scratch`` and stage aliases declared via ``AS``.

    Returns empty list on parse failure; downstream gates handle the failure
    via parse_gate's own ok flag.
    """
    try:
        commands = dockerfile.parse_string(df_text)
    except dockerfile.GoParseError:
        return []
    seen: set[str] = set()
    out: list[str] = []
    declared_aliases: set[str] = set()
    for cmd in commands:
        if cmd.cmd.upper() != "FROM":
            continue
        # cmd.value is a tuple; first element is the base ref, then optional 'AS' alias
        if not cmd.value:
            continue
        base = cmd.value[0]
        # Capture any 'AS <name>' alias to skip it on subsequent FROM lines
        if "AS" in (s.upper() for s in cmd.value[1:]):
            # value looks like ('python:3.12', 'AS', 'builder')
            for i, tok in enumerate(cmd.value[1:]):
                if tok.upper() == "AS" and i + 2 < len(cmd.value):
                    declared_aliases.add(cmd.value[i + 2])
        if base.lower() == "scratch":
            continue
        if base in declared_aliases:  # internal stage reference
            continue
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out
```

**Step 4: Run test**

```bash
pytest tests/test_annotate.py -v -k "find_from_images"
```

Expected: all four tests PASS.

**Step 5: Commit**

```bash
git add src/dockermin/dataset/annotate.py tests/test_annotate.py
git commit -m "feat(annotate): add find_from_images + ManifestResult dataclass"
```

---

### Task 4: `manifest_gate` function + orchestrator wiring

**Files:**
- Modify: `src/dockermin/dataset/annotate.py` — add `manifest_gate` near `build_gate` at `:114`
- Modify: `src/dockermin/reward/dockermin_reward.py` — call `manifest_gate` after `parse_gate`, replace all four `manifest_ok=True` placeholders with the actual result
- Test: docker-gated tests in `tests/test_annotate.py` and `tests/test_reward.py`

**Step 1: Write the failing tests (docker-gated)**

Append to `tests/test_annotate.py`:

```python
@pytest.mark.docker
def test_manifest_gate_real_tag_passes() -> None:
    """python:3.12-slim resolves in Docker Hub — manifest_gate returns ok=True."""
    result = manifest_gate("FROM python:3.12-slim\nRUN echo hi\nCMD [\"python\"]")
    assert result.ok is True
    assert result.missing == ()


@pytest.mark.docker
def test_manifest_gate_hallucinated_tag_fails() -> None:
    """eclipse-temurin:25-jdk-resolute does not exist; manifest_gate returns ok=False."""
    result = manifest_gate("FROM eclipse-temurin:25-jdk-resolute\nRUN echo hi\nCMD [\"java\"]")
    assert result.ok is False
    assert "eclipse-temurin:25-jdk-resolute" in result.missing


@pytest.mark.docker
def test_manifest_gate_multistage_one_hallucinated_fails() -> None:
    df = (
        "FROM python:3.12-slim AS builder\n"
        "FROM does-not-exist:fake-tag\n"
        "RUN echo hi\n"
        "CMD [\"python\"]\n"
    )
    result = manifest_gate(df)
    assert result.ok is False
    assert "does-not-exist:fake-tag" in result.missing
```

**Step 2: Run tests**

```bash
pytest tests/test_annotate.py -v -k "manifest_gate" -m docker
```

Expected: `ImportError: cannot import name 'manifest_gate'`.

**Step 3: Implement**

In `src/dockermin/dataset/annotate.py`, add `manifest_gate` near `build_gate`:

```python
def _image_exists_locally(image_ref: str, timeout_s: int = 5) -> bool:
    """Cheap probe: is this image already present in the local Docker image store?
    If yes, ``docker build`` will succeed regardless of registry state.
    """
    proc = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", image_ref],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode == 0


def _manifest_exists_in_registry(image_ref: str, timeout_s: int = 15) -> bool:
    """Probe the registry for the manifest of ``image_ref``. Returns True if the
    tag resolves (returncode 0), False on a clean "not found", and True on any
    timeout or transport failure (defensive: we do not penalize network blips).
    """
    try:
        proc = subprocess.run(  # noqa: S603
            ["docker", "manifest", "inspect", image_ref],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Pass-on-timeout — same defensive posture as the BLE001 safety net in
        # dockermin_reward.py:118. A network blip must not penalize the model.
        return True
    return proc.returncode == 0


def manifest_gate(df_text: str, timeout_s: int = 15) -> ManifestResult:
    """Check every unique FROM image resolves either locally or in the registry.

    Probe order per image: (1) ``docker image inspect`` for local cache hit;
    (2) ``docker manifest inspect`` for registry resolution. Pass if either
    succeeds. Skip ``FROM scratch`` and stage aliases (handled by
    ``find_from_images``).

    Returns ``ManifestResult.ok=False`` with the offending image refs in
    ``missing`` if any FROM cannot be resolved.
    """
    images = find_from_images(df_text)
    if not images:
        # No registry images to probe — multi-stage scratch-only, or parse
        # failure already handled by parse_gate. Treat as pass; downstream
        # gates will catch real problems.
        return ManifestResult(ok=True)
    missing: list[str] = []
    for image in images:
        if _image_exists_locally(image):
            continue
        if _manifest_exists_in_registry(image, timeout_s=timeout_s):
            continue
        missing.append(image)
    if missing:
        return ManifestResult(
            ok=False,
            missing=tuple(missing),
            error=f"manifest not found: {', '.join(missing)}",
        )
    return ManifestResult(ok=True)
```

In `src/dockermin/reward/dockermin_reward.py`, replace the placeholder threading:

```python
# After parse_gate succeeds (around dockermin_reward.py:91):
p = parse_gate(new_df)
if not p.ok:
    return compute_score(parse_ok=False, manifest_ok=True, ...)  # parse already failed
if p.command_count < MIN_COMMANDS:
    return compute_score(parse_ok=True, manifest_ok=True, ...)  # too_few short-circuits

# NEW: manifest gate runs inline (no semaphore — ~100ms subprocess call)
m = manifest_gate(new_df, timeout_s=15)
if not m.ok:
    logger.info("manifest_gate failed: missing=%s", m.missing)
    return compute_score(
        parse_ok=True,
        manifest_ok=False,
        build_ok=False,
        test_ok=False,
        command_count=p.command_count,
        baseline_size=info["baseline_size"],
        new_size=0,
        dockerfile_text=new_df,
        baseline_command_count=info.get("baseline_command_count"),
    )

# Existing _BUILD_SEM block runs only if manifest gate passes
async with _BUILD_SEM:
    ...
    return compute_score(parse_ok=True, manifest_ok=True, build_ok=..., test_ok=..., ...)
```

The exact insertion point is between the current parse_gate ok-check and the `_BUILD_SEM` block (`dockermin_reward.py` around `:91-103`).

**Step 4: Run tests**

```bash
pytest tests/test_annotate.py -v -k "manifest_gate" -m docker  # docker-required
pytest tests/test_reward.py -v -m "not docker"  # the placeholder tests still pass
make quality  # full suite
```

Expected: all PASS. The non-docker tests confirm no regression; the docker tests verify the gate works against a real daemon.

**Step 5: Commit**

```bash
git add src/dockermin/dataset/annotate.py src/dockermin/reward/dockermin_reward.py tests/test_annotate.py
git commit -m "feat(reward): wire manifest_gate into the reward pipeline"
```

Open PR. Quality green is mandatory before merging.

---

## Phase 2 — Fix B: dataset expansion (additive, holdout-preserving)

### Task 5: Freeze v0 holdout via explicit id list

**Rationale:** Currently `scripts/push_to_hf.py` calls `grouped_train_test_split` which re-derives the holdout on every push. If the corpus grows, the holdout drifts and v2 vs. v3 comparisons stop being comparable. Freeze the 37-id holdout into an explicit fixture.

**Files:**
- Create: `data/curated/holdout_v0_ids.txt` — one id per line, 37 lines
- Modify: `scripts/push_to_hf.py` — replace `grouped_train_test_split` call with id-filter
- Test: `tests/test_push_to_hf.py` (new)

**Step 1: Write the failing test**

Create `tests/test_push_to_hf.py`:

```python
"""Tests for the holdout-id pinning in push_to_hf."""
from __future__ import annotations

from pathlib import Path

import pytest

from dockermin.dataset.split import split_with_frozen_holdout


def test_split_with_frozen_holdout_uses_id_file(tmp_path: Path) -> None:
    """Rows whose id is in holdout_ids go to test split; everything else to train."""
    holdout_ids = {"a", "b", "c"}
    rows = [{"id": x, "dockerfile": f"FROM scratch  # {x}"} for x in ("a", "b", "c", "d", "e", "f")]
    train, test = split_with_frozen_holdout(rows, holdout_ids)
    assert {r["id"] for r in test} == holdout_ids
    assert {r["id"] for r in train} == {"d", "e", "f"}


def test_split_with_frozen_holdout_ignores_missing(tmp_path: Path) -> None:
    """If a holdout id is not in the corpus, the function skips it without raising
    (corpora can shrink over time; the holdout fixture remains the source of truth).
    """
    holdout_ids = {"a", "b", "missing"}
    rows = [{"id": x, "dockerfile": "..."} for x in ("a", "b", "c")]
    train, test = split_with_frozen_holdout(rows, holdout_ids)
    assert {r["id"] for r in test} == {"a", "b"}
    assert {r["id"] for r in train} == {"c"}


def test_split_with_frozen_holdout_rejects_empty_holdout(tmp_path: Path) -> None:
    """An empty holdout set is a configuration error, not a degenerate input."""
    with pytest.raises(ValueError, match="holdout_ids must be non-empty"):
        split_with_frozen_holdout([{"id": "a"}], set())
```

**Step 2: Run test**

```bash
pytest tests/test_push_to_hf.py -v
```

Expected: `ImportError: No module named 'dockermin.dataset.split'`.

**Step 3: Implement**

Create `src/dockermin/dataset/split.py`:

```python
"""Frozen-holdout splitter — the v0 holdout is an explicit fixture, not derived.

The v0 holdout was an emergent property of ``grouped_train_test_split(seed=0)``
on the 145-row triples_with_variants.jsonl that existed at the time. As the
training corpus grows, we MUST keep the holdout ids constant so v2/v3/v4 numbers
stay comparable. This module pins the holdout via an explicit id list rather
than re-deriving it.
"""

from __future__ import annotations

from typing import Any


def split_with_frozen_holdout(
    rows: list[dict[str, Any]], holdout_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``rows`` into (train, test) where test contains exactly the rows
    whose ``id`` is in ``holdout_ids``. Missing ids are silently skipped — the
    holdout fixture is the source of truth, not the corpus.
    """
    if not holdout_ids:
        msg = "holdout_ids must be non-empty; the v0 holdout is the comparison contract"
        raise ValueError(msg)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        if rid in holdout_ids:
            test.append(row)
        else:
            train.append(row)
    return train, test
```

Generate the holdout id list. Run once:

```bash
.venv/bin/python -c "
from datasets import load_dataset
ds = load_dataset('vtemian/dockermin-v0', split='test')
ids = sorted(ex['id'] for ex in ds)
print('\n'.join(ids))
" > data/curated/holdout_v0_ids.txt
wc -l data/curated/holdout_v0_ids.txt  # must read: 37
```

Update `scripts/push_to_hf.py` to consume the fixture:

```python
# Near the top of push_to_hf.py:
from dockermin.dataset.split import split_with_frozen_holdout
from pathlib import Path

HOLDOUT_FIXTURE = Path(__file__).parent.parent / "data" / "curated" / "holdout_v0_ids.txt"


def _load_holdout_ids() -> set[str]:
    return set(HOLDOUT_FIXTURE.read_text().splitlines())


# Replace the grouped_train_test_split call with:
train, test = split_with_frozen_holdout(rows, _load_holdout_ids())
```

**Step 4: Run tests**

```bash
pytest tests/test_push_to_hf.py -v
make quality
```

Expected: PASS.

**Step 5: Commit**

```bash
git checkout -b feat/v3-frozen-holdout
git add data/curated/holdout_v0_ids.txt src/dockermin/dataset/split.py scripts/push_to_hf.py tests/test_push_to_hf.py
git commit -m "feat(dataset): freeze v0 holdout via explicit id fixture"
```

---

### Task 6: Extend scraper code-search queries for missing ecosystems

**Files:**
- Modify: `src/dockermin/dataset/scrape.py:570-576` — add queries for rust, ruby (as FROM), go, dotnet
- Test: `tests/test_scrape.py`

**Step 1: Write the failing test**

Append to `tests/test_scrape.py`:

```python
def test_install_pattern_queries_cover_new_ecosystems() -> None:
    """v3 must query for rust/ruby/go/dotnet bases to expand FROM diversity."""
    from dockermin.dataset.scrape import _INSTALL_PATTERN_QUERIES

    queries = set(_INSTALL_PATTERN_QUERIES)
    must_have = {
        "cargo language:Dockerfile filename:Dockerfile size:<2500",
        "\"bundle install\" language:Dockerfile filename:Dockerfile size:<2500",
        "\"go mod\" language:Dockerfile filename:Dockerfile size:<2500",
        "\"dotnet restore\" language:Dockerfile filename:Dockerfile size:<2500",
    }
    missing = must_have - queries
    assert not missing, f"v3 must query: {sorted(missing)}"
```

**Step 2: Run test**

```bash
pytest tests/test_scrape.py::test_install_pattern_queries_cover_new_ecosystems -v
```

Expected: FAIL with the four missing queries listed.

**Step 3: Implement**

In `src/dockermin/dataset/scrape.py:570-576`, extend the tuple:

```python
_INSTALL_PATTERN_QUERIES: tuple[str, ...] = (
    "pip language:Dockerfile filename:Dockerfile size:<2500",
    "npm language:Dockerfile filename:Dockerfile size:<2500",
    "gem language:Dockerfile filename:Dockerfile size:<2500",
    "composer language:Dockerfile filename:Dockerfile size:<2500",
    '"go build" language:Dockerfile filename:Dockerfile size:<2500',
    # v3 expansion — broaden FROM-base diversity per the failure analysis
    # in docs/decisions/2026-06-02-v2-headline-result.md.
    "cargo language:Dockerfile filename:Dockerfile size:<2500",
    '"bundle install" language:Dockerfile filename:Dockerfile size:<2500',
    '"go mod" language:Dockerfile filename:Dockerfile size:<2500',
    '"dotnet restore" language:Dockerfile filename:Dockerfile size:<2500',
)
```

**Step 4: Run test**

```bash
pytest tests/test_scrape.py -v -k "install_pattern_queries"
```

Expected: PASS.

**Step 5: Commit**

```bash
git checkout -b feat/v3-scraper-queries
git add src/dockermin/dataset/scrape.py tests/test_scrape.py
git commit -m "feat(scrape): add cargo/bundle/go-mod/dotnet code-search queries"
```

---

### Task 7: Bump official-images limit + add chainguard fetcher

**Files:**
- Modify: `scripts/run_scrape.py:18-31` — bump official-images limit from 120 → 250, add a call to a new `fetch_chainguard_images`
- Modify: `src/dockermin/dataset/scrape.py` — add `fetch_chainguard_images` (similar shape to `fetch_awesome_compose`)
- Test: `tests/test_scrape.py`

**Step 1: Write the failing test**

Append to `tests/test_scrape.py`:

```python
def test_fetch_chainguard_images_is_callable() -> None:
    """Smoke: chainguard fetcher exists and accepts a limit kwarg."""
    from dockermin.dataset.scrape import fetch_chainguard_images
    import inspect

    sig = inspect.signature(fetch_chainguard_images)
    assert "limit" in sig.parameters
```

**Step 2: Run test**

```bash
pytest tests/test_scrape.py::test_fetch_chainguard_images_is_callable -v
```

Expected: `ImportError: cannot import name 'fetch_chainguard_images'`.

**Step 3: Implement**

In `src/dockermin/dataset/scrape.py`, after `fetch_awesome_compose`:

```python
_CHAINGUARD_REPO = "chainguard-images/images"


def fetch_chainguard_images(limit: int = 50) -> Iterator[Candidate]:
    """Walk chainguard-images/images on GitHub for Wolfi-derived Dockerfiles.

    Chainguard images use ``apk`` (Wolfi/Alpine) instead of apt — entirely
    unrepresented in dockermin-v0. Yields candidates the same shape as
    ``fetch_awesome_compose``. License: Apache-2.0 per the repo.
    """
    seen_hashes: set[str] = set()
    yielded = 0
    for path in _walk_chainguard_dir(_CHAINGUARD_REPO, "images", max_depth=2):
        if yielded >= limit:
            break
        if not path.endswith("/Dockerfile") and not path.endswith(".Dockerfile"):
            continue
        candidate = _candidate_from_github_path(_CHAINGUARD_REPO, path, license="Apache-2.0")
        if candidate is None:
            continue
        if not _is_self_contained_probeable(candidate.dockerfile):
            continue
        if not _is_new_content(candidate.dockerfile, seen_hashes):
            continue
        yield candidate
        yielded += 1


def _walk_chainguard_dir(repo: str, root: str, max_depth: int) -> Iterator[str]:
    """Yield file paths from ``repo`` under ``root`` via the GH contents API."""
    # Reuse the same walker pattern as _walk_awesome_compose_dir — refactor
    # opportunity for v4 if a third walker shows up.
    ...  # implementation mirrors _walk_awesome_compose_dir at scrape.py:361-370
```

The implementation re-uses the same `gh api repos/{repo}/contents/{path}` recursive walker pattern from `_walk_awesome_compose_dir` at `scrape.py:361-370`. Keep it as a sibling helper for now — extract into a shared `_walk_github_tree` only if a fourth walker appears.

In `scripts/run_scrape.py`, update the writer loop:

```python
fetchers = (
    ("official_images", fetch_official_images(limit=250)),  # was 120
    ("awesome_compose", fetch_awesome_compose(limit=80)),
    ("chainguard_images", fetch_chainguard_images(limit=50)),  # NEW
    ("github_search", fetch_github_search(limit=300)),
)
```

**Step 4: Run test**

```bash
pytest tests/test_scrape.py -v -k "fetch_chainguard"
make quality
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/dockermin/dataset/scrape.py scripts/run_scrape.py tests/test_scrape.py
git commit -m "feat(scrape): chainguard images fetcher + bump official-images limit to 250"
```

---

### Task 8: Run scrape → annotate → variants → push pipeline (operational, not TDD)

**Manual orchestration task — not a code change.** Records the exact commands and verification checkpoints. Run on a GPU pod (massedcompute A100, see Task 11 for rental).

**Steps:**

1. **Scrape** (network only, no GPU):
   ```bash
   make scrape  # scripts/run_scrape.py — writes data/raw/candidates.jsonl
   wc -l data/raw/candidates.jsonl  # expect 500-800 raw candidates
   ```

2. **Annotate** (needs docker, ~5-10 min per candidate cold; runs in parallel):
   ```bash
   make annotate  # scripts/run_annotate.py — writes data/curated/triples.jsonl
   # Watch for failures; expect ~30-50 % of candidates to drop at annotate (no test_cmd, build fails, etc.)
   wc -l data/curated/triples.jsonl  # expect ~200-400 surviving annotated triples
   ```

3. **Synthesize variants** (needs Anthropic API; ~$20 budget at ~$0.10/variant):
   ```bash
   python scripts/synthetic_variants.py --variants-per-base 4 --max-bases 200
   wc -l data/curated/triples_with_variants.jsonl  # expect ~1000 rows
   ```

4. **Verify base diversity** (sanity-check the gap closes):
   ```bash
   .venv/bin/python -c "
   import json
   rows = [json.loads(l) for l in open('data/curated/triples_with_variants.jsonl')]
   import re
   def base(df):
       m = re.search(r'FROM\s+([^\s:@]+)', df, re.I)
       return m.group(1) if m else 'unknown'
   from collections import Counter
   c = Counter(base(r['dockerfile']) for r in rows)
   print('unique bases:', len(c))
   for name, n in c.most_common(20): print(f'  {name}: {n}')
   "
   # Pass criteria: ≥ 12 unique base-image names (vs 7 in v2); rust, go, ruby, dotnet present.
   ```

5. **Push to HF** (with frozen holdout):
   ```bash
   python scripts/push_to_hf.py --dataset-id vtemian/dockermin-v1
   # Verify:
   .venv/bin/python -c "
   from datasets import load_dataset
   ds = load_dataset('vtemian/dockermin-v1')
   print('train:', len(ds['train']), 'test:', len(ds['test']))
   # test MUST be 37 with the same ids as v0
   v0_ids = set(open('data/curated/holdout_v0_ids.txt').read().splitlines())
   v1_test_ids = {ex['id'] for ex in ds['test']}
   assert v1_test_ids == v0_ids, f'holdout drifted: {v0_ids - v1_test_ids} | {v1_test_ids - v0_ids}'
   print('holdout invariant: OK')
   "
   ```

6. **Record dataset stats** for the v3 retrospective:
   ```bash
   python scripts/dataset_stats.py --dataset vtemian/dockermin-v1 > docs/v3_dataset_stats.md
   ```
   (Add a small `dataset_stats.py` if it doesn't exist — counts unique source_urls, bases, tags, ecosystems.)

**Commit point (after push succeeds):**

```bash
git add data/curated/holdout_v0_ids.txt docs/v3_dataset_stats.md
git commit -m "data(v3): expand training set to ~1000 examples / ≥12 unique bases"
```

The dataset itself lives on HF (`vtemian/dockermin-v1`); only the metadata / fixture lives in the repo.

---

## Phase 3 — Training config for v3

### Task 9: New `configs/dockermin_v3.toml` with extended training schedule

**Files:**
- Create: `configs/dockermin_v3.toml`
- Test: smoke test via `uv run --no-sync rl @ configs/dockermin_v3.toml --dry-run` (operational)

**Step 1: Write the failing test (config-structure smoke)**

Create `tests/test_dockermin_v3_config.py`:

```python
"""Smoke: the v3 config exists, is valid TOML, and encodes the hyperparam choices
from docs/plans/2026-06-02-grpo-v3-manifest-gate-and-dataset-expansion.md."""
from pathlib import Path
import tomllib


def test_v3_config_max_steps_in_range() -> None:
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert 400 <= cfg["max_steps"] <= 500


def test_v3_config_group_size_unchanged_from_v2() -> None:
    """Agent 4: no prime-rl precedent for group_size=32 on a 7B + LoRA."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["orchestrator"]["group_size"] == 16


def test_v3_config_batch_size_raised() -> None:
    """Agent 4: prime-rl reference configs run at batch_size 256-512; v2's 16 was unusually small.
    v3 raises to at least 32 to reduce variance estimate noise."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["orchestrator"]["batch_size"] >= 32


def test_v3_config_ckpt_interval_at_50() -> None:
    """Eval at steps 100, 200, 300, 400, 500 requires ckpts at the 50-step grid
    (the eval is offline — we read STABLE broadcasts post-hoc)."""
    cfg = tomllib.loads((Path(__file__).parent.parent / "configs" / "dockermin_v3.toml").read_text())
    assert cfg["ckpt"]["interval"] == 50
```

**Step 2: Run tests**

```bash
pytest tests/test_dockermin_v3_config.py -v
```

Expected: FAIL — no v3 config.

**Step 3: Implement**

Copy `configs/dockermin_full.toml` to `configs/dockermin_v3.toml` and modify these values (exact line numbers will vary; search for the keys):

```toml
# Top of file:
# v3 GRPO retry — see docs/plans/2026-06-02-grpo-v3-manifest-gate-and-dataset-expansion.md
# Key deltas vs v2 (dockermin_full.toml):
# - max_steps: 250 -> 500 (prime-rl precedent for hard tasks; eval every 50)
# - batch_size: 16 -> 32 (prime-rl ref configs run at 256-512; 32 is the v2-stretch budget)
# - group_size: 16 -> 16 (unchanged; no prime-rl precedent for 32)
# - ckpt interval: 25 -> 50 (eval grid aligned)
# - dataset: vtemian/dockermin-v0 -> vtemian/dockermin-v1 (expanded corpus)
# KL/beta intentionally unchanged — Agent 4 found no evidence for tuning.

max_steps = 500

[orchestrator]
batch_size = 32
group_size = 16

[ckpt]
interval = 50
keep_last = 6  # 100, 200, 300, 400, 500 + headroom

[wandb]
project = "dockermin"
name = "v3-manifest-and-dataset"
offline = true  # same defense as v2

[trainer.wandb]
project = "dockermin"
name = "v3-manifest-and-dataset"
offline = true

[orchestrator.wandb]
project = "dockermin"
name = "v3-manifest-and-dataset"
offline = true

[trainer.dataset]
id = "vtemian/dockermin-v1"  # was vtemian/dockermin-v0

[[orchestrator.filters]]
type = "zero_advantage"
enforce = false  # unchanged from v1.1; the new manifest rung can starve advantage on hallucination-heavy groups
```

Run a dry-run to verify the config parses through prime-rl:

```bash
# On a fresh pod (Task 11):
uv run --no-sync rl @ ~/dockermin/configs/dockermin_v3.toml --dry-run
# Check outputs/configs/{inference,orchestrator,trainer}.toml were written.
```

**Step 4: Run tests**

```bash
pytest tests/test_dockermin_v3_config.py -v
make quality
```

Expected: PASS.

**Step 5: Commit**

```bash
git checkout -b feat/v3-training-config
git add configs/dockermin_v3.toml tests/test_dockermin_v3_config.py
git commit -m "config(v3): training schedule for manifest-gated, 1000-row corpus"
```

---

## Phase 4 — Pod ops + execution

### Task 10: Reuse `scripts/pod_ops/` as-is

**No code changes.** Verify v1.1 pod-ops scripts still work as intended. They are the contract for the next pod.

**Verification (operational, no commits):**

```bash
# After renting pod #10 (see Task 11):
scp scripts/pod_ops/{install.sh,launch_v1_1.sh,post_train.sh,docker_pruner.sh,hf_pusher.sh} ubuntu@$IP:~
ssh ubuntu@$IP 'bash ~/install.sh'  # idempotent: uv, prime-rl, dockermin, env-adapter, deps, wandb-offline, sysctl
```

**One-line edits at pod-run-time (not in repo) — point the launcher at the v3 config:**

The `launch_v1_1.sh` `rl @ ~/dockermin/configs/dockermin_full.toml` becomes `rl @ ~/dockermin/configs/dockermin_v3.toml`. This is a *runtime* edit on the pod, not a repo change — keep `launch_v1_1.sh` as the canonical v1.1 reference.

---

### Task 11: Rent pod, run training, run eval matrix

**Operational task. No code changes. Wallet check first.**

**Pre-flight:**
- Wallet at start of plan execution: **$12 estimated remaining** from the v2 session. **A top-up may be required.** Flag this to Vlad before renting.
- Target pod: A100 80 GB PCIe via massedcompute (`prime availability list --gpu-type A100_80GB --gpu-count 1`). Avoid lambdalabs unless massedcompute is OOS — three lambdalabs failures earlier.
- Disk: 500 GB (v3 training is 2× v2's step count + larger corpus; the docker_pruner.sh holds the line but margins are tighter).

**Pod-rental commands:**

```bash
prime pods create --id ad4b66 \
  --name dockermin-grpo-v3 \
  --disk-size 500 \
  --image ubuntu_22_cuda_12 \
  --yes --plain < /dev/null
```

Poll until `Status=ACTIVE` and `Installation Status=FINISHED`. Stage secrets (`~/.cache/huggingface/token`, `~/.prime/config.json`). Run `~/install.sh`. Launch the three components (inference, trainer, orchestrator) per the v1.1 recipe but with `configs/dockermin_v3.toml`. Launch `docker_pruner.sh` and `hf_pusher.sh` as background services so a pod death never loses progress.

**Expected wall-clock:** ~14-18 h at ~3 min/step × 500 steps on A100 (slower than H100). Cost ~$25-30 at $1.79/h.

**Holdout eval at steps 100, 200, 300, 400, 500:**

After the training pod terminates (or while it runs, on a sibling eval pod), eval each STABLE checkpoint:

```bash
# On a fresh A100 eval pod (~$2/h × ~1.5 h per checkpoint × 5 checkpoints = ~$15):
for STEP in 100 200 300 400 500; do
  python scripts/run_eval.py \
    --baselines dockermin \
    --dockermin-model vtemian/dockermin-qwen7b-lora-v3 \
    --dockermin-subfolder step_${STEP} \
    --temperature 0.2 --max-new-tokens 1024 \
    --out ~/eval_v3_step${STEP}.jsonl
done
```

(The `--dockermin-subfolder` flag needs adding — see Task 12.)

**Decision after eval:**
- If holdout `mean_reduction | pass` improves AND `pass_rate` improves over v2 step_250 @ T=0.2 → ship v3 step_X as the new headline.
- If `mean_reduction | pass` improves but `pass_rate` regresses → revisit the manifest-gate magnitude or dataset balance; document and stop.
- If neither improves → ship NEGATIVE_RESULT.md per the project's pre-commitment.

---

### Task 12: Add `--dockermin-subfolder` flag to run_eval.py (TDD)

**Files:**
- Modify: `scripts/run_eval.py` — add the CLI flag
- Modify: `src/dockermin/eval/baselines.py` — accept `subfolder` kwarg in `baseline_dockermin` and thread to `PeftModel.from_pretrained(..., subfolder=...)`
- Test: `tests/test_run_eval.py` (new — pure CLI test, no GPU)

**Step 1: Write the failing test**

Create `tests/test_run_eval.py`:

```python
import subprocess
import sys
from pathlib import Path


def test_run_eval_cli_accepts_dockermin_subfolder() -> None:
    """CLI must expose --dockermin-subfolder so we can eval per-step adapters
    (e.g. step_100 vs step_250) from the same HF repo."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "scripts" / "run_eval.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--dockermin-subfolder" in result.stdout
```

**Step 2: Run test**

```bash
pytest tests/test_run_eval.py -v
```

Expected: FAIL (flag not present).

**Step 3: Implement**

In `scripts/run_eval.py`, add to the argparse setup near the `--dockermin-model` flag:

```python
p.add_argument(
    "--dockermin-subfolder",
    default=None,
    help=(
        "Subfolder within --dockermin-model that contains adapter_model.safetensors. "
        "Used to eval per-step checkpoints stored as step_N/ subfolders (e.g. step_250)."
    ),
)
```

In the dispatch loop, thread it to the baseline:

```python
if baseline == "dockermin":
    kwargs["model_id"] = args.dockermin_model
    if args.dockermin_subfolder:
        kwargs["subfolder"] = args.dockermin_subfolder
```

In `src/dockermin/eval/baselines.py`, update `baseline_dockermin` and `_hf_dockermin`:

```python
def _hf_dockermin(model_id: str, subfolder: str | None = None) -> tuple[Any, Any]:
    cache_key = (model_id, subfolder)
    if "model" in _HF_HANDLE and _HF_HANDLE.get("adapter") == cache_key:
        return _HF_HANDLE["model"], _HF_HANDLE["tok"]
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16", device_map="auto")
    if subfolder is not None:
        model = PeftModel.from_pretrained(base, model_id, subfolder=subfolder)
    else:
        model = PeftModel.from_pretrained(base, model_id)
    _HF_HANDLE["model"] = model
    _HF_HANDLE["tok"] = tok
    _HF_HANDLE["adapter"] = cache_key
    return model, tok


def baseline_dockermin(  # noqa: PLR0913
    triple: dict[str, Any],
    model_id: str = "vtemian/dockermin-qwen7b-lora-v1",
    subfolder: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
) -> EvalEntry:
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        model, tok = _hf_dockermin(model_id, subfolder=subfolder)
        ...  # rest unchanged
```

**Step 4: Run tests**

```bash
pytest tests/test_run_eval.py -v
make quality
```

Expected: PASS.

**Step 5: Commit**

```bash
git checkout -b feat/v3-eval-subfolder
git add scripts/run_eval.py src/dockermin/eval/baselines.py tests/test_run_eval.py
git commit -m "feat(eval): --dockermin-subfolder flag for per-step checkpoint eval"
```

---

## Phase 5 — Write up results

### Task 13: Generate the v3 leaderboard

**Operational. No new code. Pull v1 + v2 + v3 evals from HF; run `scripts/leaderboard.py`; commit the markdown.**

```bash
.venv/bin/python <<'PY'
from huggingface_hub import hf_hub_download
import json, pathlib
out = pathlib.Path('/tmp/v3_merged_eval.jsonl')
rows = []

for repo, file_, tag_prefix in (
    ("vtemian/dockermin-qwen7b-lora-v1", "eval/results.jsonl", None),  # all baselines
    ("vtemian/dockermin-qwen7b-lora-v2", "eval/results.jsonl", "dockermin_v2_s250_t02"),
    ("vtemian/dockermin-qwen7b-lora-v3", "eval/results_step_100.jsonl", "dockermin_v3_s100"),
    ("vtemian/dockermin-qwen7b-lora-v3", "eval/results_step_500.jsonl", "dockermin_v3_s500"),
):
    p = hf_hub_download(repo, file_, force_download=True)
    for r in (json.loads(l) for l in open(p)):
        if tag_prefix and r["baseline"] == "dockermin":
            r["baseline"] = tag_prefix
        rows.append(r)
out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"wrote {len(rows)} rows")
PY

.venv/bin/python scripts/leaderboard.py --in /tmp/v3_merged_eval.jsonl --out docs/leaderboard.md
```

**Commit:**

```bash
git checkout -b docs/v3-leaderboard
git add docs/leaderboard.md
git commit -m "docs(v3): leaderboard with v3 step grid + v2 baseline + all v1 references"
```

---

### Task 14: Decision doc for v3

**Files:**
- Create: `docs/decisions/2026-06-XX-v3-results.md` (substitute eval date)

Mirror `docs/decisions/2026-06-02-v2-headline-result.md`'s structure:

1. **TL;DR** with the leaderboard table.
2. **Context** — what v2 left open, what v3 attempted.
3. **What each fix did:**
   - Fix A (manifest gate): expected reduction in hallucinated-FROM failures from 9 → near-0; quote the v3 holdout numbers.
   - Fix B (dataset): unique-base count moved from 7 → N; report per-base pass/fail breakdown.
4. **Checkpoint sweep** — pass/reduction at step 100, 200, 300, 400, 500.
5. **Cost** — pod-by-pod table; update `docs/cost_log.md`.
6. **What would v4 try** (if budget remains).

```bash
git checkout -b docs/v3-results
git add docs/decisions/2026-06-XX-v3-results.md docs/cost_log.md
git commit -m "docs(v3): final results, sweep table, cost log, retrospective"
```

---

## Risk register & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pod dies UNKNOWN mid-training (3 of 9 v2 pods did) | High | `scripts/pod_ops/hf_pusher.sh` pushes every STABLE broadcast every 5 min; pod death never loses > 1 ckpt of progress. `docker_pruner.sh` + `fs.mount-max=1000000` cut the failure rate by half empirically. |
| Wallet runs out mid-training | Medium-high | Flag the top-up to Vlad before renting Task 11's pod. Estimated v3 total: ~$50. |
| Manifest gate calls a stale local docker cache and passes a hallucinated tag | Low | `_image_exists_locally` is BY DESIGN a pass — if it's locally cached, the build will succeed. The risk reduces to "the pod's local cache contains an image that doesn't exist in the registry," which is harmless (the build path still works). |
| `docker manifest inspect` rate-limits (Docker Hub unauthenticated) | Low-medium | Default 100 pulls/6h for anonymous. The gate runs once per *unique* FROM per rollout. At 32×16=512 rollouts/step × 500 steps × ~1.5 unique FROMs/rollout / 6h-window, we hit ~300 manifest calls/h — under the limit. If hit: log a warning, treat as pass-on-timeout. |
| Dataset expansion misses target diversity (e.g. all new bases are still python:3.12-something variants) | Medium | Task 8 step 4 verifies ≥ 12 unique base-image *names* before pushing to HF. Hard fail if not. |
| v3 wins on pass rate but loses mean_reduction (the manifest gate pushes the model to "play it safe") | Medium | This is the central trade-off. Decision rule: if mean_reduction drops below ~50 %, the manifest-gate score may need rebalancing (e.g. `MANIFEST_FAIL_SCORE = -0.02` instead of `-0.05`). Document and consider a v3.1 with lighter magnitude. |
| Subfolder loading of LoRA fails silently (PEFT loads a different adapter or none) | Low | Test it on step_100 of v2's existing HF repo before the v3 eval — verify the model behaves differently from a no-adapter base. |

---

## Quality gate checklist (per commit)

Before every commit:

- [ ] `make quality` — fmt + lint + typecheck + test-pure all green
- [ ] No `# noqa` without a trailing reason
- [ ] No `print(...)` in `src/dockermin/` (logger.* instead)
- [ ] No `from scratch + RUN echo` cheat in new test fixtures
- [ ] Conventional-commit message: `feat|fix|docs|refactor|test|chore(scope): description`
- [ ] On a feat/* or fix/* branch, not main

---

## Estimated cost & timeline

| Phase | Wall-clock | Cost (est.) |
|---|---|---|
| Phase 1 (manifest gate, TDD) | 1-2 h coding | $0 |
| Phase 2 (dataset growth) | 4-6 h ops (scrape + annotate + variants + push) | $20-25 (Claude variants + 1-2 h annotation pod) |
| Phase 3 (config) | 30 min | $0 |
| Phase 4 (train + eval pod) | 18-22 h (14-18 train + 5 eval) | $35-45 |
| Phase 5 (writeup) | 1 h | $0 |
| **Total** | **~28-34 h end-to-end** | **~$60-70** |

**Wallet check required before Task 11.** Current estimate ~$12 remaining from v2; need at least $50 buffer for v3. Flag the top-up.

---

## Out-of-scope for v3

- Switching from `docker buildx` to plain `docker build` (would simplify pod-ops, but breaks v1.1's cache-from/cache-to optimization and isn't on the failure-mode critical path).
- Adding a Docker Hub auth token to the manifest gate (would lift rate limits but adds a secret to manage; revisit if v3 hits rate limits in practice).
- Curriculum learning (training on easier examples first, harder last). Cited in Agent 4's report; no prior art for prime-rl specifically; defer to v4.
- Training KL adjustment. Agent 4 found no evidence; defer to v4 if v3 shows reward-hacking signal.
- A dedicated post-mortem after the next pod death. Defer until the next pod death; reuse the existing `docs/decisions/2026-05-29-pod-failure-postmortem.md` if the failure mode matches.
