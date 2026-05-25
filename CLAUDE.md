# CLAUDE.md

This file provides guidance to Claude Code when working with the dockermin codebase.

## Project Overview

**dockermin** is a GRPO-fine-tuned Qwen 2.5 Coder 7B Instruct that rewrites working
Dockerfiles into smaller, functionally-equivalent ones. The reward signal comes from real
`docker build` + `docker run <test_cmd>` pairs: a smaller image that still passes its test
wins; a broken image scores zero. The repo is plain Python (3.11) — a library under
`src/dockermin/` (dataset scraping, annotation, reward gates, eval baselines), CLI glue
under `scripts/`, and a thin prime-rl environment adapter under `prime_env/`. No web layer.

## Commands

```bash
make quality      # fmt + lint + typecheck + test-pure — RUN THIS BEFORE EVERY COMMIT
make test         # full pytest suite (includes docker-daemon-gated tests)
make test-pure    # pytest -m "not docker" (no Docker daemon required)
make fmt          # ruff format
make lint         # ruff check --fix
make typecheck    # mypy strict on src + tests

# Script entrypoints (CLI UIs, not library code)
make scrape       # scripts/run_scrape.py        — build the Dockerfile corpus
make annotate     # scripts/run_annotate.py      — parse/build/test gate each Dockerfile
python scripts/synthetic_variants.py             — generate size-reduced training variants
python scripts/run_eval.py                       — run baselines + model, emit results.jsonl
make leaderboard  # scripts/leaderboard.py       — render results into docs/leaderboard.md
```

## Pre-Commit Gate (MANDATORY)

Before every commit:

1. Run `make quality`. **Do not commit if ruff, mypy, or pytest report a failure.**
2. **Never commit directly to `main`.** Branch per change: `git checkout -b feat/description`.
3. Use conventional commits: `type(scope): description` where type is one of
   `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

The same gate runs in CI (`.github/workflows/quality.yml`); a green local gate is the
contract for a green CI run.

## Code Quality Standards (MANDATORY)

These are non-negotiable. Most are enforced mechanically by ruff + mypy; the rest are
enforced by review. Where a rule overlaps a linter, the linter wins ties.

### No Nesting

Maximum one indent level inside a function body. Flatten with guard clauses and extract
helpers. Every function reads top-to-bottom.

**BAD:**
```python
def score_rewrite(original, rewrite):
    if rewrite.parse_ok:
        if rewrite.build_ok:
            if rewrite.test_ok:
                return size_reward(original.size, rewrite.size)
    return 0.0
```

**GOOD:**
```python
def score_rewrite(original: Image, rewrite: Image) -> float:
    if not (rewrite.parse_ok and rewrite.build_ok and rewrite.test_ok):
        return 0.0
    return size_reward(original.size, rewrite.size)
```

### Names Are Contracts

Function names must reflect behavior exactly:
- `fetch_*` returns a collection (possibly empty), never `None` — e.g. `read_jsonl` -> `list`.
- `find_*` may return `None`.
- `get_*` raises if the thing is absent.

No generic names: no `Manager`, `Helper`, `Data`, `Service`, `process_*`, `handle_*`
without a real domain noun.

**BAD:**
```python
def get_base_image(dockerfile: str) -> str | None: ...   # returns None but named get_
def process_variants(rows): ...                          # process_ + no domain noun
```

**GOOD:**
```python
def find_base_image(dockerfile: str) -> str | None: ...
def fetch_variants(rows: list[dict[str, Any]]) -> list[Variant]: ...
def shrink_dockerfile(dockerfile: str) -> str: ...
```

### Fail Fast, Fail Visible

Catch specific exception tuples — never a bare `except Exception`. Raise via a local
message variable (ruff `EM`/`TRY`). In handlers, log the traceback with
`logger.exception(...)`. The only sanctioned bare catch is a **documented safety-net** in
the eval baselines — one malformed Dockerfile must not crash the eval loop — carrying
`# noqa: BLE001` with a trailing reason.

**BAD:**
```python
try:
    image = build_image(dockerfile)
except Exception:        # swallows everything, no traceback
    return None
```

**GOOD:**
```python
try:
    image = build_image(dockerfile)
except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    logger.exception("docker build failed for %s", tag)
    raise

# Sanctioned safety-net (eval baselines only):
try:
    result = run_baseline(dockerfile)
except Exception as e:  # noqa: BLE001 — one bad Dockerfile must not crash the eval loop
    result = BaselineResult(error=str(e))

# Raising via a local message (EM rule):
msg = f"build timeout after {timeout_s}s"
raise BuildError(msg)
```

### No Narration

Comment WHY, never WHAT. If a comment restates the code, delete it and improve the name.

**BAD:**
```python
# collapse adjacent RUN lines
df = _collapse_consecutive_runs(df)
# swap to the slim base
df = _swap_base_image(df)
```

**GOOD:**
```python
# hadolint DL3009: apt lists left in the layer inflate size even after removal,
# so the cleanup must share the RUN that installed them.
df = _collapse_consecutive_runs(df)
```

### Plain Functions Over Classes

Business logic is plain functions; no stateful service objects. Classes are reserved for
`@dataclass(frozen=True)` result/value types (e.g. the eval `BaselineResult`) and
exception classes.

**BAD:**
```python
class DockerfileMinimizer:
    def __init__(self, dockerfile): self.dockerfile = dockerfile
    def run(self): self._strip(); self._swap(); return self.dockerfile
```

**GOOD:**
```python
def minimize_dockerfile(dockerfile: str) -> str:
    return swap_base_image(strip_dev_packages(dockerfile))
```

### Type Hints Required

Every signature is fully annotated — params and return. `mypy --strict` enforces this on
`src` and `tests`. SDK boundaries (anthropic/openai/docker/etc.) are
`ignore_missing_imports`; pin their results with a local annotation or `cast(...)`, never
a blanket `type: ignore`.

### Logging, Not Printing

Library code (`src/dockermin/`) uses `logger = logging.getLogger(__name__)` and never
`print`. `scripts/` are the exception — they are CLI UIs where `print` is the intended
output (per-file `T20` ignore in `pyproject.toml`).

## The Machine Gates

The machine-readable source of truth is `pyproject.toml` (`[tool.ruff]` + `[tool.mypy]`);
the prose above must agree with it. Explicit complexity limits:

| Limit | Value | ruff key |
|---|---|---|
| Cyclomatic complexity | 5 | `mccabe.max-complexity` |
| Branches per function | 4 | `pylint.max-branches` |
| Return statements | 4 | `pylint.max-returns` |
| Statements per function | 15 | `pylint.max-statements` |
| Arguments per function | 6 | `pylint.max-args` |

Hitting a limit is a signal to extract a helper, not to raise the cap. The genuinely-wide
signatures (`compute_score`, `annotate_one`) carry a per-line `# noqa: PLR0913` with a
reason; new ones need the same justification. Per-file relaxations (tests, scripts,
`cli.py`, `prime_env/`) live in `[tool.ruff.lint.per-file-ignores]` and
`[[tool.mypy.overrides]]` — read those before adding a `# noqa`; the right fix is usually
code, not a suppression.
