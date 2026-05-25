# Dockermin Dataset Growth Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Grow the training set from 11 distinct bases to 30-50, fixing the convergent root cause a 3-agent investigation found: `infer_test_cmd` drops 62% of candidates as "unknown ecosystem" and the scraper targets file-size instead of the install pattern. Pure local work, no GPU.

**Architecture:** Three code fixes raise yield (probe-extraction bug fixes, ecosystem breadth, scraper retargeting), then a re-scrape + re-annotate produces the new base set, then variant generation multiplies it ~6x. The empty-build-context gate stays strict (relaxing it reopens the reward-hack surface); we grow by sourcing self-contained Dockerfiles, not by weakening the gate.

**Tech Stack:** unchanged - Python 3.11, `dockerfile` lib, docker buildx, `gh` CLI, ruff+mypy gates (stay green). All work runs locally with Docker Desktop.

**Evidence (this session, 3 context agents):** yield-analysis (455 sample), re-annotation-viability (11 bases), source-expansion. See `docs/journal.md` for the consolidated findings.

---

## Ground truth (do not re-derive)

- **62% (283/455) drop at `infer_test_cmd` -> (None,None)** - unknown ecosystem (go 26, nginx 19, php 8, ruby 4, rust, DB/middleware images). Pure code fix; the single biggest lever.
- **31% (141/455) need a local build context** (COPY/ADD repo files, `git pull && make`, `./mvnw`). Structural - need self-contained sources, NOT gate relaxation (relaxing reopens `RUN echo` cheats).
- **~6% (~27) are self-contained + known ecosystem** = today's ceiling -> ~16 passing bases.
- **3 probe-extraction bugs** in the just-hardened `infer_test_cmd` drop 4 of the existing 11 bases (all build fine):
  1. Quote-strip: `pip install "hy..."` -> `import_module('"hy')` (leading quote captured).
  2. Cross-line npm: bare `npm install` regex spans newline, grabs `COPY` from the next line.
  3. Global npm: `npm install -g X` -> global module not on bare `require()` path.
- **official-images = structural dead end** (base-image factories: compile from source, COPY entrypoints).
- **GitHub code-search legacy syntax: no `NOT`/`OR`** - must filter COPY/unknown client-side.
- Scraper currently optimizes `size:<5000` (raw volume), orthogonal to the gate -> 80% of hits have COPY.

## Target & honest ceiling

- **Target: 30-50 distinct bases** -> ~180-300 rows after variant generation. Clears the 100-row floor AND fixes base-diversity (vs 11 today).
- Ceiling with empty-context kept: ~35-45 realistically. The optional synthetic-app-file relaxation (Phase 5, deferred) unlocks 60+ but adds machinery - YAGNI unless we fall short.

---

# Phase 1: Fix the probe-extraction bugs (recover 11 -> 11 bases)

## Task 1.1: Fix `_first_pip_package` quote-stripping (TDD)

**Files:** `src/dockermin/dataset/annotate.py`, `tests/test_annotate.py`

**Step 1: Failing test:**
```python
def test_first_pip_package_strips_quotes() -> None:
    cmd, _ = infer_test_cmd('FROM python:3.12\nRUN pip install "hy == 1.0"\n')
    joined = " ".join(cmd)
    assert "'\"hy'" not in joined and '"hy' not in joined
    assert "import_module('hy')" in joined or "'hy'" in joined
```

**Step 2: Run, expect fail** (current captures the leading `"`).

**Step 3: Implement** - in `_first_pip_package`, strip quotes from the token before the version split: `token = token.strip("'\"")`. Also strip a trailing `"`/`'`.

**Step 4: Run, pass + full suite green.**

**Step 5: Commit** (on a feature branch - main is hook-blocked):
```bash
git commit -m "fix(dataset): strip quotes in pip package extraction (hy/wheel probes)"
```

## Task 1.2: Fix npm extraction - same-line + global fallback (TDD)

**Files:** `src/dockermin/dataset/annotate.py`, `tests/test_annotate.py`

**Step 1: Failing tests:**
```python
def test_npm_bare_install_falls_back_to_version_marker() -> None:
    # `npm install` with no package (install-from-package.json) must NOT
    # grab the next Dockerfile line (COPY) as the module name
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install\nCOPY . .\n")
    assert "COPY" not in " ".join(cmd) and "copy" not in " ".join(cmd)
    assert expected == "NODEOK"  # version-marker fallback

def test_npm_global_install_uses_version_marker() -> None:
    # `npm install -g X` -> global module not on bare require() path
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install -g tough-cookie\n")
    assert "require(" not in " ".join(cmd)  # falls back, doesn't require() a global
    assert expected == "NODEOK"
```

**Step 2: Run, expect fail.**

**Step 3: Implement** in the npm probe path:
- Anchor the package match to the same line; if the next token is a Dockerfile keyword (`COPY`, `RUN`, `CMD`, ...) or absent, treat as "no package" -> version-marker fallback (`node -e "console.log('NODEOK', process.version)"`).
- If the install line carries `-g`/`--global`, also use the version-marker fallback (global modules aren't require-able on the bare path).

**Step 4: Run, pass + suite green.**

**Step 5: Commit** `fix(dataset): npm probe - same-line match + global/bare fallback`

---

# Phase 2: Broaden `infer_test_cmd` ecosystems (the 62% lever)

## Task 2.1: Add go/ruby/php/rust runtime probes (TDD)

**Files:** `src/dockermin/dataset/annotate.py`, `tests/test_annotate.py`

**Step 1: Failing tests** - each new ecosystem returns a runtime probe (not None):
```python
def test_infer_test_cmd_go() -> None:
    cmd, expected = infer_test_cmd("FROM golang:1.22\nRUN go build ./...\n")
    assert cmd is not None and cmd[0] == "go" and expected == "go version"  # or GOOK marker

def test_infer_test_cmd_ruby() -> None:
    cmd, expected = infer_test_cmd("FROM ruby:3.3\nRUN gem install rails\n")
    assert cmd is not None and cmd[0] == "ruby"

def test_infer_test_cmd_php() -> None:
    cmd, expected = infer_test_cmd("FROM php:8.3\nRUN docker-php-ext-install pdo\n")
    assert cmd is not None and cmd[0] == "php"
```

**Step 2: Run, expect fail** (these currently return (None,None)).

**Step 3: Implement** - extend the ecosystem dispatch with runtime probes that require the interpreter (resist `RUN echo`):
- go: `["go", "version"]` expecting `"go version"` (go toolchain present). For built binaries we can't generically probe, so the toolchain check is the floor.
- ruby: `["ruby", "-e", "puts 'RUBYOK', RUBY_VERSION"]` expecting `"RUBYOK"`.
- php: `["php", "-r", "echo 'PHPOK', PHP_VERSION;"]` expecting `"PHPOK"`.
- rust: `["rustc", "--version"]` expecting `"rustc"`.
Keep the import-the-installed-package layer where a package is named (ruby `gem`, php has no easy import - version marker is the floor).

**Step 4: Run, pass + suite green.**

**Step 5: Commit** `feat(dataset): go/ruby/php/rust runtime probes (converts the 62% unknown-drop)`

## Task 2.2: Resolve ecosystem from RUN lines + FROM, widen the map (TDD)

**Files:** `src/dockermin/dataset/annotate.py` (and `scrape.py` `_ecosystem_from_dockerfile` if shared), `tests/test_annotate.py`

**Step 1: Failing test** - a `debian` base that `pip install`s should resolve to python:
```python
def test_ecosystem_from_run_line_not_just_from() -> None:
    cmd, _ = infer_test_cmd("FROM debian:bookworm\nRUN apt-get install -y python3-pip && pip install flask\n")
    assert cmd is not None and cmd[0] == "python"
```

**Step 2: Run, expect fail** (debian base + pip currently may miss).

**Step 3: Implement** - `infer_test_cmd` already checks `"pip install" in text` etc.; ensure the RUN-line signals (`pip install`, `npm install`, `gem install`, `go build`, `composer`) take precedence so a generic debian/ubuntu/alpine base with an installer resolves to the right ecosystem. Widen `_ECOSYSTEM_MAP` only where needed.

**Step 4: Run, pass + suite green.**

**Step 5: Commit** `feat(dataset): resolve ecosystem from RUN installer lines, not only FROM`

---

# Phase 3: Retarget the scraper at install patterns (§4)

## Task 3.1: Install-pattern queries + client-side COPY/unknown rejection (TDD where pure)

**Files:** `src/dockermin/dataset/scrape.py`, `tests/test_scrape.py`

**Step 1: Failing test** for a new pure helper `_is_self_contained_probeable(dockerfile) -> bool`:
```python
def test_self_contained_rejects_copy() -> None:
    assert _is_self_contained_probeable("FROM python:3.12\nRUN pip install flask\n") is True
    assert _is_self_contained_probeable("FROM python:3.12\nCOPY . /app\nRUN pip install -r reqs\n") is False
    assert _is_self_contained_probeable("FROM scratch\nADD x.tar /\n") is False
```

**Step 2: Run, expect fail.**

**Step 3: Implement:**
- Add `_is_self_contained_probeable`: returns False if the body has a `COPY ` or local `ADD ` (non-URL), or if `_ecosystem_from_dockerfile` returns "unknown". Pure, unit-tested.
- Rewrite `fetch_github_search` `query_variants` to per-ecosystem installer queries (legacy syntax, positive terms only, no NOT/OR):
  - `pip language:Dockerfile filename:Dockerfile size:<2500`
  - `npm language:Dockerfile filename:Dockerfile size:<2500`
  - `gem language:Dockerfile filename:Dockerfile size:<2500`
  - `composer language:Dockerfile filename:Dockerfile size:<2500`
  - `"go build" language:Dockerfile filename:Dockerfile size:<2500`
- After `_fetch_dockerfile`, gate each candidate through `_is_self_contained_probeable` before yielding (client-side filter - the actual win).

**Step 4: Run unit tests + ruff/mypy green.** (Do NOT run the live scrape here - that's Task 4.1.)

**Step 5: Commit** `feat(scrape): install-pattern queries + client-side self-contained filter`

---

# Phase 4: Re-scrape, re-annotate, regenerate variants (produce the data)

These are RUN tasks (local Docker + gh). They produce `data/` (gitignored) - no code commits, but update the journal.

## Task 4.1: Re-scrape with the retargeted fetcher
```bash
PYTHONPATH=src .venv/bin/python scripts/run_scrape.py 2>&1 | tee logs/scrape3.log
wc -l data/raw/candidates.jsonl
.venv/bin/python -c "import json; rows=[json.loads(l) for l in open('data/raw/candidates.jsonl')]; from collections import Counter; print(Counter(r['ecosystem'] for r in rows))"
```
**Checkpoint:** materially more candidates with KNOWN ecosystem and no COPY than the prior 455 (target: a few hundred self-contained probeable).

## Task 4.2: Re-annotate the full candidate set (fresh probes)
```bash
PYTHONPATH=src .venv/bin/python scripts/run_annotate.py --in data/raw/candidates.jsonl --out data/curated/triples.jsonl --workers 4 --target 60 --lockfile logs/annotate.lock 2>&1 | tee logs/annotate3.log
.venv/bin/python -c "import json; rows=[json.loads(l) for l in open('data/curated/triples.jsonl')]; print('bases:', len(rows))"
```
**Checkpoint:** distinct bases >= 30 (goal 30-50). If < 30, this is the decision point for Phase 5 (synthetic app files) - report and ask Vlad before adding that machinery.

Wall-clock: bounded by docker builds; ~4 workers x median 60s. Expect 30-90 min. Use the `--lockfile` to prevent the respawn races from prior sessions.

## Task 4.3: Regenerate variants off the new bases
```bash
cp data/curated/triples.jsonl data/curated/triples.snapshot.jsonl
PYTHONPATH=src .venv/bin/python scripts/synthetic_variants.py --in data/curated/triples.snapshot.jsonl --out data/curated/triples_with_variants.jsonl --variants-per-base 5 --max-bases 50 --lockfile logs/variants.lock 2>&1 | tee logs/variants3.log
.venv/bin/python -c "import json; rows=[json.loads(l) for l in open('data/curated/triples_with_variants.jsonl') if l.strip()]; v=sum(1 for r in rows if r.get('is_synthetic_variant')); print(f'total {len(rows)} = {len(rows)-v} bases + {v} variants')"
```
**Checkpoint:** total rows >= 150 (bases + variants). Cost: ~$0.10/variant via claude -p; budget ~$15-25 for 150-250 variants.

## Task 4.4: Update the dataset-diversity note
Update `docs/journal.md` with the new base count, ecosystem distribution, total rows, and whether Phase 5 was triggered. Commit (on branch).

---

# Phase 5: Synthetic build-context (DEFERRED - only if Phase 4.2 yields < 30 bases)

If re-annotation still falls short, synthesize a trivial app file into the build context to unlock the COPY-using corpus. Out of scope unless triggered - do NOT build this speculatively. If triggered, write a follow-up sub-plan (it's real machinery: parse the COPY target, synthesize `app.py`/`index.js`/`Main.java` + `requirements.txt`/`package.json` matching the RUN line, inject into the temp build context).

---

# Verification (after each code phase)
```bash
.venv/bin/ruff check src tests scripts prime_env
.venv/bin/ruff format --check src tests scripts prime_env
.venv/bin/mypy src tests
.venv/bin/pytest tests/ -m "not docker" --cov=src/dockermin --cov-fail-under=45
```
All green. Branch + PR (main is commit-guard-blocked); CI must pass.

# Risk register
| Risk | Mitigation |
|---|---|
| Broadened probes let a weak ecosystem through that's echo-able | go/php version checks are the floor; they still need the toolchain present (scratch image fails). Acceptable; the tiny-image penalty backstops. |
| Re-annotate wall-clock blows up | --workers 4, --target cap, --lockfile; bounded build timeout |
| Variant gen cost | ~$0.10/variant; cap --max-bases, watch logs/variants3.log; stop at budget |
| Still < 30 bases after Phase 4 | Phase 5 decision point - ask Vlad before adding synthetic-context machinery |
| GitHub rate limit on re-scrape | per-call sleep already in scrape.py; if 403, wait for reset |

# Done means
- `infer_test_cmd` recovers all 11 existing bases + handles go/ruby/php/rust + resolves ecosystem from RUN lines (probe bugs fixed, unit-tested)
- scraper targets install patterns + rejects COPY/unknown client-side
- re-scrape + re-annotate yields >= 30 distinct bases (or Phase 5 triggered with sign-off)
- variants regenerated -> >= 150 total rows
- `make quality` + CI green; journal updated with the new dataset shape
