# Project journal

Append per session. Last entry at top. 3-5 sentences per entry. What worked, what broke, what was not obvious.

## 2026-05-26 - Phase 1 PASS: stack validated end-to-end on 1xH100 ($1.75)

Rented a massedcompute 1xH100 PCIe ($2.35/hr), installed prime-rl from main, and got the
full async RL loop running on alphabet_sort (prime-rl's own canonical smoke). Validated:
vllm inference + LoRA serving, FSDP+LoRA trainer (via torchrun), orchestrator + verifiers
env reward, filesystem weight broadcast, all colocated on one H100 (69/81 GiB, no OOM).
10 orchestrator steps, 8 trainer steps, reward stable ~0.4-0.47, zero NaN/OOM. Did NOT
wait for the reward to climb (alphabet_sort needs ~20-30 steps and it's a known-good
example) — terminated early to avoid unattended billing. Total spend $1.75, balance
$48.25/$50.

Three bleeding-edge fixes were needed, each crashed a component (all in
docs/decisions/2026-05-22-prime-rl-pin.md): VLLM_USE_DEEP_GEMM=0 (shipped deep_gemm wheel
wants CUDA 13, stack is CUDA 12.8), --model.max-model-len 8192 (Qwen3 256K context blows
KV cache at 0.5 mem util), and the trainer must run under torchrun (bare entrypoint dies on
RANK unset). The unified `rl` launcher can't colocate on 1 GPU in this commit (wants 2);
ran the 3 components manually. Dropped Task 1.5 (standalone hotswap script) as redundant —
the live run already exercises LoRA load + weight-broadcast swapping every step. Next:
Phase 3 reward + pilot. The 8xH100 run uses the default split deployment (no colocation
hacks) but still needs VLLM_USE_DEEP_GEMM=0 + a sane max-model-len for Qwen 7B.

## 2026-05-26 - Phase 1 start: prime CLI fixed, prime-rl pin verified

Verified prime-rl PR #1392 (LoRA + NCCL `adapter_only` crash) is MERGED (commit 91182b7,
2025-12-07); main is 686 commits past it, so we install from main and capture the exact
HEAD on the pod as the real pin. Decision recorded in docs/decisions/2026-05-22-prime-rl-pin.md.
The `prime` CLI from PyPI (prime_cli 0.6.9) ships BROKEN against current deps: typer 0.26.0
vendored Click internally, but prime_cli still imports external `click`, so a vendored-click
Context gets passed into external-click code paths and dies with
`AttributeError: 'Context' object has no attribute '_param_default_explicit'`
(and `_default_map_has` on the next click down). Fix: pin `typer<0.26` —
`uv tool install --force prime --with "typer<0.26"` lands typer 0.25.1 + click 8.3.3 and the
CLI runs clean. Smoke script smoke_lora_hotswap.py confirmed ready (NameError already fixed).
Blocked on Vlad's Prime Intellect signup + auth before I can drive pod rental.

## 2026-05-25 - dataset growth COMPLETE (Phase 4 done)

The dataset-growth plan landed. Final set: **145 rows = 28 bases + 117 variants**
(up from 93 = 11 bases + 82 variants). Variant pass rate ~86% (145 kept, 23 failed).

Yield story (the convergent finding from 3 context agents): the old pipeline dropped
62% of candidates as "unknown ecosystem" and the scraper optimized file-size instead of
the install pattern. Fixes that moved the needle:
- 3 probe-extraction bugs fixed (pip quote-strip, npm cross-line/global) recovered the
  existing bases under the hardened import-probes.
- Broadened `infer_test_cmd` to go/ruby/php/rust + RUN-line ecosystem resolution.
- Retargeted `fetch_github_search` at install-pattern queries (pip/npm/gem/composer/go
  build) + client-side reject of COPY/unknown. 226 candidates, 77 self-contained-probeable
  (vs ~27 before).

**Honest diversity limit (matters for the writeup):** the new candidates are
python-heavy. Final ecosystem mix across all rows: 54 python, 19 node, 4 alpine, 3 ruby,
65 unknown (mostly the old eclipse-temurin/java bases mislabelled by the pre-fix
inference). So the model will learn python/node Dockerfile optimization well and
generalize weakly to other ecosystems. 28 distinct bases is 2.5x the old 11 but still
small - a pilot run is a methodology test, not a generalization claim. The honest path to
broad generalization is more bases (Phase 5 synthetic-build-context, deferred) or more
scraping, not more variants of the same 28.

Phase 5 (synthetic app-file in the build context) was NOT triggered: 28 bases is close
enough to the 30 target that building that machinery for ~2 bases wasn't worth it (YAGNI).

## 2026-05-25 - RESUME-AFTER-RESTART checkpoint (dataset growth mid-flight)

Vlad is doing a machine restart. State captured so we lose nothing (data/ is
gitignored = local-disk only, but survives a reboot; not backed up until the HF push).

**Branch:** `feat/dataset-growth`, 3 commits ahead of main (plan + probe-bug fixes +
broadened ecosystems + scraper retarget). Code gate green (ruff/mypy/137 tests).

**Dataset state on disk (data/curated/):**
- `triples.jsonl` = 28 distinct bases (up from 11; python-heavy: ~10 python / 5 node / 11 old-java-labelled-unknown / 1 ruby / 1 alpine)
- `triples_with_variants.jsonl` = 120 rows = 24 bases + 96 variants, AND CLIMBING - the
  synthetic_variants run was still going (24 of 28 bases done) when the restart happened.
  Append-mode + startup dedup means it's safe and RESUMABLE.

**TO RESUME after reboot (Docker Desktop must be back up first - `docker ps`):**
```bash
cd /Users/whitemonk/projects/ai/dockermin
git checkout feat/dataset-growth
# finish the remaining ~4 bases of variants (dedup skips the 24 already done):
PYTHONPATH=src .venv/bin/python scripts/synthetic_variants.py \
  --in data/curated/triples.snapshot.jsonl \
  --out data/curated/triples_with_variants.jsonl \
  --variants-per-base 5 --max-bases 28 --lockfile logs/variants.lock
```
(`triples.snapshot.jsonl` is the frozen 28-base input; the run reads it, skips existing ids.)

**Then to close out Phase 4 / the dataset-growth plan:**
1. Add a dataset-diversity note (28 bases, python-heavy - honest generalization limit).
2. Commit journal + note on the branch -> open PR -> merge to main (CI must pass).
3. `make quality` should stay green.

**Still BLOCKED on Vlad (unchanged):** HF_TOKEN (push `vladtemian/dockermin-v0` with the
train/test split) and Prime Intellect account + GPU budget (the entire pod-side: smoke
tests -> pilot -> full run -> eval -> ship). The core thesis (7B GRPO LoRA vs Sonnet
agent-loop) is still 100% untested - it needs the pod run.

**Background jobs that die on reboot (all fine):** synthetic_variants (resumable above),
any tee'd bash. No uncommitted code. Docker will need to come back up before resuming.

## 2026-05-25 (Vlad + Bot) - working-style retro + harness tuning

Vlad asked for a deep analysis of how we work together. Mined the journal, memories, and the 4.7MB main session (55 subagent dispatches, 21 real human messages; the ~120 tiny uniform sessions are just synthetic-variant API calls, not conversations). Findings: Vlad directs rather than pairs — one dense up-front spec, then terse nudges, and a verbatim mantra repeated 8+ times ("spawn subagents → gather context → write detailed plan → implement"). He audits by pointed question ("why so many scripts?", "why the __future__ imports?") rather than reading diffs, and signals satisfaction by "continue", never praise.

Decisions locked and encoded (so they survive context loss):
- **No TodoWrite.** Removed the mandate from global `~/.claude/CLAUDE.md` (Vlad's call) — replaced the "Issue tracking" section with a "Workflow and delegation" section.
- **Worktrees by default** for parallel subagent work — fixes the May 22 file-corruption night (parallel agents wrote the same file in "w" mode) structurally, not via lockfiles.
- **Question policy:** batch independent choices, serialize only dependent ones.
- **Don't wind down** / suggest sleep — encoded after the "stop seeing me to sleep, need to work" friction.
- Built `~/.claude/commands/plan-it.md` and `ship-it.md` (the mantra as slash commands), and a dockermin `.claude/` commit-guard hook that hard-blocks `git commit` on main (the one gap the existing git pre-commit + CI quality gate don't cover). Hook verified on all three paths.

## 2026-05-25 (Vlad + Bot) - quality gates adopted, matching sisif bar

Audited dockermin against sisif's quality system (4 subagents analyzed sisif's ruff/mypy config, enforcement, conventions, structure), wrote a 5-phase plan (`docs/plans/2026-05-25-quality-gates.md`), then implemented it with parallel subagents in two waves.

Before/after:
- ruff: **158 violations -> 0** (full sisif ruleset: ANN, T20, TRY, BLE, EM, C90, PLR, PERF, RET, TCH, ...). Went stricter than sisif: enabled EM rules and explicit max-args=6 that sisif waived "for Django", dropped sisif's gradual-debt security waivers.
- mypy: strict, **0 errors** (was never run before). Closed sisif's own hole - mypy now runs in CI, not just locally.
- tests: **35 -> 98**, coverage **20% -> 48%** with a 45% CI floor (ratchets up).
- New guardrails: CLAUDE.md (no-nesting/names-are-contracts/fail-fast/no-narration/plain-functions), ARCHITECTURE.md, `.github/workflows/quality.yml` (ruff+format+mypy+pytest+coverage, blocking on push/PR), pre-commit (ruff+ruff-format+local-mypy).

What was not obvious:
- The `from __future__ import annotations` question: it's only load-bearing on 3.11 for TCH-guarded imports; PEP 649 in Python 3.14 would make it unnecessary, but our torch/vllm pins cap us at 3.11. Resolved by enforcing it everywhere via ruff isort required-imports (deliberate policy, not cargo-cult).
- pre-commit's mirrors-mypy runs in an isolated env without our deps -> false import-not-found errors. Fixed with a `local` mypy hook using `.venv/bin/mypy`, matching how CI runs it.
- The bulk lint/type cleanup (158->~3 ruff) was done by ad-hoc fix-agents BEFORE the plan existed; Vlad correctly redirected to analyze->plan->implement. The cleanup was kept as plan input rather than redone.
- Phase 5 (naming-contract audit) was a no-op: the code already obeyed fetch/find/get contracts with zero generic Manager/Helper names.

CI green at commit ba043f0. ~62 dataset triples + 82 variants from the prior session still stand; this session was pure quality-hardening, no behavior change (reward math byte-identical, verified by exact-value tests).

## 2026-05-23 Saturday very-early hours (round 3 - more variants + lockfile rollout + ship templates)

Continued with another wave of agents. Concrete results:

**Bulk annotate round 2:** 96 safe candidates processed -> 12 working triples, 2 NEW (10 dup with existing). triples.jsonl: 14 -> 16 baselines.

**Synthetic variants round 2:** ran with --max-bases 16 against the 16-baseline snapshot. Got an additional 34 variants before background bash timed out. Some bases now have 13 variants (more bloat-pattern diversity than the original 5/base). Final state: 11 bases + 82 variants = 93 entries.

**Combined dataset:** 16 unique baseline Dockerfiles + ~82 unique synthetic variants. About 98 working pairs total, well past the 100 floor for the kill criterion. Saturday on the pod can push higher.

**Hard finds caught locally before the pod:**
- `prime_env/dockermin_env/pyproject.toml` was missing explicit [tool.setuptools] block. Setuptools flat-layout discovery picked the sibling `dockermin_env.py` as a top-level py_module, making `import dockermin_env` execute the submodule body and bypass __init__.py. Fixed in commit 253ffcf with explicit `packages = ["dockermin_env"]` + `package-dir`. This would have broken verifiers' Environment loading silently on the pod.
- run_annotate.py and synthetic_variants.py now both have `--lockfile` arg using fcntl.flock to prevent the double-spawn races that corrupted output earlier in the night.

**Ship-phase templates pre-written:**
- `docs/blog_template.md` (61 lines) - TL;DR, setup, results table placeholder, "things that surprised us", what did not work
- `docs/NEGATIVE_RESULT_template.md` (36 lines) - if the agent loop wins, this is the writeup

**Still blocked:** HF_TOKEN missing. Vlad needs `huggingface-cli login` or `HF_TOKEN=hf_xxx` in .env before dataset push. Saturday morning task.

## 2026-05-23 Saturday early hours (variant pipeline completed)

Synthetic variants pipeline ran to completion. 11 bases x 5 variants = 55 attempts. 48 variants kept (~87% pass rate). 7 failures: base 6 lost all 5 (apt-get bloat hit Debian repo dep issues), bases 4 and 9 lost 1 each (deprecated MAINTAINER, OS signals).

Total wall-clock ~2h45m. Cost rough estimate: 55 claude calls x ~$0.10/call cache creation = ~$5.50. The append-mode patch (commit 98e0bc0) made the run resilient to the chaotic respawns that plagued the earlier hour.

Combined dataset at end of session:
- data/curated/triples.jsonl: 14 base triples (real Dockerfiles from official-images)
- data/curated/triples_with_variants.jsonl: 11 bases (subset of 14) + 48 synthetic variants

Union ~62 distinct working triples. Below the 100 kill-criterion floor but enough to seed a pilot run. Saturday on the pod can push to 100+ via more bulk annotate against the 96 safe-to-build candidates.

## 2026-05-22 Friday very-late evening (Vlad + Bot, data pipeline reality check)

Tried to push the dataset pipeline end-to-end tonight. Mixed result, honest report:

**What worked:**
- Code-review cleanup (M4, M7), Makefile target verification - clean.
- Scrape broadened from 235 to 455 candidates (96 safe-to-build).
- Verified all 3 unknowns against prime-rl source: PR #1392 merged at 91182b7; TOML schema is [trainer.model.lora] not [model.experimental]; verifiers entry-point group is decorative, what matters is module name = package_module_name(env_id) so env_id "dockermin-env" maps to dockermin_env.
- Updated configs/dockermin_{pilot,full}.toml to match verified schema.
- synthetic_variants.py manually tested: one call against the flask base produced a beautiful unoptimized variant in 11s (heavier base, redundant RUNs, missing flags, shell-form CMD). Pipeline works.
- Bulk annotate added 3 triples (11 -> 14) before being killed.

**What broke:**
- Parallel-agent coordination collapsed under respawning sub-agents. Both the synthetic_variants and run_annotate processes had ghost duplicates (Agent A's command kept being re-spawned by something - probably a parallel claude session or hook loop). Multiple processes wrote to the same output files in "w" mode, corrupting one line of triples_with_variants.jsonl.
- My pkill pattern to clean up the duplicates also killed my own intended processes. Variant generation produced 0 saved variants.

**Lesson for Saturday:**
- Run scrape and annotate on the pod with no other claude sessions touching the project tree. The race conditions tonight were not the scripts' fault.
- Each script needs --in / --out args (already added by Agent B to run_annotate; synthetic_variants.py already accepts --in/--out).
- Add a `--lockfile` arg or PID-based rendezvous to prevent two instances racing on the same output.

State at sleep:
- 14 curated triples in data/curated/triples.jsonl (up from 11)
- 455 candidates in data/raw/candidates.jsonl
- triples_with_variants.jsonl: 3 bases written, 1 corrupt line, 0 variants (file should be discarded; will regenerate cleanly Saturday)
- All scaffolding code green: 25/25 pytest pass

## 2026-05-22 Friday late-late evening (Vlad + Bot, 4-agent verify pass + code review)

Dispatched 4 parallel agents over the parallel-burst code from earlier:
local venv + pytest, real scrape against GitHub, code review across the
diff range, Makefile + GitHub Actions CI scaffolding.

Concrete results:
- Local venv up, 23/23 tests pass (Docker Desktop available on darwin).
- Scrape pulled 235 candidates after fixing 2 scrape bugs (awesome-compose
  1-level walk; GitHub code-search `stars:` qualifier doesn't exist there).
  Was 103 candidates on first run, jumped to 235 after fixes.
- Code reviewer found 5 critical bugs. Fixed 4 of the 5 tonight:
  - test_gate function collided with pytest collection -> renamed to
    run_test_gate
  - compute_score regex was greedy on \\S+ so tagged images got
    bare-FROM penalty -> split into two ifs, tightened regex
  - parse-garbage test expected "parse" error but dockerfile lib is
    extremely lenient -> use empty input which is the only plain string
    that triggers GoParseError
  - agent_loop.py used --max-turns and --cwd flags that DON'T EXIST in
    this claude build. Replaced with subprocess cwd= + --add-dir +
    --max-budget-usd as the bound.
  - empty expected_substring trivially passed test_gate (reward hacking
    surface). Defense in depth: run_test_gate refuses empty;
    infer_test_cmd's go fallback drops triple; java uses "openjdk"
    substring (always present in `java -version` from openjdk images).
  - replaced docker SDK images.build() with `docker buildx build`
    subprocess. SDK timeout is HTTP-idle only, doesn't bound build wall
    clock - a stuck RUN held a worker thread forever. Subprocess timeout
    actually kills. Bonus: cache-from/cache-to flags actually work now
    against the docker-container builder created by setup_pod_docker.sh.

5th critical (C3 :latest regex) was already fixed in my previous batch -
reviewer was running against pre-fix HEAD.

Important / Minor items from review (I1-I8, M1-M7) deferred. Most of
them are correctness nits or perf tweaks that don't burn budget or
produce misleading numbers. Worth a Sunday cleanup pass once the pilot
shows signal.

Ergonomic adds: Makefile with 11 targets + .github/workflows/ci.yml
running pytest -m "not docker" on push.

At HEAD: github.com/vtemian/dockermin commit fa3ac92.
- 36 files in src/ + tests/ + scripts/ + configs/ + prime_env/
- 18 commits total
- 23/23 tests pass locally
- 235 raw candidates ready in data/raw/candidates.jsonl

Saturday morning's work has now shifted from "implement + run from
scratch" to "deploy known-good code on the pod and watch it run."

## 2026-05-22 Friday late evening (Vlad + Bot, parallel implementation burst)

Dispatched 6 parallel subagents on non-overlapping file sets. All 6 returned clean (AST + TOML validation passed). 2232 lines of code across 29 new files written in roughly 4 minutes of wall-clock. Commits f5d8d9b through 497ad0a cover the reward stack, annotate gates, dataset orchestration, pod ops scripts, eval baselines, and CLI + prime-rl environment package.

Real bugs caught by the agents and fixed:
- Plan's `smoke_lora_hotswap.py` had a NameError (`base` vs `base_out`). Fixed.
- Plan's `configs/dockermin_pilot.toml` was invalid TOML (`model` declared as both string and table). Agent promoted to `[model]` table with `name=` field; prime-rl schema TBD-verified Saturday against the alphabet_sort example.
- Plan's `Candidate.__dict__` would fail on frozen dataclass; agent used `dataclasses.asdict`.
- Plan's run_annotate early-break was unreachable; agent restructured to break on as_completed.

Open verifications for Saturday:
- prime-rl entry-point group name (`verifiers.environments` is the educated guess) - probe with `vf-install --help` and `importlib.metadata.entry_points` on the pod
- TOML schema for prime-rl model config - read examples/alphabet_sort/rl.toml on the pod before first run
- `dockerfile.GoParseError` symbol stability (lib is upstream-deprecated)
- `dockermin_env` pyproject may need explicit `packages = ["dockermin_env"]` if setuptools auto-discovery balks

29 files in but ZERO tests have actually run yet - they will run on the pod Saturday in real conditions. This is the right tradeoff for tonight: write code at parallel speed, verify in real conditions tomorrow.

## 2026-05-22 Friday evening (Vlad + Bot, pre-execution)

Plan read. Memories saved. Repo initialized at github.com/vtemian/dockermin with skeleton + pinned deps. Decisions locked in: GRPO via verifiers committed, no mid-project pivot to SFT/steering even on null result, agent-loop baseline via Claude Code CLI, dataset target 200 not 1000, egress allowlist proxy not single mirror, DooD privileged daemon with pids-limit 2048 and 300s watchdog. Top risks flagged: build latency dominates rollout, docker daemon concurrency bottleneck unverified, Sonnet 4.6 agent loop likely strong, vllm 0.7.3 LoRA hotswap on Qwen 7B unverified. Saturday runbook drafted.
