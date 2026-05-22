# Project journal

Append per session. Last entry at top. 3-5 sentences per entry. What worked, what broke, what was not obvious.

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
