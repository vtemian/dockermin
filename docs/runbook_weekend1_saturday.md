# Weekend 1 Saturday runbook (v2, updated 2026-05-22 late evening)

Goal of the day: prove the stack works end-to-end on the pod before paying for big compute. The code is already written and tested locally (25/25 pytest green on darwin with Docker Desktop). Today's work is **deploy, validate, smoke-test**, not implement.

## Pre-flight (do before opening the laptop)

- [ ] Coffee
- [ ] Phone on do-not-disturb, no Opnble, no Twitter

## Phase 0: credentials (10 min, no GPU spend)

- [ ] Open https://app.primeintellect.ai, create account, generate API key
- [ ] `cd /Users/whitemonk/projects/ai/dockermin && cp .env.example .env && $EDITOR .env`
- [ ] Fill: `PRIME_INTELLECT_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY`
- [ ] `git check-ignore .env` should print `.env` (confirm it stays out of git)

## Phase 1: rent H100, install, validate (60 min, ~$2 spend)

- [ ] `prime availability --gpu-type H100_80GB --gpu-count 1` to find a provider
- [ ] `prime pods create --gpu-type H100_80GB --gpu-count 1 --provider <id>`
- [ ] SSH in, `git clone git@github.com:vtemian/dockermin.git`
- [ ] On pod: `make install` (Makefile target installs deps including prime-rl)
- [ ] `wandb login` and `huggingface-cli login`

**Checkpoint A:** `python -c "import verifiers, vllm, peft, prime_rl; print('OK')"` returns OK.

**Checkpoint B (verified-Saturday items from docs/research_findings_2026-05-22.md):**
- [ ] `cd /path/to/prime-rl && git log --oneline | grep 91182b7` confirms PR #1392 fix is in HEAD
- [ ] `python -c "from prime_rl.configs import RLConfig; RLConfig.from_toml('configs/dockermin_pilot.toml')"` validates the config schema. If it fails, diff against `prime-rl/examples/alphabet_sort/rl.toml` and fix.
- [ ] `cd prime_env/dockermin_env && pip install -e . && python -c "import dockermin_env; print('module OK')"` confirms the verifiers Environment package installs and imports.

If any checkpoint fails, STOP. Do not burn more GPU time debugging silently. Ping the Prime Intellect Discord support channel or fall back to a known-good prior commit.

## Phase 2: LoRA hotswap risk check (30 min on the same 1xH100, ~$0.75)

- [ ] `python scripts/smoke_lora_hotswap.py`

Expected: prints `PASS: base != A != B, A reproducible after swap`. Latencies: base ~3s first call, a1 first load ~500ms, a2 (warm swap) <100ms.

Hard exit: any FAIL print, OR swap latency > 1s. If fails, file vllm issue with repro, and fall back to NCCL weight-broadcast (slower training but known-good).

## Phase 3: alphabet_sort smoke (60 min on 1xH100, ~$1.50)

Sanity-check that prime-rl + verifiers + LoRA hotswap all wire together end-to-end on a known task before pointing them at our dockermin reward.

- [ ] `prime env install primeintellect/alphabet-sort`
- [ ] `uv run rl @ <prime-rl>/examples/alphabet_sort/rl.toml --model.experimental.lora --model.max_lora_rank 32 --wandb.project dockermin --wandb.name alphabet-sort-smoke`

Watch wandb. Expected: reward climbs above 0.5 by step 50. If flat at 30, kill and debug.

## Phase 4: shutdown + log (10 min)

- [ ] `prime pods terminate <pod_id>`
- [ ] Update `docs/cost_log.md` with start/end/hours/cost
- [ ] 3-line journal entry: what worked, what broke, what was not obvious

Pod down by lunch. Total spend Saturday: ~$5.

## What you are NOT doing today

- Curating the full dataset (Sunday)
- Writing reward function (already written, tested)
- Building dataset annotator (already written, tested)
- Implementing baselines (already written)

If you finish Phase 4 by lunch with all checkpoints green, you have a green light to start Sunday's dataset work immediately.

## Sunday preview

- Bulk-annotate the 235 candidates already in `data/raw/candidates.jsonl` (regenerate on pod if needed). Expect ~30-50 working triples from official-images Dockerfiles that don't need build context.
- Run `scripts/synthetic_variants.py` to generate 5 unoptimized variants per working base via Sonnet 4.6 (~$2 API). Yields 100-250 triples total.
- Push to HF as `vtemian/dockermin-v0`.
- 50-step pilot GRPO run on the curated set, watch wandb.

## What we already verified Friday night (locally)

- 25/25 pytest pass with Docker Desktop on darwin (M-series Mac)
- `docker buildx build` subprocess pipeline working
- prime-rl TOML schema verified against alphabet_sort example
- PR #1392 confirmed merged at SHA 91182b7
- verifiers entry-point resolution is by module name; env_id "dockermin-env" maps to `dockermin_env` module
- `dockerfile.GoParseError` symbol confirmed stable in v3.3.1

## Hard exit triggers for the day

- Phase 1 deps install fails, 30 min escalation past: STOP
- Phase 2 LoRA hotswap shows FAIL: STOP, file issue
- Phase 3 alphabet_sort reward flat at step 30: STOP, debug locally before more GPU
- Cumulative spend > $20: STOP and verify no leaks
