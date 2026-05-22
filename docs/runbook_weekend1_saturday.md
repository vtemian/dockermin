# Weekend 1 Saturday runbook

Goal of the day: prove the stack works end-to-end on a trivial problem before touching Dockerfiles. If the stack does not run cleanly by lunch, the weekend is at risk and we reassess.

## Pre-flight (do before opening the laptop)

- [ ] Coffee
- [ ] Phone on do-not-disturb, no Opnble, no Twitter
- [ ] `~/projects/dd/tools/focus-blocker/focus-blocker.sh on` (optional)

## Phase 0: credentials (10 min, no GPU spend)

- [ ] Open https://app.primeintellect.ai, create account if needed
- [ ] Generate API key, save to `.env` as `PRIME_INTELLECT_API_KEY=...`
- [ ] Confirm billing setup. Set a soft cap reminder in your head: $200 = pause and verify pilot signal
- [ ] Fill remaining `.env` fields: `HF_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY`
- [ ] `cd /Users/whitemonk/projects/ai/dockermin && cp .env.example .env && $EDITOR .env`
- [ ] Verify `.env` is gitignored (`git check-ignore .env` should print `.env`)

## Phase 1: rent H100 and bring up env (45 min, $1.49/hr starts now)

- [ ] Prime Intellect dashboard, "Pods" or "On-demand" tab, select 1xH100 80GB SXM
- [ ] Note rental start time in `docs/cost_log.md` (create the file)
- [ ] SSH in
- [ ] `python3.11 -m venv .venv && source .venv/bin/activate`
- [ ] Clone this repo
- [ ] `pip install -e .` (will install pinned deps from pyproject.toml)
- [ ] `wandb login` with key from `.env`
- [ ] `huggingface-cli login` with HF_TOKEN

**Checkpoint:** `python -c "import verifiers, vllm, torch, peft, trl; print(verifiers.__version__, vllm.__version__, torch.__version__)"`
Expected: `0.1.4 0.7.3 2.5.x` with no traceback.

If deps fail to resolve: STOP. Open Prime Intellect Discord support channel. Do not debug for more than 30 min before escalating. This is the cheap kill criterion.

## Phase 2: GSM8K smoke test (90 min)

This proves verifiers + vLLM + LoRA serving + wandb all work. Reference: https://verifiers.readthedocs.io/en/latest/components.html

- [ ] Clone https://github.com/PrimeIntellect-ai/verifiers somewhere outside the dockermin repo
- [ ] Find the GSM8K example (likely `examples/gsm8k/` or in the README)
- [ ] Adapt it minimally: use Qwen 2.5 Coder 7B Instruct as the base model (not whatever they default to). LoRA config: r=32, alpha=64, modules per plan.
- [ ] Run for 10-20 steps only. Watch wandb.

**Checkpoint:** Reward curve trends up (does not need to converge, just needs to move). vLLM serves rollouts. No CUDA OOM. wandb logs reward, loss, kl. Adapter saves every N steps.

If reward is flat or NaN after 20 steps with the default GSM8K config: something is wired wrong. Do not proceed to Dockermin reward.

## Phase 3: LoRA hotswap risk check (45 min)

This is the highest-risk-not-yet-verified piece. If hotswap is broken with vllm 0.7.3 + Qwen 7B, the whole pipeline is dead.

Test:
- [ ] Train two distinct tiny LoRAs (rank 16, 30 steps each, different seeds)
- [ ] Bring up vLLM with `enable_lora=True max_loras=4 max_lora_rank=32`
- [ ] Call OpenAI-compat endpoint with `extra_body={"lora_request": adapter_A}` on a prompt
- [ ] Same prompt with `lora_request": adapter_B`
- [ ] Confirm completions differ (they should, with two different LoRAs)
- [ ] Time the swap. Plan says sub-ms; verify <100ms in practice

If swap silently uses one adapter regardless of which was requested, or swap takes >1s, file an issue with verifiers/vllm immediately and decide whether to wait for fix or fall back to no-hotswap mode (kills async overlap, doubles training time).

## Phase 4: shutdown and log (10 min)

- [ ] Terminate the H100 pod (do not leave it running)
- [ ] Log spend in `docs/cost_log.md`
- [ ] `git add docs/ && git commit -m "weekend 1 sat: smoke tests passed"`
- [ ] Push to github

## Hard exit triggers for the day

- Deps do not resolve, 30 min escalation window past: STOP, ping Prime Intellect support, do not spend more compute today
- GSM8K reward flat after 20 steps with default config: STOP, debug locally, do not run any further GPU steps
- LoRA hotswap broken: file issue, decide async fallback, do not start Sunday on dataset until decision is made

## What you are NOT doing today

- Writing the dockermin reward function (Sunday afternoon)
- Scraping Dockerfiles (Sunday morning)
- Touching docker daemon on training node (Sunday or weekend 2)
- Synthetic variant generation via Claude API (weekend 2)

Stay narrow today. Today is just "does the framework work."

## Dinner check-in

By 8pm, write 3 lines in `docs/journal.md`:
- What worked
- What broke
- What you learned that was not obvious

If by dinner none of the phases passed, weekend 1 Saturday is over. Sleep on it. We reassess Sunday morning.
