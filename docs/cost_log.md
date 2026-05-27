# Cost log

Tracks compute spend against the $400 cap. Append per session, do not edit prior entries.

## Triggers (per plan section 8)

- $200 cumulative: pause, confirm pilot signal looks good before continuing
- $400 cumulative: cap to one more run, no further experimentation
- $500 cumulative: stop, ship what we have

## Log

| Date | Session | Provider | Resource | Start | End | Hours | $/hr | Cost | Cumulative |
|------|---------|----------|----------|-------|-----|-------|------|------|------------|
| 2026-05-26 | 1.2-1.6 smoke (dev pod) | massedcompute | 1xH100 80GB PCIe | 2026-05-26T16:47:51Z | 2026-05-26T17:34:50Z | 0.78 | $2.35 | $1.75 | $1.75 |
| 2026-05-26 | Phase 3 pilot + full run (same pod, not terminated between) | massedcompute | 1xH100 80GB PCIe | 2026-05-26T18:38:04Z | 2026-05-27T13:47:00Z | 19.15 | $2.35 | $45.01 | $46.76 |

| 2026-05-27 | Phase 5 eval pod (setup only, terminated before eval run) | datacrunch | 1xH100 80GB SXM5 | 2026-05-27T14:48:25Z | 2026-05-27T15:07:00Z | 0.32 | $3.25 | $0.96 | $47.72 |

**Eval note (2026-05-27):** eval pod fully set up (prime-rl/vllm/transformers/peft + dockermin +
hadolint + slim + DooD + OpenAI key) then TERMINATED without running the eval — Vlad left and an
unattended multi-hour sequential eval would bill idle after completion. Eval is reproducible;
run it supervised (or as a self-terminating job) next session. Adapter already shipped to HF.

**Cost note (2026-05-27):** this single pod ran ~19h and cost ~$45 — far more than the
work warranted. ~10h of it (≈22:15→08:08) was idle/crash-debug limbo across overnight
conversation gaps where the pod billed without productively training (~$20+ wasted).
Lesson: terminate the pod whenever it is not actively training; do not leave it up across
gaps. Full run reached step 151/200 (orchestrator crashed on a zero-advantage batch); the
trained LoRA adapter was salvaged from the filesystem broadcast and pushed to HF.
