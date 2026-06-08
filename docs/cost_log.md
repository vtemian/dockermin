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
| 2026-05-28 | GRPO v1.1 retry (DEAD @ step ~35) | lambdalabs | 1xH100 80GB | 2026-05-28T14:27:50Z | 2026-05-28T16:09:25Z | 1.69 | $4.29 | $7.26 | platform went UNKNOWN, no recovery |
| 2026-05-28 | GRPO v1.1 retry 2 (DEAD @ step ~234) | lambdalabs | 1xH100 80GB | 2026-05-28T16:10:00Z | 2026-05-28T20:39:00Z | 4.48 | $4.29 | $19.22 | platform went UNKNOWN, no adapter pushed |
| 2026-05-29 | GRPO v1.1 retry 3 massedcompute (DEAD @ step ~130 post-resume) | massedcompute | 1xA100 80GB SXM4 | 2026-05-29T07:36:08Z | 2026-05-29T13:00:00Z | 5.40 | $1.79 | $9.66 | disk full from docker accumulation; resumed step 125→129 then UNKNOWN |
| 2026-05-29 | GRPO v1.1 retry 4 (COMPLETED step 250) | massedcompute | 1xA100 80GB PCIe | 2026-05-29T14:48:05Z | 2026-05-30T08:50:00Z | 18.03 | $1.79 | $32.27 | adapter step_250 pushed to HF; manual terminate (watcher missed trigger - trainer logged Step 249 only) |
| 2026-05-30 | Eval pod (qwen_zs partial, DEAD) | massedcompute | 1xA100 80GB | 2026-05-30T08:53:36Z | 2026-05-30T09:38:00Z | 0.74 | $1.79 | $1.34 | died UNKNOWN before dockermin baseline ran |
| 2026-05-30 | Eval pod (qwen_zs at H200, DEAD) | nebius | 1xH200 141GB | 2026-05-30T17:25:13Z | 2026-05-30T19:25:00Z | 2.00 | $2.00 | $4.00 | qwen_zs ran to 36/37 then 2h watchdog terminated |
| 2026-05-31 | Eval pod (dockermin v2 step_250) | crusoe | 1xA100 80GB | 2026-05-31T08:43:54Z | 2026-05-31T11:00:00Z | 2.27 | $1.79 | $4.06 | dockermin v2 step_250 = 21/37 pass (56.8%); successful |
| 2026-05-31 | Eval pod (step_100+step_250 @ T=1.0) | massedcompute | 1xA100 80GB | 2026-05-31T14:55:00Z | 2026-05-31T17:25:00Z | 2.50 | $1.79 | $4.48 | both T=1.0 runs completed before pod died UNKNOWN; results pushed via 60s loop |
| 2026-06-01 | Eval pod (step_100 @ T=0.2) | massedcompute | 1xA100 80GB | 2026-06-01T19:05:30Z | 2026-06-01T20:15:00Z | 1.16 | $1.79 | $2.08 | step_100 @ T=0.2 = 20/37 pass (54.1%); 37 rows pushed before pod died UNKNOWN |
| 2026-06-08 | Gemma 4 12B-it zero-shot probe | massedcompute | 1xA100 80GB SXM4 | 2026-06-08T18:45:04Z | 2026-06-08T19:46:00Z | 0.99 | $1.23 | $1.22 | 30/37 pass (81.1%), reduction/pass 38.2%, 13.67 GB saved; first model swap to clear MDE; Cost = actual wallet delta ($80.57 → $79.35), Hours back-solved from billing |

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
