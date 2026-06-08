# Gemma 4 12B-it Zero-Shot Probe — Result and Next Move

**Date:** 2026-06-08
**Pod:** massedcompute A100 80GB SXM4 — Datacrunch unavailable at launch
**Cost:** ~$2.50 (pod up ~2h including install debugging; eval itself ran ~25 min)
**Probe results JSONL:** `vtemian/dockermin-gemma4-12b-it-probe/eval/results.jsonl` (mirror at `data/eval/gemma_zs_probe.jsonl`)
**Plan:** [docs/plans/2026-06-08-gemma4-zero-shot-probe.md](../plans/2026-06-08-gemma4-zero-shot-probe.md)
**PR:** [#22](https://github.com/vtemian/dockermin/pull/22)

## Numbers (n=37 holdout, `T=0.2`, `max_new_tokens=1024`)

| Model | Pass rate | Mean reduction\|pass | **Bytes saved (sum)** |
|---|---|---|---|
| qwen_zs (per leaderboard.md) | 56.8% | 32.1% | — |
| qwen_zs (per controlled re-eval, plan ref) | 56.8% | 60.6% | — |
| dockermin v2 step_250 | 56.8% | 75.3%–85.7% (per measurement) | **7.05 GB** |
| dockermin v3 abl_c step_244 | 59.5% | 88.3% | 6.08 GB |
| sonnet_zs (ceiling reference) | 91.9% | 38.9%–59.6% (per measurement) | 4.84 GB |
| **gemma_zs (this probe)** | **81.1%** (30/37) | **38.2%** (over 30 passes) | **13.67 GB** |

The pass-rate delta vs `qwen_zs` is **+24.3 pp**, which **exceeds the n=37 minimum-detectable-effect (~23 pp at 80% power)**. This is the first model swap in the project to clear the statistical floor.

## Failure-mode breakdown

| Bucket | Count |
|---|---|
| pass | 30 |
| build_fail (gradle/php/node not present in substituted base) | 4 |
| parse_fail (no fenced dockerfile block) | 3 |
| test_fail | 0 |
| hallucinated FROM tag | **0** |

Three notable patterns:
- **Zero hallucinated FROM tags.** Gemma consistently picks valid Docker Hub tags (`python:3.12-slim-bookworm`, never the inverted `bookworm-slim`; `python:3.9-slim`; `eclipse-temurin:25-jdk` directly; `node:14-alpine`). The 7 Qwen failure modes that v3 forensics flagged are absent.
- **Gemma is conservative on PHP** — five `php:8.5-apache-trixie` rows pass but at ~20% reduction (it keeps the original base).
- **Three parse failures** are real — Gemma sometimes returns prose without a code fence. Easy reward-signal fix in any GRPO follow-on.

## Branch fired

**Branch B** — strict decision-tree application.

Decision rule from the plan, applied verbatim:
```
delta_pass = pass_rate(gemma_zs) - 56.8 = +24.3 pp  ≥ +5 pp
reduction_on_pass(gemma_zs) = 38.2%  < 50%
→ Branch B (Gemma is too timid; GRPO can't recover reduction ceiling)
```

## Tension worth surfacing

The decision tree was designed around `reduction|pass` as the proxy for "headline metric." But the **total bytes saved** picture inverts the conclusion:

| Model | Bytes saved | vs gemma_zs |
|---|---|---|
| gemma_zs (untrained) | **13.67 GB** | 1.00× |
| dockermin v2 step_250 (10h GRPO) | 7.05 GB | 0.52× |
| dockermin v3 abl_c step_244 (~10h GRPO) | 6.08 GB | 0.44× |
| sonnet_zs (API baseline) | 4.84 GB | 0.35× |

On the project's actual user-facing outcome (image-bytes shipped after rewrite), **Gemma 4 12B-it zero-shot beats every model we've trained or evaluated to date — by ~2×**.

The shape of the result:
- Gemma trades aggression-per-row for accuracy-across-rows.
- It passes 30/37 (81%) where Qwen-trained passes 21/37 (57%).
- Even at lower per-row reduction (38% vs 85%), the larger denominator dominates.

The Branch B trigger ("Gemma is too timid; GRPO can't recover ceiling") assumed reduction-on-pass is the bottleneck. But Qwen-GRPO already demonstrated +25-50 pp reduction lift from base; if Gemma responds similarly, projected Gemma-GRPO ceiling is 63%-88% reduction at 81%+ pass rate, which would dominate every existing model on both axes simultaneously.

## Recommendation (separate from the strict tree)

The plan's decision tree is binding by its own terms ("Do not retroactively soften the thresholds after seeing the numbers"). I'm reporting Branch B as the literal call.

**But** the qualitative result is the strongest pro-Gemma signal possible: cleared MDE, zero hallucinated tags, 2× total bytes saved, well-understood remaining failure modes (3 parse failures + 4 base-substitution build breaks — both addressable). If the headline metric had been "bytes saved on the holdout" instead of "reduction-on-pass," the same tree would have fired Branch A unambiguously.

**Decision for Vlad:** the strict tree says B. The bytes-saved metric and the qualitative pattern point hard at A. Your call.

- **Branch A path:** draft `docs/plans/2026-06-09-v4-gemma-grpo.md` for a full GRPO arc on `google/gemma-4-12B-it` as base. Expected cost ~$60-80 for one training arc + eval.
- **Branch B path (literal tree):** tag-RAG over Docker Hub manifest on the existing Qwen base. Hold Gemma swap.
- **Branch C path:** ruled out — `gemma_zs >> qwen_zs` clears MDE.

## Per-row qualitative evidence

Sample of Gemma's correct tag choices (these are the ones Qwen hallucinated in v3 forensics):
- `python:3.12-slim-bookworm` (right order; Qwen produced `python:3.12-bookworm-slim`)
- `python:3.9-slim` (clean, valid)
- `eclipse-temurin:25-jdk` (Qwen produced the non-existent `25-jdk-slim`)
- `node:14-alpine`

Sample of Gemma's failure shapes (build-side, not tag-side):
- `eclipse-temurin:25-jdk-resolute` + gradle steps → `gradle: not found` (temurin doesn't ship gradle)
- `node:14.18.1` + alpine package names that aren't in debian-based node:14 → `firewalld beep imagema...` parse error
- `php:8.5-apache-trixie` + opcache config → multi-line RUN syntax issue

## Install gotchas worth remembering (now in `scripts/pod_ops/install_gemma_probe.sh`)

These bit me on launch day and are now baked into the install script. Recording so the next operator knows what they're inheriting:

1. **massedcompute ubuntu_22_cuda_12 ships python3.10 only.** No python3.11; the project pin is 3.11. Use `--ignore-requires-python` on the editable install — eval-only code paths don't need 3.11.
2. **`python3.10-venv` apt package is missing.** `python3 -c "import venv"` succeeds (stdlib has it) but `ensurepip` does not, so `python3 -m venv` fails at the pip bootstrap step. Detect via `import ensurepip`, not `import venv`.
3. **transformers 5.10.2 needs torch ≥ 2.7.** Earlier torch pins (2.5, 2.6) lack `torch.float8_e8m0fnu` and `import transformers` crashes at module load.
4. **peft pulls in transformers's fp8 module unconditionally** at import time. Since `gemma_zs` never needs peft, omit it entirely from the install.
5. **`dockerfile==3.3.1` is a top-level import** of `dataset/annotate.py`. The `pip install -e . --no-deps` skips it; add it to the explicit pip line.
6. **Ubuntu user is not in the docker group on massedcompute.** `chmod 666 /var/run/docker.sock` is the single-line probe-pod fix.
7. **`huggingface-cli` is deprecated in huggingface_hub ≥ 1.** Use `hf auth whoami` instead.
8. **No prime CLI on the probe venv.** Don't expect the pod to self-terminate; terminate from the operator's laptop. The 4h watchdog is the cost backstop.
