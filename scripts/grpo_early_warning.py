"""GRPO v1.1 early-warning panel: catch the 5 first-20-step failure modes the
devil's-advocate review flagged against the original single-metric stop-the-line.

The v1.0 stop-the-line check ("`stdev > 0.05` once") was wrong on two axes:

1. **Magnitude.** The v1.0 threshold was tuned for the pre-smoothing reward range.
   After the v1.1 build-fail/test-fail smoothing (gates.py: BUILD_FAIL_CMD_CREDIT_MAX
   = 0.10, TEST_FAIL_CMD_CREDIT_MAX = 0.15), the relevant stdev floor moved.
2. **Coverage.** A single stdev reading conflates five distinct failure modes that
   each need their own probe:

| # | Failure mode             | What it looks like in the rollouts                     | Caught by metric |
|---|--------------------------|---------------------------------------------------------|------------------|
| A | Smoothing saturation     | most failed builds at command_count >= 20 -> rewards    | 1 (cap-sat rate) |
|   |                          | pile up at BUILD_FAIL_SCORE + 0.10 (or test+0.15) cap;  |                  |
|   |                          | smoothing buys no advantage, GRPO group stdev collapses | + 2 (stdev)      |
| B | Pad-hacking              | model emits arbitrarily many no-op RUN lines to climb   | 3 (cmd_count     |
|   |                          | the partial-credit ramp                                 | slope)           |
| C | Mode collapse            | every rollout in a group is byte-identical              | 4 (unique/group) |
|   |                          | (entropy crash, sampler stuck)                          | + 5 (entropy)    |
| D | Dataset unlearnable      | a subset of problem_ids never sees reward >= 0.5,       | 6 (stuck-prompt  |
|   |                          | wasting batch and skewing advantage                     | fraction)        |
| E | Silent stuckness         | trainer prints losses but adapter weights barely move   | 7 (L2 drift)     |
|   |                          | / grad_norm pinned near zero (NaN-suppressed grads)     | + 8 (grad norm)  |

Each metric outputs GREEN / YELLOW / RED with calibration thresholds below. The
script exits 0 if no RED, 1 if any RED. Missing data is reported as `(no data)`
with no status — it never crashes. Intended invocation on the pod:

    python scripts/grpo_early_warning.py --run-dir outputs/run_default

Optionally narrow the analysis window (default last 20 steps):

    python scripts/grpo_early_warning.py --run-dir outputs/run_default --step-window 10,20

The script is read-only — it scrapes rollouts, trainer.log, and the adapter
safetensors checkpoint, computes everything in-process, and prints a markdown
table for `cat` over SSH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dockermin.dataset.annotate import parse_gate
from dockermin.ops import read_jsonl
from dockermin.reward.dockermin_reward import _completion_text
from dockermin.reward.gates import (
    BUILD_FAIL_CMD_CREDIT_MAX,
    BUILD_FAIL_SCORE,
    TEST_FAIL_CMD_CREDIT_MAX,
    TEST_FAIL_SCORE,
)
from dockermin.reward.prompts import extract_dockerfile

GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"
NA = "N/A"

_CAP_TOL = 1e-4
_BUILD_CAP = BUILD_FAIL_SCORE + BUILD_FAIL_CMD_CREDIT_MAX
_TEST_CAP = TEST_FAIL_SCORE + TEST_FAIL_CMD_CREDIT_MAX

_STEP_DIR_RX = re.compile(r"^step_(\d+)$")
_ENTROPY_RX = re.compile(r"entropy[\s]*[=:][\s]*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
_GRAD_NORM_RX = re.compile(r"grad_norm[\s]*[=:][\s]*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)

_REWARD_PASS_THRESHOLD = 0.5
_MIN_GROUP_SIZE = 2
_STEP_WINDOW_PARTS = 2
_GRAD_NORM_TAIL = 10
_PAD_HACK_INFLATION_RATIO = 1.3

# Per-metric calibration thresholds. Each tuple is (green_cut, yellow_cut)
# interpreted by either _status_high (higher is better) or _status_low.
# Reviewed against the v1.1 reward range (gates.py: BUILD_FAIL_SCORE..1.0).
_CAP_SAT_THRESHOLDS = (0.20, 0.40)
_STDEV_THRESHOLDS = (0.05, 0.02)
_SLOPE_GREEN = 0.5
_SLOPE_YELLOW = 1.5
_UNIQUE_THRESHOLDS = (0.75, 0.50)
_ENTROPY_THRESHOLDS = (0.05, 0.01)
_STUCK_THRESHOLDS = (0.15, 0.30)
_L2_DRIFT_THRESHOLDS = (0.5, 0.1)
_GRAD_NORM_THRESHOLDS = (0.01, 0.001)


def _status_high(value: float, green_cut: float, yellow_cut: float) -> str:
    """Higher is better: GREEN if value > green_cut, RED if value < yellow_cut."""
    if value > green_cut:
        return GREEN
    if value >= yellow_cut:
        return YELLOW
    return RED


def _status_low(value: float, green_cut: float, yellow_cut: float) -> str:
    """Lower is better: GREEN if value < green_cut, RED if value > yellow_cut."""
    if value < green_cut:
        return GREEN
    if value <= yellow_cut:
        return YELLOW
    return RED


@dataclass(frozen=True)
class Rollout:
    step: int
    problem_id: str
    reward: float
    completion_text: str
    info: dict[str, Any]


@dataclass(frozen=True)
class MetricRow:
    name: str
    value: str
    status: str


def parse_step_window(raw: str) -> tuple[int, int]:
    parts = raw.split(",")
    if len(parts) != _STEP_WINDOW_PARTS:
        msg = f"--step-window must be 'start,end', got {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end < start:
        msg = f"--step-window invalid: start={start}, end={end}"
        raise argparse.ArgumentTypeError(msg)
    return start, end


def fetch_step_dirs(rollouts_root: Path, window: tuple[int, int]) -> list[Path]:
    if not rollouts_root.is_dir():
        return []
    matched: list[tuple[int, Path]] = []
    for child in rollouts_root.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_DIR_RX.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        if window[0] <= step <= window[1]:
            matched.append((step, child))
    matched.sort(key=lambda pair: pair[0])
    return [p for _, p in matched]


def _rollout_from_row(row: dict[str, Any], step: int) -> Rollout | None:
    reward = row.get("reward")
    problem_id = row.get("problem_id")
    if reward is None or problem_id is None:
        return None
    text = _completion_text(row.get("completion", ""))
    info = row.get("info") or {}
    return Rollout(
        step=step,
        problem_id=str(problem_id),
        reward=float(reward),
        completion_text=text,
        info=info if isinstance(info, dict) else {},
    )


def fetch_rollouts(step_dirs: list[Path]) -> list[Rollout]:
    rollouts: list[Rollout] = []
    for step_dir in step_dirs:
        step = int(_STEP_DIR_RX.match(step_dir.name).group(1))  # type: ignore[union-attr]
        for jsonl in step_dir.glob("*.jsonl"):
            for row in read_jsonl(jsonl):
                r = _rollout_from_row(row, step)
                if r is not None:
                    rollouts.append(r)
    return rollouts


def _is_cap_saturated(reward: float) -> bool:
    near_build_cap = _BUILD_CAP <= reward <= _BUILD_CAP + _CAP_TOL
    near_test_cap = _TEST_CAP <= reward <= _TEST_CAP + _CAP_TOL
    return near_build_cap or near_test_cap


def cap_saturation_rate(rollouts: list[Rollout]) -> MetricRow:
    if not rollouts:
        return MetricRow("Cap-saturation rate", "(no data)", NA)
    saturated = sum(1 for r in rollouts if _is_cap_saturated(r.reward))
    rate = saturated / len(rollouts)
    status = _status_low(rate, *_CAP_SAT_THRESHOLDS)
    return MetricRow("Cap-saturation rate", f"{rate:.3f}", status)


def _groups_by_problem(rollouts: list[Rollout]) -> dict[tuple[int, str], list[Rollout]]:
    groups: dict[tuple[int, str], list[Rollout]] = defaultdict(list)
    for r in rollouts:
        groups[(r.step, r.problem_id)].append(r)
    return groups


def group_reward_stdev_median(rollouts: list[Rollout]) -> MetricRow:
    groups = _groups_by_problem(rollouts)
    stdevs = [
        statistics.pstdev([r.reward for r in members]) for members in groups.values() if len(members) >= _MIN_GROUP_SIZE
    ]
    if not stdevs:
        return MetricRow("Group reward stdev (med)", "(no data)", NA)
    med = statistics.median(stdevs)
    status = _status_high(med, *_STDEV_THRESHOLDS)
    return MetricRow("Group reward stdev (med)", f"{med:.4f}", status)


def _command_count(rollout: Rollout) -> int | None:
    df_text = extract_dockerfile(rollout.completion_text) or ""
    if not df_text:
        return None
    p = parse_gate(df_text)
    return p.command_count if p.command_count else None


def _step_mean_cmd_counts(rollouts: list[Rollout]) -> dict[int, float]:
    by_step: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        c = _command_count(r)
        if c is not None:
            by_step[r.step].append(c)
    return {step: sum(vals) / len(vals) for step, vals in by_step.items() if vals}


def _linear_slope(xs: list[int], ys: list[float]) -> float:
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    slope, _ = np.polyfit(arr_x, arr_y, 1)
    return float(slope)


def cmd_count_slope(rollouts: list[Rollout]) -> MetricRow:
    means = _step_mean_cmd_counts(rollouts)
    if len(means) < _MIN_GROUP_SIZE:
        return MetricRow("Mean cmd_count slope", "(no data)", NA)
    steps_sorted = sorted(means.keys())
    ys = [means[s] for s in steps_sorted]
    slope = _linear_slope(steps_sorted, ys)
    inflated_last_vs_first = ys[-1] > _PAD_HACK_INFLATION_RATIO * ys[0]
    status = _cmd_slope_status(abs(slope), inflated_last_vs_first)
    return MetricRow("Mean cmd_count slope", f"{slope:+.3f}", status)


def _cmd_slope_status(abs_slope: float, inflated_last_vs_first: bool) -> str:
    if abs_slope < _SLOPE_GREEN:
        return GREEN
    if abs_slope <= _SLOPE_YELLOW:
        return YELLOW
    return RED if inflated_last_vs_first else YELLOW


def _dockerfile_hash(rollout: Rollout) -> str:
    df_text = extract_dockerfile(rollout.completion_text) or rollout.completion_text
    return hashlib.sha1(df_text.encode("utf-8", "replace")).hexdigest()  # noqa: S324 — identity hash, not crypto


def unique_completions_per_group(rollouts: list[Rollout]) -> MetricRow:
    groups = _groups_by_problem(rollouts)
    ratios: list[float] = []
    for members in groups.values():
        if not members:
            continue
        unique = len({_dockerfile_hash(r) for r in members})
        ratios.append(unique / len(members))
    if not ratios:
        return MetricRow("Unique completions/group", "(no data)", NA)
    mean_ratio = sum(ratios) / len(ratios)
    status = _status_high(mean_ratio, *_UNIQUE_THRESHOLDS)
    return MetricRow("Unique completions/group", f"{mean_ratio:.3f}", status)


def _find_log(run_dir: Path) -> Path | None:
    candidates = [run_dir / "trainer.log", Path.home() / "trainer.log"]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _grep_last_floats(log_path: Path, pattern: re.Pattern[str]) -> list[float]:
    values: list[float] = []
    with log_path.open(encoding="utf-8", errors="replace") as fin:
        for line in fin:
            m = pattern.search(line)
            if m:
                try:
                    values.append(float(m.group(1)))
                except ValueError:
                    continue
    return values


def trainer_entropy_last(run_dir: Path) -> MetricRow:
    log_path = _find_log(run_dir)
    if log_path is None:
        return MetricRow("Trainer entropy", "(no data)", NA)
    values = _grep_last_floats(log_path, _ENTROPY_RX)
    if not values:
        return MetricRow("Trainer entropy", "(no data)", NA)
    last = values[-1]
    status = _status_high(last, *_ENTROPY_THRESHOLDS)
    return MetricRow("Trainer entropy", f"{last:.4f}", status)


def stuck_prompt_fraction(rollouts: list[Rollout]) -> MetricRow:
    if not rollouts:
        return MetricRow("Stuck-prompt fraction", "(no data)", NA)
    best_by_pid: dict[str, float] = defaultdict(lambda: -math.inf)
    for r in rollouts:
        best_by_pid[r.problem_id] = max(best_by_pid[r.problem_id], r.reward)
    stuck = sum(1 for best in best_by_pid.values() if best < _REWARD_PASS_THRESHOLD)
    frac = stuck / len(best_by_pid)
    status = _status_low(frac, *_STUCK_THRESHOLDS)
    return MetricRow("Stuck-prompt fraction", f"{frac:.3f}", status)


def _find_adapter_pair(run_dir: Path) -> tuple[Path, Path] | None:
    bcasts = run_dir / "broadcasts"
    ckpts = run_dir / "checkpoints"
    target = bcasts / "step_25" / "adapter_model.safetensors"
    if not target.is_file():
        target = ckpts / "step_25" / "adapter_model.safetensors"
    if not target.is_file():
        return None
    baseline = _find_earliest_adapter([bcasts, ckpts])
    if baseline is None or baseline.samefile(target):
        return None
    return baseline, target


def _find_earliest_adapter(roots: list[Path]) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            m = _STEP_DIR_RX.match(child.name)
            adapter = child / "adapter_model.safetensors"
            if m and adapter.is_file():
                candidates.append((int(m.group(1)), adapter))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def _l2_diff(baseline_path: Path, target_path: Path) -> float:
    # Lazy import: safetensors ships with the prime-rl pod venv (peft/transformers
    # pull it in), but not with the local dockermin dev venv used by `make quality`.
    from safetensors import safe_open

    total_sq = 0.0
    with safe_open(str(baseline_path), framework="numpy") as fb, safe_open(str(target_path), framework="numpy") as ft:
        keys = set(fb.keys()) & set(ft.keys())
        for key in keys:
            wb = fb.get_tensor(key).astype(np.float32, copy=False)
            wt = ft.get_tensor(key).astype(np.float32, copy=False)
            if wb.shape != wt.shape:
                continue
            diff = wt - wb
            total_sq += float(np.sum(diff * diff))
    return math.sqrt(total_sq)


def adapter_l2_drift(run_dir: Path) -> MetricRow:
    pair = _find_adapter_pair(run_dir)
    if pair is None:
        return MetricRow("Adapter L2 drift (s25)", "(no data)", NA)
    try:
        drift = _l2_diff(*pair)
    except (
        OSError,
        ValueError,
        ModuleNotFoundError,
    ) as e:  # safetensors may be absent in the local dev venv; a corrupt file must not crash the panel
        return MetricRow("Adapter L2 drift (s25)", f"(error: {e})", NA)
    status = _status_high(drift, *_L2_DRIFT_THRESHOLDS)
    return MetricRow("Adapter L2 drift (s25)", f"{drift:.4f}", status)


def grad_norm_median(run_dir: Path) -> MetricRow:
    log_path = _find_log(run_dir)
    if log_path is None:
        return MetricRow("Grad norm median (last10)", "(no data)", NA)
    values = _grep_last_floats(log_path, _GRAD_NORM_RX)
    if not values:
        return MetricRow("Grad norm median (last10)", "(no data)", NA)
    tail = values[-_GRAD_NORM_TAIL:]
    med = statistics.median(tail)
    status = _status_high(med, *_GRAD_NORM_THRESHOLDS)
    return MetricRow("Grad norm median (last10)", f"{med:.5f}", status)


def render_panel(run_dir: Path, window: tuple[int, int], rows: list[MetricRow]) -> str:
    header = (
        f"# GRPO v1.1 early-warning panel — run: {run_dir} — window: steps {window[0]}-{window[1]}\n\n"
        "| # | Metric                    | Value     | Status |\n"
        "|---|---------------------------|----------:|--------|\n"
    )
    body = "\n".join(
        f"| {i + 1} | {row.name:<25s} | {row.value:>9s} | {row.status:<6s} |" for i, row in enumerate(rows)
    )
    return header + body + "\n"


def render_verdict(rows: list[MetricRow]) -> tuple[str, int]:
    n_red = sum(1 for r in rows if r.status == RED)
    n_yellow = sum(1 for r in rows if r.status == YELLOW)
    n_green = sum(1 for r in rows if r.status == GREEN)
    if n_red > 0:
        text = f"Verdict: {n_red} RED / {n_yellow} YELLOW / {n_green} GREEN — STOP THE RUN."
        return text, 1
    if n_yellow > 0:
        text = f"Verdict: 0 RED / {n_yellow} YELLOW / {n_green} GREEN — proceed with caution."
        return text, 0
    text = f"Verdict: 0 RED / 0 YELLOW / {n_green} GREEN — proceed."
    return text, 0


def collect_rows(run_dir: Path, window: tuple[int, int]) -> list[MetricRow]:
    step_dirs = fetch_step_dirs(run_dir / "rollouts", window)
    rollouts = fetch_rollouts(step_dirs)
    return [
        cap_saturation_rate(rollouts),
        group_reward_stdev_median(rollouts),
        cmd_count_slope(rollouts),
        unique_completions_per_group(rollouts),
        trainer_entropy_last(run_dir),
        stuck_prompt_fraction(rollouts),
        adapter_l2_drift(run_dir),
        grad_norm_median(run_dir),
    ]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO v1.1 pod-side early-warning panel — 8 metrics for the first 20 training steps."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the run output dir (e.g. outputs/run_default).",
    )
    parser.add_argument(
        "--step-window",
        type=parse_step_window,
        default=(10, 20),
        help="Inclusive step window 'start,end' (default: 10,20).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit a JSON line with each metric for downstream pipes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    rows = collect_rows(args.run_dir, args.step_window)
    panel = render_panel(args.run_dir, args.step_window, rows)
    print(panel)
    verdict, exit_code = render_verdict(rows)
    print(verdict)
    if args.json:
        payload = {row.name: {"value": row.value, "status": row.status} for row in rows}
        print(json.dumps(payload))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
