"""Baseline runners: zero-shot models, hadolint, slim, manual rewriter, dockermin LoRA.

Each baseline is a function ``(triple: dict, ...) -> EvalEntry``. A triple is the
schema produced by ``scripts/run_annotate.py``: ``{id, dockerfile, test_cmd,
expected_substring, baseline_size, ...}``.

Each baseline must:
  1. Produce a rewritten Dockerfile (or detect failure).
  2. Call ``dataset.annotate.annotate_one`` to verify build + test.
  3. Return ``EvalEntry(new_size_bytes, test_passes, elapsed_s, error)``.

Heavy deps (vllm, transformers, peft, anthropic, openai, docker) are pinned in
``pyproject.toml`` but may not be installed on a dev box. Imports at module top
are deliberate per project convention (AST-only verification locally; runtime
verification on the GPU pod).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dockermin.dataset.annotate import annotate_one
from dockermin.reward.prompts import extract_dockerfile, format_messages

if TYPE_CHECKING:
    from collections.abc import Callable

# Narrow retry policies for the two API-only baselines. Transient connection
# blips and rate limits are worth a few exponential-backoff retries; docker
# build/test failures are real signal and must NOT be retried (that's why this
# lives here and not around ``run_one``).
_OPENAI_TRANSIENT = (openai.APIConnectionError, openai.RateLimitError)
_ANTHROPIC_TRANSIENT = (anthropic.APIConnectionError, anthropic.RateLimitError)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalEntry:
    baseline: str
    triple_id: str
    new_size_bytes: int | None
    test_passes: bool
    elapsed_s: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BUILD_TIMEOUT_S = 600
_DEFAULT_TEST_TIMEOUT_S = 30


def _verify(triple: dict[str, Any], new_df: str) -> tuple[int | None, bool, str]:
    """Run annotate_one on a rewritten Dockerfile. Returns (size_or_None, passed, err)."""
    if not new_df or not new_df.strip():
        return None, False, "empty dockerfile"
    res = annotate_one(
        new_df,
        triple["test_cmd"],
        triple.get("expected_substring", ""),
        build_timeout_s=_DEFAULT_BUILD_TIMEOUT_S,
        test_timeout_s=_DEFAULT_TEST_TIMEOUT_S,
    )
    if not res.ok:
        return None, False, res.error
    return res.baseline_size, True, ""


def _triple_id(triple: dict[str, Any]) -> str:
    return str(triple.get("id") or triple.get("source_url") or "unknown")


def _entry(name: str, triple: dict[str, Any], new_df: str, t0: float) -> EvalEntry:
    """Convenience: verify ``new_df`` and build an EvalEntry."""
    size, passed, err = _verify(triple, new_df)
    return EvalEntry(
        baseline=name,
        triple_id=_triple_id(triple),
        new_size_bytes=size,
        test_passes=passed,
        elapsed_s=time.perf_counter() - t0,
        error=err,
    )


def _error_entry(name: str, triple: dict[str, Any], t0: float, err: str) -> EvalEntry:
    return EvalEntry(
        baseline=name,
        triple_id=_triple_id(triple),
        new_size_bytes=None,
        test_passes=False,
        elapsed_s=time.perf_counter() - t0,
        error=err,
    )


# ---------------------------------------------------------------------------
# Baseline 1: Qwen 2.5 Coder 7B zero-shot (transformers)
# ---------------------------------------------------------------------------
# Loaded via transformers (NOT vLLM) so it mirrors the dockermin baseline exactly
# — same engine, the only difference being the LoRA adapter — and so it needs no
# CUDA toolkit. vLLM's prime build enters a DeepGEMM/FP8 path that requires nvcc,
# which the inference pods do not ship; transformers avoids that entirely.
_BASE_HANDLE: dict[str, Any] = {}


def _hf_base() -> tuple[Any, Any]:
    """Lazy-load base Qwen 2.5 Coder 7B (no adapter) via transformers."""
    if "model" in _BASE_HANDLE:
        return _BASE_HANDLE["model"], _BASE_HANDLE["tok"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16", device_map="auto")
    _BASE_HANDLE["model"] = model
    _BASE_HANDLE["tok"] = tok
    return model, tok


def baseline_qwen_zero_shot(triple: dict[str, Any]) -> EvalEntry:
    """Base Qwen 2.5 Coder 7B with the standard prompt, no fine-tuning (transformers)."""
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        model, tok = _hf_base()
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=1024, temperature=0.2, do_sample=True)
        text = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        new_df = extract_dockerfile(text)
        if new_df is None:
            return _error_entry("qwen_zs", triple, t0, "no fenced dockerfile block")
        return _entry("qwen_zs", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("qwen_zs", triple, t0, f"qwen error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 2: GPT-4o zero-shot
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_OPENAI_TRANSIENT),
)
def _openai_chat(
    client: openai.OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> openai.types.chat.ChatCompletion:
    """Narrow retry wrapper for OpenAI chat completions."""
    return client.chat.completions.create(
        model=model,
        messages=cast("Any", messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def baseline_gpt4o(triple: dict[str, Any]) -> EvalEntry:
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        client = openai.OpenAI()
        resp = _openai_chat(
            client,
            model="gpt-4o-2024-11-20",
            messages=msgs,
            temperature=0.2,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        new_df = extract_dockerfile(text)
        if new_df is None:
            return _error_entry("gpt4o", triple, t0, "no fenced dockerfile block")
        return _entry("gpt4o", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("gpt4o", triple, t0, f"gpt4o error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 3: Claude Sonnet 4.6 zero-shot
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(_ANTHROPIC_TRANSIENT),
)
def _anthropic_messages(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> anthropic.types.Message:
    """Narrow retry wrapper for Anthropic messages.create."""
    return client.messages.create(
        model=model,
        system=system,
        messages=cast("Any", messages),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def baseline_sonnet_zero_shot(triple: dict[str, Any]) -> EvalEntry:
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        # Anthropic API splits system from the message list.
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        chat = [m for m in msgs if m["role"] != "system"]
        client = anthropic.Anthropic()
        resp = _anthropic_messages(
            client,
            model="claude-sonnet-4-6",
            system=system,
            messages=chat,
            max_tokens=1024,
            temperature=0.2,
        )
        # Concatenate text blocks (per Anthropic SDK content-block shape).
        text = "".join(getattr(b, "text", "") for b in resp.content)
        new_df = extract_dockerfile(text)
        if new_df is None:
            return _error_entry("sonnet_zs", triple, t0, "no fenced dockerfile block")
        return _entry("sonnet_zs", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("sonnet_zs", triple, t0, f"sonnet error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 4: hadolint mechanical-fix
# ---------------------------------------------------------------------------

_HADOLINT_FIXERS: dict[str, Callable[[str], str]] = {}


def _append_to_line(line: str, cleanup: str) -> str:
    """Append ``&& <cleanup>`` to a line, preserving a trailing backslash continuation."""
    stripped = line.rstrip()
    if stripped.endswith("\\"):
        core = stripped[:-1].rstrip()
        return f"{core} && {cleanup} \\"
    return f"{stripped} && {cleanup}"


def _append_cleanup_to_matching_lines(
    df: str,
    trigger: str,
    skip_marker: str,
    cleanup: str,
) -> str:
    """Append ``&& <cleanup>`` to each line matching ``trigger`` that lacks ``skip_marker``."""
    out = [
        _append_to_line(line, cleanup) if re.search(trigger, line) and skip_marker not in line else line
        for line in df.splitlines()
    ]
    return "\n".join(out)


def _fix_dl3009(df: str) -> str:
    """DL3009: clean up apt cache. Append rm -rf /var/lib/apt/lists/* to apt-get install lines."""
    return _append_cleanup_to_matching_lines(
        df,
        r"\bapt-get\s+install\b",
        "/var/lib/apt/lists",
        "rm -rf /var/lib/apt/lists/*",
    )


def _fix_dl3015(df: str) -> str:
    """DL3015: --no-install-recommends on apt-get install."""

    def sub(m: re.Match[str]) -> str:
        head = m.group(0)
        if "--no-install-recommends" in head:
            return head
        return head + " --no-install-recommends"

    return re.sub(r"\bapt-get\s+install\b(?!\s+--no-install-recommends)", sub, df)


def _fix_dl3019(df: str) -> str:
    """DL3019: apk add --no-cache."""

    def sub(m: re.Match[str]) -> str:
        head = m.group(0)
        if "--no-cache" in head:
            return head
        return head + " --no-cache"

    return re.sub(r"\bapk\s+add\b(?!\s+--no-cache)", sub, df)


def _fix_dl3042(df: str) -> str:
    """DL3042: pip install --no-cache-dir."""

    def sub(m: re.Match[str]) -> str:
        head = m.group(0)
        if "--no-cache-dir" in head:
            return head
        return head + " --no-cache-dir"

    return re.sub(r"\bpip\s+install\b(?!\s+--no-cache-dir)", sub, df)


def _fix_dl3060(df: str) -> str:
    """DL3060: yarn install followed by yarn cache clean."""
    return _append_cleanup_to_matching_lines(
        df,
        r"\byarn\s+install\b",
        "yarn cache clean",
        "yarn cache clean",
    )


_HADOLINT_FIXERS.update(
    {
        "DL3009": _fix_dl3009,
        "DL3015": _fix_dl3015,
        "DL3019": _fix_dl3019,
        "DL3042": _fix_dl3042,
        "DL3060": _fix_dl3060,
    }
)


def _run_hadolint(df_text: str) -> list[dict[str, Any]]:
    """Run hadolint --format json over a Dockerfile string; return list of findings."""
    proc = subprocess.run(
        ["hadolint", "--format", "json", "-"],  # noqa: S607 — hadolint resolved via PATH by design
        input=df_text,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    # hadolint exits non-zero when it finds issues; that's the common path.
    out = proc.stdout.strip() or "[]"
    try:
        findings: list[dict[str, Any]] = json.loads(out)
    except json.JSONDecodeError:
        return []
    return findings


def baseline_hadolint(triple: dict[str, Any]) -> EvalEntry:
    """Mechanically apply fixes for DL3009/DL3015/DL3019/DL3042/DL3060."""
    t0 = time.perf_counter()
    try:
        df = triple["dockerfile"]
        try:
            findings = _run_hadolint(df)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return _error_entry("hadolint", triple, t0, f"hadolint not available: {e!r}")

        codes = {f.get("code", "") for f in findings}
        new_df = df
        for code, fixer in _HADOLINT_FIXERS.items():
            if code in codes:
                new_df = fixer(new_df)
        return _entry("hadolint", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("hadolint", triple, t0, f"hadolint error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 5: SlimToolkit (slim build)
# ---------------------------------------------------------------------------


def _run_slim_build(workdir: Path, original_tag: str, slim_tag: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``slim build``. http-probe disabled (no http target); continue after 15s steady-state."""
    return subprocess.run(  # noqa: S603 — fixed CLI args, no untrusted executable
        [  # noqa: S607 — slim resolved via PATH by design
            "slim",
            "build",
            "--http-probe=false",
            "--continue-after",
            "15",
            "--target",
            original_tag,
            "--tag",
            slim_tag,
        ],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _slim_image_entry(triple: dict[str, Any], slim_tag: str, t0: float) -> EvalEntry:
    """Run the slim image directly and report its size (no slim-emitted Dockerfile)."""
    from dockermin.dataset.annotate import _docker_client, run_test_gate

    t = run_test_gate(
        slim_tag, triple["test_cmd"], triple.get("expected_substring", ""), timeout_s=_DEFAULT_TEST_TIMEOUT_S
    )
    client = _docker_client()
    slim_size = client.images.get(slim_tag).attrs["Size"]
    return EvalEntry(
        baseline="slim",
        triple_id=_triple_id(triple),
        new_size_bytes=slim_size if t.ok else None,
        test_passes=t.ok,
        elapsed_s=time.perf_counter() - t0,
        error="" if t.ok else t.error,
    )


def _run_slim(triple: dict[str, Any], t0: float) -> EvalEntry:
    """Build the original image, run slim in a temp workdir, and verify the slim output."""
    from dockermin.dataset.annotate import build_gate

    # 1) Build the original image so slim has something to chew on.
    b = build_gate(triple["dockerfile"], timeout_s=_DEFAULT_BUILD_TIMEOUT_S)
    if not b.ok:
        return _error_entry("slim", triple, t0, f"original build failed: {b.error}")

    original_tag = b.tag
    slim_tag = f"{original_tag}.slim"

    workdir = Path(tempfile.mkdtemp(prefix="slim_"))
    try:
        proc = _run_slim_build(workdir, original_tag, slim_tag)
        if proc.returncode != 0:
            return _error_entry("slim", triple, t0, f"slim build failed rc={proc.returncode}: {proc.stderr[-300:]}")

        # Prefer a slim-emitted Dockerfile; otherwise report the slim image size.
        slim_df_path = workdir / ".slim-state" / original_tag / "Dockerfile"
        if slim_df_path.exists():
            return _entry("slim", triple, slim_df_path.read_text(), t0)
        return _slim_image_entry(triple, slim_tag, t0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def baseline_slim(triple: dict[str, Any]) -> EvalEntry:
    """Build the original image, then run `slim build` to produce a minified image.

    If slim emits a rewritten Dockerfile (in ./.slim-state/<tag>/Dockerfile),
    verify it with annotate_one. Otherwise fall back to reporting the slim image
    size directly (we still test_gate against the slim tag).
    """
    t0 = time.perf_counter()
    try:
        return _run_slim(triple, t0)
    except FileNotFoundError as e:
        return _error_entry("slim", triple, t0, f"slim binary not found: {e!r}")
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("slim", triple, t0, f"slim error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 6: Manual best-practice rewriter (5 mechanical transforms)
# ---------------------------------------------------------------------------

_BASE_IMAGE_MAP = {
    "python:3.12": "python:3.12-slim",
    "node:20": "node:20-alpine",
    "openjdk:17": "eclipse-temurin:17-jre-alpine",
}


def _swap_base_image(df: str) -> str:
    """(a) Replace FROM <known> with the slim/alpine equivalent."""

    def sub(m: re.Match[str]) -> str:
        prefix = m.group("prefix")
        image = m.group("image")
        suffix = m.group("suffix") or ""
        return f"{prefix}{_BASE_IMAGE_MAP.get(image, image)}{suffix}"

    pattern = re.compile(
        r"(?P<prefix>^\s*FROM\s+)"
        r"(?P<image>[A-Za-z0-9_./-]+:[A-Za-z0-9_.-]+)"
        r"(?P<suffix>(\s+AS\s+\S+)?\s*$)",
        re.MULTILINE | re.IGNORECASE,
    )
    return pattern.sub(sub, df)


def _consume_continuations(body_parts: list[str], lines: list[str], i: int) -> int:
    """Absorb trailing-backslash continuation lines into ``body_parts``; return the next index."""
    while body_parts[-1].rstrip().endswith("\\") and i < len(lines):
        body_parts[-1] = body_parts[-1].rstrip()[:-1].rstrip()
        body_parts.append(lines[i].lstrip())
        i += 1
    return i


_RUN_RE = re.compile(r"^(\s*)RUN\s+(.*)$")


def _collect_run_block(m: re.Match[str], lines: list[str], i: int) -> tuple[str, int]:
    """Starting at a matched RUN line, render the merged RUN (absorbing following RUN blocks).

    ``m`` is the match for ``lines[i]``. Returns (rendered_run_line, next_index).
    """
    indent = m.group(1)
    body_parts: list[str] = [m.group(2)]
    i = _consume_continuations(body_parts, lines, i + 1)
    # Greedily merge following RUN blocks.
    while i < len(lines):
        m2 = _RUN_RE.match(lines[i])
        if not m2:
            break
        next_body = [m2.group(2)]
        i = _consume_continuations(next_body, lines, i + 1)
        body_parts.append("&& " + next_body[0])
        body_parts.extend(next_body[1:])
    joined = body_parts[0] if len(body_parts) == 1 else " \\\n    ".join(body_parts)
    return f"{indent}RUN {joined}", i


def _collapse_consecutive_runs(df: str) -> str:
    """(e) Merge consecutive RUN statements with ' && \\' separators."""
    lines = df.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _RUN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        rendered, i = _collect_run_block(m, lines, i)
        out.append(rendered)
    return "\n".join(out)


def baseline_manual_best_practice(triple: dict[str, Any]) -> EvalEntry:
    """Apply the 5 mechanical transforms from Task 5.7."""
    t0 = time.perf_counter()
    try:
        df = triple["dockerfile"]
        df = _swap_base_image(df)  # (a)
        df = _fix_dl3015(df)  # (b) --no-install-recommends
        df = _fix_dl3009(df)  # (c) rm -rf /var/lib/apt/lists/*
        df = _fix_dl3042(df)  # (d) pip --no-cache-dir
        # (e) RUN-collapse intentionally dropped: it mangles multi-line RUN bodies
        # (internal `&&` + backslash continuations) and build-fails the result,
        # which would make this an unfairly-broken baseline. The layer-merge saving
        # is marginal next to the base-swap + cache-cleanup transforms above.
        return _entry("manual", triple, df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("manual", triple, t0, f"manual error: {e!r}")


# ---------------------------------------------------------------------------
# Baseline 7: dockermin LoRA (base Qwen + PEFT adapter)
# ---------------------------------------------------------------------------

_HF_HANDLE: dict[str, Any] = {}


def _hf_dockermin(model_id: str) -> tuple[Any, Any]:
    """Lazy-load Qwen base + PEFT adapter for inference."""
    if "model" in _HF_HANDLE and _HF_HANDLE.get("adapter") == model_id:
        return _HF_HANDLE["model"], _HF_HANDLE["tok"]
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16", device_map="auto")
    model = PeftModel.from_pretrained(base, model_id)
    _HF_HANDLE["model"] = model
    _HF_HANDLE["tok"] = tok
    _HF_HANDLE["adapter"] = model_id
    return model, tok


def baseline_dockermin(triple: dict[str, Any], model_id: str = "vtemian/dockermin-qwen7b-lora-v1") -> EvalEntry:
    """The fine-tuned dockermin LoRA adapter applied on top of base Qwen."""
    t0 = time.perf_counter()
    try:
        msgs = format_messages(triple["dockerfile"])
        model, tok = _hf_dockermin(model_id)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=1024, temperature=0.2, do_sample=True)
        text = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        new_df = extract_dockerfile(text)
        if new_df is None:
            return _error_entry("dockermin", triple, t0, "no fenced dockerfile block")
        return _entry("dockermin", triple, new_df, t0)
    except Exception as e:  # noqa: BLE001 — safety net: one bad Dockerfile must not crash the eval loop
        return _error_entry("dockermin", triple, t0, f"dockermin error: {e!r}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., EvalEntry]] = {
    "qwen_zs": baseline_qwen_zero_shot,
    "gpt4o": baseline_gpt4o,
    "sonnet_zs": baseline_sonnet_zero_shot,
    "hadolint": baseline_hadolint,
    "slim": baseline_slim,
    "manual": baseline_manual_best_practice,
    "dockermin": baseline_dockermin,
}


def available_baselines() -> list[str]:
    """Names of all dispatch-routable baselines (agent_loop registers itself)."""
    return sorted(_REGISTRY.keys())


def register_baseline(name: str, fn: Callable[..., EvalEntry]) -> None:
    """Side-channel registration so ``agent_loop.py`` can add itself."""
    _REGISTRY[name] = fn


def run_one(baseline_name: str, triple: dict[str, Any], **kwargs: object) -> EvalEntry:
    """Dispatch by name. Extra kwargs forwarded to the baseline (e.g. model_id)."""
    if baseline_name not in _REGISTRY:
        return EvalEntry(
            baseline=baseline_name,
            triple_id=_triple_id(triple),
            new_size_bytes=None,
            test_passes=False,
            elapsed_s=0.0,
            error=f"unknown baseline: {baseline_name}",
        )
    return _REGISTRY[baseline_name](triple, **kwargs)
