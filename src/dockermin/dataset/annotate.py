"""Per-Dockerfile annotation pipeline: parse, build, test."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import time

import dockerfile
import docker
from docker.errors import BuildError, APIError

MIN_COMMANDS = 2


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    command_count: int = 0
    error: str = ""


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    tag: str = ""
    size_bytes: int = 0
    build_seconds: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class TestResult:
    ok: bool
    output: str = ""
    exit_code: int | None = None
    error: str = ""


@dataclass(frozen=True)
class AnnotateResult:
    ok: bool
    baseline_size: int = 0
    baseline_build_s: float = 0.0
    tag: str = ""
    error: str = ""


def _docker_client():
    return docker.from_env(timeout=600)


def parse_gate(df_text: str) -> ParseResult:
    """Validate Dockerfile parses and has at least MIN_COMMANDS instructions."""
    try:
        cmds = dockerfile.parse_string(df_text)
    except dockerfile.GoParseError as e:
        return ParseResult(ok=False, error=f"parse error: {e}")
    if len(cmds) < MIN_COMMANDS:
        return ParseResult(ok=False, command_count=len(cmds),
                           error=f"too short: {len(cmds)} < minimum {MIN_COMMANDS}")
    return ParseResult(ok=True, command_count=len(cmds))


def build_gate(df_text: str, timeout_s: int = 300) -> BuildResult:
    """Build the Dockerfile and return image size. Uses the classic builder via SDK (no BuildKit)."""
    client = _docker_client()
    digest = hashlib.sha256(df_text.encode()).hexdigest()[:12]
    tag = f"dockermin/curate:{digest}"
    t0 = time.perf_counter()
    try:
        image, log_stream = client.images.build(
            fileobj=io.BytesIO(df_text.encode()),
            tag=tag, rm=True, forcerm=True, timeout=timeout_s,
        )
        # Drain the log stream so the http connection isn't held.
        for _ in log_stream:
            pass
    except (BuildError, APIError) as e:
        return BuildResult(ok=False, error=f"build error: {e}")
    elapsed = time.perf_counter() - t0
    size = client.images.get(tag).attrs["Size"]
    return BuildResult(ok=True, tag=tag, size_bytes=size, build_seconds=elapsed)


def test_gate(tag: str, cmd: list[str], expected_substring: str, timeout_s: int = 30) -> TestResult:
    """Run cmd inside the image, capture combined output, match expected substring."""
    client = _docker_client()
    try:
        container = client.containers.run(
            tag, command=cmd, detach=True,
            network_mode="bridge",
            mem_limit="1g", memswap_limit="1g",
            nano_cpus=2_000_000_000,
            pids_limit=512,
        )
    except APIError as e:
        return TestResult(ok=False, error=f"start error: {e}")
    try:
        status = container.wait(timeout=timeout_s)
        exit_code = status.get("StatusCode")
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        combined = stdout + "\n" + stderr
        if exit_code != 0:
            return TestResult(ok=False, output=combined, exit_code=exit_code,
                              error=f"non-zero exit {exit_code}")
        if expected_substring not in combined:
            return TestResult(ok=False, output=combined, exit_code=exit_code,
                              error="expected substring not found")
        return TestResult(ok=True, output=combined, exit_code=exit_code)
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def annotate_one(df_text: str, test_cmd: list[str], expected_substring: str,
                 build_timeout_s: int = 300, test_timeout_s: int = 30) -> AnnotateResult:
    p = parse_gate(df_text)
    if not p.ok:
        return AnnotateResult(ok=False, error=p.error)
    b = build_gate(df_text, timeout_s=build_timeout_s)
    if not b.ok:
        return AnnotateResult(ok=False, error=b.error)
    t = test_gate(b.tag, test_cmd, expected_substring, timeout_s=test_timeout_s)
    if not t.ok:
        return AnnotateResult(ok=False, error=t.error)
    return AnnotateResult(ok=True, baseline_size=b.size_bytes,
                          baseline_build_s=b.build_seconds, tag=b.tag)


def infer_test_cmd(df_text: str) -> tuple[list[str] | None, str | None]:
    """Best-effort default test_cmd per ecosystem. Returns (None, None) if unknown."""
    text = df_text.lower()
    if "from python" in text or "pip install" in text:
        return (["python", "-c", "import sys;print('ok',sys.version_info[:2])"], "ok")
    if "from node" in text or "npm install" in text:
        return (["node", "-e", "console.log('ok',process.version)"], "ok")
    if "from golang" in text or "go build" in text:
        return (["/app/server", "--version"], "")
    if "from openjdk" in text or "from eclipse-temurin" in text:
        return (["java", "-version"], "")
    return (None, None)
