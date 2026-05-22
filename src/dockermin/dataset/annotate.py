"""Per-Dockerfile annotation pipeline: parse, build, test."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import dockerfile
import docker
from docker.errors import APIError

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
    __test__ = False  # tell pytest not to auto-collect this as a test class
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


def build_gate(df_text: str, timeout_s: int = 300,
               builder: str | None = None, cache_dir: str | None = None) -> BuildResult:
    """Build the Dockerfile via `docker buildx build` and return image size.

    BuildKit hot-path so apt/pip/npm cache mounts amortize across rollouts.
    Subprocess timeout actually kills the build on the wall clock (the
    docker SDK `timeout` is only an HTTP idle timeout - a stuck RUN keeps
    streaming logs and never trips it).

    The temp build context is empty other than the Dockerfile; rollout
    Dockerfiles must be self-contained (FROM + RUN + CMD). Curation
    candidates that depend on a build context need explicit handling.

    Args:
      builder: optional named buildx builder (default uses current).
      cache_dir: optional local cache path for --cache-from / --cache-to.
        On the pod this points at /scratch/bkcache for cross-rollout reuse.
    """
    digest = hashlib.sha256(df_text.encode()).hexdigest()[:12]
    tag = f"dockermin/curate:{digest}"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bldctx_") as ctx:
        ctx_path = Path(ctx)
        (ctx_path / "Dockerfile").write_text(df_text)
        cmd = ["docker", "buildx", "build", "--load", "-t", tag,
               "-f", str(ctx_path / "Dockerfile")]
        if builder:
            cmd += ["--builder", builder]
        if cache_dir:
            cmd += [
                "--cache-from", f"type=local,src={cache_dir}",
                "--cache-to", f"type=local,dest={cache_dir},mode=max",
            ]
        cmd.append(str(ctx_path))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return BuildResult(ok=False, error=f"build timeout after {timeout_s}s")
        if proc.returncode != 0:
            return BuildResult(ok=False,
                               error=f"buildx rc={proc.returncode}: {proc.stderr[-400:]}")
    elapsed = time.perf_counter() - t0
    try:
        size = _docker_client().images.get(tag).attrs["Size"]
    except APIError as e:
        return BuildResult(ok=False, error=f"image inspect failed: {e}")
    return BuildResult(ok=True, tag=tag, size_bytes=size, build_seconds=elapsed)


def run_test_gate(tag: str, cmd: list[str], expected_substring: str, timeout_s: int = 30) -> TestResult:
    """Run cmd inside the image, capture combined output, match expected substring.

    Renamed from test_gate to avoid pytest auto-collection.

    Defense in depth: an empty expected_substring makes the test trivially
    pass (``"" in anything == True``). That is a reward-hacking surface
    if a triple slipped through with no expectation. Reject at the gate.
    """
    if not expected_substring:
        return TestResult(ok=False,
                          error="empty expected_substring would trivially pass; refuse")
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
                 build_timeout_s: int = 300, test_timeout_s: int = 30,
                 builder: str | None = None, cache_dir: str | None = None) -> AnnotateResult:
    p = parse_gate(df_text)
    if not p.ok:
        return AnnotateResult(ok=False, error=p.error)
    b = build_gate(df_text, timeout_s=build_timeout_s,
                   builder=builder, cache_dir=cache_dir)
    if not b.ok:
        return AnnotateResult(ok=False, error=b.error)
    t = run_test_gate(b.tag, test_cmd, expected_substring, timeout_s=test_timeout_s)
    if not t.ok:
        # Best-effort cleanup of the tagged image so the disk doesn't fill.
        try:
            _docker_client().images.remove(b.tag, force=True)
        except Exception:
            pass
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
        # /app/server is a guess; we cannot give an empty expected (reward
        # hacking surface) so drop the triple if we cannot verify. The bulk
        # annotator will skip these without a meaningful test.
        return (None, None)
    if "from openjdk" in text or "from eclipse-temurin" in text:
        # java -version writes to stderr; the openjdk/temurin runtime always
        # prints a line starting with "openjdk" or "OpenJDK". Match on that.
        return (["java", "-version"], "openjdk")
    return (None, None)
