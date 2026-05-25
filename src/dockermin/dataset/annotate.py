"""Per-Dockerfile annotation pipeline: parse, build, test."""

from __future__ import annotations

import contextlib
import hashlib
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import docker
import dockerfile
from docker.errors import APIError, NotFound

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _docker_client() -> docker.DockerClient:
    return docker.from_env(timeout=600)


def parse_gate(df_text: str) -> ParseResult:
    """Validate Dockerfile parses and has at least MIN_COMMANDS instructions."""
    try:
        cmds = dockerfile.parse_string(df_text)
    except dockerfile.GoParseError as e:
        return ParseResult(ok=False, error=f"parse error: {e}")
    if len(cmds) < MIN_COMMANDS:
        return ParseResult(ok=False, command_count=len(cmds), error=f"too short: {len(cmds)} < minimum {MIN_COMMANDS}")
    return ParseResult(ok=True, command_count=len(cmds))


def _buildx_command(
    tag: str, dockerfile_path: Path, context_path: Path, builder: str | None, cache_dir: str | None
) -> list[str]:
    """Assemble the `docker buildx build` argv for this build."""
    cmd = ["docker", "buildx", "build", "--load", "-t", tag, "-f", str(dockerfile_path)]
    if builder:
        cmd += ["--builder", builder]
    if cache_dir:
        cmd += [
            "--cache-from",
            f"type=local,src={cache_dir}",
            "--cache-to",
            f"type=local,dest={cache_dir},mode=max",
        ]
    cmd.append(str(context_path))
    return cmd


def _run_buildx(df_text: str, tag: str, timeout_s: int, builder: str | None, cache_dir: str | None) -> str | None:
    """Run `docker buildx build` in a temp context; return an error or None."""
    with tempfile.TemporaryDirectory(prefix="bldctx_") as ctx:
        ctx_path = Path(ctx)
        (ctx_path / "Dockerfile").write_text(df_text)
        cmd = _buildx_command(tag, ctx_path / "Dockerfile", ctx_path, builder, cache_dir)
        try:
            # Fixed `docker buildx` argv (no shell); returncode checked below.
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"build timeout after {timeout_s}s"
    if proc.returncode != 0:
        return f"buildx rc={proc.returncode}: {proc.stderr[-400:]}"
    return None


def build_gate(
    df_text: str, timeout_s: int = 300, builder: str | None = None, cache_dir: str | None = None
) -> BuildResult:
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
    build_error = _run_buildx(df_text, tag, timeout_s, builder, cache_dir)
    if build_error is not None:
        return BuildResult(ok=False, error=build_error)
    elapsed = time.perf_counter() - t0
    try:
        size = _docker_client().images.get(tag).attrs["Size"]
    except APIError as e:
        return BuildResult(ok=False, error=f"image inspect failed: {e}")
    return BuildResult(ok=True, tag=tag, size_bytes=size, build_seconds=elapsed)


def _evaluate_run(container: docker.models.containers.Container, expected_substring: str, timeout_s: int) -> TestResult:
    """Wait for the container, then match exit code and expected substring."""
    status = container.wait(timeout=timeout_s)
    exit_code = status.get("StatusCode")
    stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
    stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    combined = stdout + "\n" + stderr
    if exit_code != 0:
        return TestResult(ok=False, output=combined, exit_code=exit_code, error=f"non-zero exit {exit_code}")
    if expected_substring not in combined:
        return TestResult(ok=False, output=combined, exit_code=exit_code, error="expected substring not found")
    return TestResult(ok=True, output=combined, exit_code=exit_code)


def run_test_gate(tag: str, cmd: list[str], expected_substring: str, timeout_s: int = 30) -> TestResult:
    """Run cmd inside the image, capture combined output, match expected substring.

    Renamed from test_gate to avoid pytest auto-collection.

    Defense in depth: an empty expected_substring makes the test trivially
    pass (``"" in anything == True``). That is a reward-hacking surface
    if a triple slipped through with no expectation. Reject at the gate.
    """
    if not expected_substring:
        return TestResult(ok=False, error="empty expected_substring would trivially pass; refuse")
    client = _docker_client()
    try:
        container = client.containers.run(
            tag,
            command=cmd,
            detach=True,
            network_mode="bridge",
            mem_limit="1g",
            memswap_limit="1g",
            nano_cpus=2_000_000_000,
            pids_limit=512,
        )
    except APIError as e:
        return TestResult(ok=False, error=f"start error: {e}")
    try:
        return _evaluate_run(container, expected_substring, timeout_s)
    finally:
        # Best-effort cleanup: the container may already be gone (NotFound) or
        # the daemon may transiently error (APIError); neither should mask the
        # gate result.
        with contextlib.suppress(APIError, NotFound):
            container.remove(force=True)


def annotate_one(  # noqa: PLR0913 — parse/build/test gate inputs + per-gate timeouts/builder are an irreducible pipeline contract
    df_text: str,
    test_cmd: list[str],
    expected_substring: str,
    build_timeout_s: int = 300,
    test_timeout_s: int = 30,
    builder: str | None = None,
    cache_dir: str | None = None,
) -> AnnotateResult:
    p = parse_gate(df_text)
    if not p.ok:
        return AnnotateResult(ok=False, error=p.error)
    b = build_gate(df_text, timeout_s=build_timeout_s, builder=builder, cache_dir=cache_dir)
    if not b.ok:
        return AnnotateResult(ok=False, error=b.error)
    t = run_test_gate(b.tag, test_cmd, expected_substring, timeout_s=test_timeout_s)
    # Best-effort cleanup of the tagged image regardless of outcome: callers
    # only need size + test_ok. Keeping every tag fills the disk fast under
    # parallel curation (8 workers x 200 candidates x ~150MB base = ~250GB).
    # Already gone -> NotFound; daemon hiccup -> APIError; neither must fail us.
    with contextlib.suppress(APIError, NotFound):
        _docker_client().images.remove(b.tag, force=True)
    if not t.ok:
        return AnnotateResult(ok=False, error=t.error)
    return AnnotateResult(ok=True, baseline_size=b.size_bytes, baseline_build_s=b.build_seconds, tag=b.tag)


# Runtime probes that require the real interpreter/runtime to exist inside the
# built image. The probe runs INSIDE the image, so a `FROM scratch + RUN echo`
# cheat (no interpreter) exits non-zero and fails the gate - the model cannot
# satisfy a runtime import by echoing a constant at build time.
#
# python: import the installed package, fall back to a stdlib module when the
#   package is unknown, then print a marker. The import requires a python
#   interpreter -> a scratch image fails.
# node: require the installed module + print a marker; a missing module exits
#   non-zero, so the require itself is the runtime check.
# java: `java -version` already needs the JRE; the openjdk/temurin runtime
#   prints a line starting with "openjdk"/"OpenJDK" on stderr.
# go/ruby/php/rust: a toolchain/interpreter version check is the floor. Each
#   needs the real toolchain inside the image (`go version`, `ruby -e`,
#   `php -r`, `rustc --version`), so a `FROM scratch + RUN echo` cheat lacking
#   the toolchain exits non-zero and fails the gate. Where a package is named
#   (ruby `gem install`) the import layer still applies; go/php/rust have no
#   cheap generic import, so the version marker is the floor.
_PYTHON_MARKER = "PYOK"
_NODE_MARKER = "NODEOK"
_RUBY_MARKER = "RUBYOK"
_PHP_MARKER = "PHPOK"

# A python module that always exists in any CPython runtime, used when the
# Dockerfile names no pip-installable package we can import.
_PYTHON_STDLIB_FALLBACK = "json"

# Pull the first non-flag token after `pip install` / `npm install` so the probe
# imports a package that this Dockerfile actually installs.
_PIP_INSTALL_RE = re.compile(r"pip(?:3)?\s+install\s+((?:-[^\s]+\s+)*)([^\s]+)", re.IGNORECASE)
# Anchor the npm match to a single line (`[^\S\n]` = horizontal whitespace) so
# the flag-skip group cannot consume a newline and grab the next Dockerfile
# instruction (e.g. COPY) as the package name.
_NPM_INSTALL_RE = re.compile(
    r"npm[^\S\n]+(?:install|i|add)((?:[^\S\n]+-[^\s]+)*)(?:[^\S\n]+([^\s]+))?",
    re.IGNORECASE,
)
# First non-flag token after `gem install`, anchored to the same line.
_GEM_INSTALL_RE = re.compile(r"gem[^\S\n]+install(?:[^\S\n]+-[^\s]+)*(?:[^\S\n]+([^\s]+))?", re.IGNORECASE)

# Dockerfile instruction keywords that must never be mistaken for a package
# name when a bare `npm install` is followed by another instruction.
_DOCKERFILE_KEYWORDS = frozenset(
    {
        "from",
        "run",
        "cmd",
        "copy",
        "add",
        "env",
        "arg",
        "workdir",
        "entrypoint",
        "expose",
        "label",
        "user",
        "volume",
        "onbuild",
        "healthcheck",
        "shell",
        "stopsignal",
        "maintainer",
    }
)


def _first_pip_package(text: str) -> str | None:
    """First pip-installed package name (without version/extras), or None."""
    m = _PIP_INSTALL_RE.search(text)
    if not m:
        return None
    token = m.group(2).strip("'\"")
    # Strip version specifiers and extras: "flask==3.0", "uvicorn[standard]".
    name = re.split(r"[=<>!~\[]", token, maxsplit=1)[0].strip().strip("'\"")
    # Distribution names use hyphens; the import name uses underscores.
    return name.replace("-", "_") or None


def _strip_npm_version(token: str) -> str:
    """Drop a trailing @version, keeping a leading @scope/name intact."""
    if token.startswith("@"):
        scope, _, rest = token.partition("/")
        return f"{scope}/{rest.split('@', 1)[0]}" if rest else scope
    return token.split("@", 1)[0]


def _first_npm_package(text: str) -> str | None:
    """First require-able npm module name on the same line as `npm install`.

    Returns None (caller falls back to a version-marker probe) when:
    - there is no package token (bare `npm install` from package.json),
    - the next token is a Dockerfile instruction keyword (the bare install was
      followed by COPY/RUN/... on the next line),
    - the install carries `-g`/`--global` (global modules are not on the bare
      `require()` path).
    """
    m = _NPM_INSTALL_RE.search(text)
    if not m:
        return None
    flags = m.group(1) or ""
    if re.search(r"(?:^|\s)(?:-g|--global)(?:\s|$)", flags):
        return None
    token = m.group(2)
    if token is None or token.lower() in _DOCKERFILE_KEYWORDS:
        return None
    return _strip_npm_version(token) or None


def _python_probe(text: str) -> tuple[list[str], str]:
    package = _first_pip_package(text) or _PYTHON_STDLIB_FALLBACK
    code = f"import sys,importlib; importlib.import_module('{package}'); print('{_PYTHON_MARKER}', sys.version_info[0])"
    return (["python", "-c", code], _PYTHON_MARKER)


def _node_probe(text: str) -> tuple[list[str], str]:
    module = _first_npm_package(text)
    if module is None:
        return (["node", "-e", f"console.log('{_NODE_MARKER}', process.version)"], _NODE_MARKER)
    code = f"require('{module}'); console.log('{_NODE_MARKER}', process.version)"
    return (["node", "-e", code], _NODE_MARKER)


def _first_gem_package(text: str) -> str | None:
    """First non-flag token after `gem install`, or None."""
    m = _GEM_INSTALL_RE.search(text)
    if not m:
        return None
    token = m.group(1)
    if token is None or token.lower() in _DOCKERFILE_KEYWORDS:
        return None
    name = token.strip("'\"").split(":", 1)[0]
    return name or None


def _ruby_probe(text: str) -> tuple[list[str], str]:
    gem = _first_gem_package(text)
    if gem is None:
        return (["ruby", "-e", f"puts '{_RUBY_MARKER}', RUBY_VERSION"], _RUBY_MARKER)
    # Require the installed gem, then print the marker; the require needs the
    # ruby runtime, so a toolchain-less image fails.
    code = f"require '{gem}'; puts '{_RUBY_MARKER}', RUBY_VERSION"
    return (["ruby", "-e", code], _RUBY_MARKER)


def _go_probe() -> tuple[list[str], str]:
    return (["go", "version"], "go version")


def _php_probe() -> tuple[list[str], str]:
    return (["php", "-r", f"echo '{_PHP_MARKER}', PHP_VERSION;"], _PHP_MARKER)


def _rust_probe() -> tuple[list[str], str]:
    return (["rustc", "--version"], "rustc")


# Ecosystem resolution, ordered by strength of evidence. RUN installer signals
# come first (a generic debian/ubuntu/alpine base that `pip install`s exercises
# python at runtime); FROM hints are the fallback. Each entry is a compiled
# regex matched against the lowercased Dockerfile. `\bgo` keeps `cargo build`
# from matching the go signal.
_ECOSYSTEM_SIGNALS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"pip3?\s+install"), "python"),
    (re.compile(r"npm\s+(?:install|i|add)\b"), "node"),
    (re.compile(r"gem\s+install"), "ruby"),
    (re.compile(r"cargo\s+(?:build|install|run)"), "rust"),
    (re.compile(r"\bgo\s+(?:build|install|run)\b"), "go"),
    (re.compile(r"composer"), "php"),
    (re.compile(r"from\s+python"), "python"),
    (re.compile(r"from\s+node"), "node"),
    (re.compile(r"from\s+(?:openjdk|eclipse-temurin)"), "java"),
    (re.compile(r"from\s+ruby"), "ruby"),
    (re.compile(r"from\s+php"), "php"),
    (re.compile(r"from\s+golang"), "go"),
    (re.compile(r"from\s+rust"), "rust"),
)


def _ecosystem(text: str) -> str | None:
    """Resolve the ecosystem from FROM and RUN installer signals (or None)."""
    for pattern, eco in _ECOSYSTEM_SIGNALS:
        if pattern.search(text):
            return eco
    return None


# Probe builders per ecosystem. Each runs INSIDE the built image and needs the
# real interpreter/toolchain, so a `RUN echo` cheat cannot satisfy it.
_PROBE_BUILDERS: dict[str, Callable[[str], tuple[list[str], str]]] = {
    "python": _python_probe,
    "node": _node_probe,
    "ruby": _ruby_probe,
    "java": lambda _text: (["java", "-version"], "openjdk"),
    "php": lambda _text: _php_probe(),
    "go": lambda _text: _go_probe(),
    "rust": lambda _text: _rust_probe(),
}


def infer_test_cmd(df_text: str) -> tuple[list[str] | None, str | None]:
    """Best-effort runtime probe per ecosystem. Returns (None, None) if unknown.

    The probe imports/requires a real package or invokes the toolchain and
    prints a marker, so it requires the interpreter/runtime to exist inside the
    built image - a `RUN echo` cheat at build time cannot satisfy it.
    """
    text = df_text.lower()
    eco = _ecosystem(text)
    if eco is None:
        return (None, None)
    return _PROBE_BUILDERS[eco](text)
