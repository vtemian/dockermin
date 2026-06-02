"""Tests for the per-Dockerfile annotate() pipeline."""

from __future__ import annotations

import docker
import pytest

from dockermin.dataset.annotate import (
    AnnotateResult,
    BuildResult,
    ManifestResult,
    ParseResult,
    TestResult,
    annotate_one,
    build_gate,
    find_from_images,
    infer_test_cmd,
    parse_gate,
    run_test_gate,
)

try:
    docker.from_env().ping()
    DOCKER_AVAILABLE = True
except Exception:  # noqa: BLE001 — probe: any failure means the docker daemon is unavailable, skip docker-gated tests
    DOCKER_AVAILABLE = False


def test_parse_gate_accepts_valid_dockerfile() -> None:
    df = 'FROM python:3.12-slim\nRUN pip install flask\nCMD ["python","-m","flask","run"]'
    result = parse_gate(df)
    assert isinstance(result, ParseResult)
    assert result.ok is True
    assert result.command_count == 3


def test_parse_gate_rejects_empty_input() -> None:
    # The dockerfile lib is extremely lenient - even '@@@' parses as 1
    # instruction. The only way to trigger GoParseError on plain strings is
    # the empty-input path ("file with no instructions").
    result = parse_gate("")
    assert result.ok is False
    assert "parse" in result.error.lower()


def test_parse_gate_rejects_too_short() -> None:
    result = parse_gate("FROM scratch")
    assert result.ok is False
    assert "too short" in result.error.lower() or "minimum" in result.error.lower()


@pytest.mark.docker
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_succeeds_on_minimal_alpine() -> None:
    df = 'FROM alpine:3.20\nRUN echo hello > /msg\nCMD ["cat","/msg"]\n'
    result = build_gate(df, timeout_s=120)
    assert isinstance(result, BuildResult)
    assert result.ok is True
    assert result.size_bytes > 0
    assert result.size_bytes < 20_000_000  # alpine + tiny file should be <20MB
    assert result.tag.startswith("dockermin/curate:")


@pytest.mark.docker
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_fails_on_broken_command() -> None:
    df = "FROM alpine:3.20\nRUN exit 1\n"
    result = build_gate(df, timeout_s=60)
    assert result.ok is False
    assert "build" in result.error.lower() or "exit" in result.error.lower()


@pytest.mark.docker
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_passes_when_substring_present() -> None:
    df = 'FROM alpine:3.20\nRUN echo readyok > /msg\nCMD ["cat","/msg"]\n'
    build = build_gate(df)
    assert build.ok
    result = run_test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert isinstance(result, TestResult)
    assert result.ok is True
    assert "readyok" in result.output


@pytest.mark.docker
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_fails_when_substring_absent() -> None:
    df = 'FROM alpine:3.20\nRUN echo nope > /msg\nCMD ["cat","/msg"]\n'
    build = build_gate(df)
    assert build.ok
    result = run_test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert result.ok is False


@pytest.mark.docker
@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_annotate_one_happy_path_flask_smoke() -> None:
    df = """FROM python:3.12-slim
RUN pip install --no-cache-dir flask==3.0.0
CMD ["python","-c","import flask,sys;print('ok',flask.__version__)"]
"""
    result = annotate_one(df, ["python", "-c", "import flask,sys;print('ok',flask.__version__)"], "ok 3.0.0")
    assert isinstance(result, AnnotateResult)
    assert result.ok is True
    assert result.baseline_size > 50_000_000
    assert result.baseline_build_s > 0


def test_infer_test_cmd_python_imports_real_package() -> None:
    """The python probe must import a real installed package and print a marker.

    Importing the package requires a python interpreter to exist in the image,
    so a ``FROM scratch + RUN echo`` cheat (no interpreter) exits non-zero and
    fails the gate.
    """
    cmd, expected = infer_test_cmd("FROM python:3.12-slim\nRUN pip install flask\n")
    assert cmd is not None
    assert expected is not None
    assert cmd[0] == "python"
    assert "import" in " ".join(cmd)
    assert "flask" in " ".join(cmd)
    assert expected == "PYOK"


def test_infer_test_cmd_python_falls_back_to_stdlib_when_package_unknown() -> None:
    """When no installable package is named, fall back to importing a stdlib
    module. This still requires a python interpreter, so a scratch image fails."""
    cmd, expected = infer_test_cmd("FROM python:3.12-slim\nCOPY app.py /app.py\n")
    assert cmd is not None
    assert cmd[0] == "python"
    assert "import" in " ".join(cmd)
    assert expected == "PYOK"


def test_infer_test_cmd_node_requires_module() -> None:
    """The node probe must ``require`` a real module; a missing module exits
    non-zero and fails the gate."""
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install express\n")
    assert cmd is not None
    assert cmd[0] == "node"
    assert "require" in " ".join(cmd)
    assert "express" in " ".join(cmd)
    assert expected == "NODEOK"


def test_infer_test_cmd_java_keeps_jre_probe() -> None:
    cmd, expected = infer_test_cmd("FROM eclipse-temurin:21\nCOPY app.jar /app.jar\n")
    assert cmd is not None
    assert cmd[0] == "java"
    assert expected == "openjdk"


def test_first_pip_package_strips_quotes() -> None:
    """A quoted pip spec (`pip install "hy == 1.0"`) must not leak the leading
    quote into the import name (`import_module('"hy')`)."""
    cmd, _ = infer_test_cmd('FROM python:3.12\nRUN pip install "hy == 1.0"\n')
    assert cmd is not None
    joined = " ".join(cmd)
    assert "'\"hy'" not in joined and '"hy' not in joined
    assert "import_module('hy')" in joined or "'hy'" in joined


def test_npm_bare_install_falls_back_to_version_marker() -> None:
    """`npm install` with no package (install-from-package.json) must NOT grab
    the next Dockerfile line (COPY) as the module name."""
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install\nCOPY . .\n")
    assert cmd is not None
    joined = " ".join(cmd)
    assert "COPY" not in joined and "copy" not in joined
    assert "require(" not in joined
    assert expected == "NODEOK"  # version-marker fallback


def test_npm_global_install_uses_version_marker() -> None:
    """`npm install -g X` installs a global module not on the bare require()
    path, so the probe falls back to the version marker."""
    cmd, expected = infer_test_cmd("FROM node:20\nRUN npm install -g tough-cookie\n")
    assert cmd is not None
    assert "require(" not in " ".join(cmd)  # falls back, doesn't require() a global
    assert expected == "NODEOK"


def test_infer_test_cmd_go() -> None:
    """Go now returns a toolchain probe; `go version` needs the go toolchain
    present, which a `FROM scratch + RUN echo` cheat lacks."""
    cmd, expected = infer_test_cmd("FROM golang:1.22\nRUN go build ./...\n")
    assert cmd is not None and cmd[0] == "go" and expected == "go version"


def test_infer_test_cmd_ruby() -> None:
    cmd, expected = infer_test_cmd("FROM ruby:3.3\nRUN gem install rails\n")
    assert cmd is not None and cmd[0] == "ruby" and expected == "RUBYOK"


def test_infer_test_cmd_php() -> None:
    cmd, expected = infer_test_cmd("FROM php:8.3\nRUN docker-php-ext-install pdo\n")
    assert cmd is not None and cmd[0] == "php" and expected == "PHPOK"


def test_infer_test_cmd_rust() -> None:
    cmd, expected = infer_test_cmd("FROM rust:1.78\nRUN cargo build --release\n")
    assert cmd is not None and cmd[0] == "rustc" and expected == "rustc"


def test_ecosystem_from_run_line_not_just_from() -> None:
    """A generic debian base that `pip install`s must resolve to python, not
    fall through to (None, None)."""
    cmd, expected = infer_test_cmd("FROM debian:bookworm\nRUN apt-get install -y python3-pip && pip install flask\n")
    assert cmd is not None and cmd[0] == "python"
    assert expected == "PYOK"


def test_ecosystem_from_run_line_gem_resolves_ruby() -> None:
    """A generic ubuntu base that `gem install`s must resolve to ruby."""
    cmd, expected = infer_test_cmd("FROM ubuntu:24.04\nRUN gem install sinatra\n")
    assert cmd is not None and cmd[0] == "ruby" and expected == "RUBYOK"


def test_ecosystem_from_run_line_go_build_resolves_go() -> None:
    cmd, expected = infer_test_cmd("FROM alpine:3.20\nRUN go build ./...\n")
    assert cmd is not None and cmd[0] == "go" and expected == "go version"


def test_infer_test_cmd_falls_back_to_none_on_unknown() -> None:
    df = "FROM scratch\nCOPY app /app\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd is None and expected is None


def test_find_from_images_single_from() -> None:
    df = 'FROM python:3.12-slim\nRUN echo hi\nCMD ["python"]'
    assert find_from_images(df) == ["python:3.12-slim"]


def test_find_from_images_multistage_deduped() -> None:
    """Multi-stage: returns each unique registry image once; stage aliases ignored."""
    df = (
        "FROM golang:1.22 AS builder\n"
        "RUN go build -o app\n"
        "FROM gcr.io/distroless/base\n"
        "COPY --from=builder /app /app\n"
        "FROM golang:1.22 AS tester\n"
        'CMD ["/app"]\n'
    )
    assert sorted(find_from_images(df)) == ["gcr.io/distroless/base", "golang:1.22"]


def test_find_from_images_skips_scratch_and_aliases() -> None:
    df = (
        "FROM scratch AS base\n"
        "COPY app /app\n"
        "FROM base AS final\n"  # 'base' is a stage alias, not a registry image
        'CMD ["/app"]\n'
    )
    assert find_from_images(df) == []


def test_find_from_images_malformed_returns_empty() -> None:
    """Unparseable Dockerfile returns empty list rather than raising."""
    assert find_from_images("") == []


def test_manifest_result_is_frozen_dataclass() -> None:
    """ManifestResult must be a frozen value type with ok/missing/error fields."""
    result = ManifestResult(ok=False, missing=("python:bogus",), error="not found")
    assert result.ok is False
    assert result.missing == ("python:bogus",)
    assert result.error == "not found"
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass disallows mutation
        result.ok = True  # type: ignore[misc]
