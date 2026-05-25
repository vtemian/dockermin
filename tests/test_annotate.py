"""Tests for the per-Dockerfile annotate() pipeline."""

from __future__ import annotations

import docker
import pytest

from dockermin.dataset.annotate import (
    AnnotateResult,
    BuildResult,
    ParseResult,
    TestResult,
    annotate_one,
    build_gate,
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


def test_infer_test_cmd_go_returns_none() -> None:
    """Go: we cannot synthesize a safe runtime probe, so drop the triple."""
    cmd, expected = infer_test_cmd("FROM golang:1.22\nRUN go build -o /server\n")
    assert cmd is None and expected is None


def test_infer_test_cmd_falls_back_to_none_on_unknown() -> None:
    df = "FROM scratch\nCOPY app /app\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd is None and expected is None
