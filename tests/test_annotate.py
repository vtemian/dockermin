"""Tests for the per-Dockerfile annotate() pipeline."""
import pytest
import docker

from dockermin.dataset.annotate import (
    parse_gate, ParseResult,
    build_gate, BuildResult,
    test_gate, TestResult,
    annotate_one, AnnotateResult,
    infer_test_cmd,
)

DOCKER_AVAILABLE = False
try:
    docker.from_env().ping()
    DOCKER_AVAILABLE = True
except Exception:
    pass


def test_parse_gate_accepts_valid_dockerfile():
    df = "FROM python:3.12-slim\nRUN pip install flask\nCMD [\"python\",\"-m\",\"flask\",\"run\"]"
    result = parse_gate(df)
    assert isinstance(result, ParseResult)
    assert result.ok is True
    assert result.command_count == 3


def test_parse_gate_rejects_garbage():
    result = parse_gate("this is not a Dockerfile")
    assert result.ok is False
    assert "parse" in result.error.lower()


def test_parse_gate_rejects_too_short():
    result = parse_gate("FROM scratch")
    assert result.ok is False
    assert "too short" in result.error.lower() or "minimum" in result.error.lower()


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_succeeds_on_minimal_alpine():
    df = "FROM alpine:3.20\nRUN echo hello > /msg\nCMD [\"cat\",\"/msg\"]\n"
    result = build_gate(df, timeout_s=120)
    assert isinstance(result, BuildResult)
    assert result.ok is True
    assert result.size_bytes > 0
    assert result.size_bytes < 20_000_000  # alpine + tiny file should be <20MB
    assert result.tag.startswith("dockermin/curate:")


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_build_gate_fails_on_broken_command():
    df = "FROM alpine:3.20\nRUN exit 1\n"
    result = build_gate(df, timeout_s=60)
    assert result.ok is False
    assert "build" in result.error.lower() or "exit" in result.error.lower()


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_passes_when_substring_present():
    df = "FROM alpine:3.20\nRUN echo readyok > /msg\nCMD [\"cat\",\"/msg\"]\n"
    build = build_gate(df)
    assert build.ok
    result = test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert isinstance(result, TestResult)
    assert result.ok is True
    assert "readyok" in result.output


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_test_gate_fails_when_substring_absent():
    df = "FROM alpine:3.20\nRUN echo nope > /msg\nCMD [\"cat\",\"/msg\"]\n"
    build = build_gate(df)
    assert build.ok
    result = test_gate(build.tag, ["cat", "/msg"], "readyok", timeout_s=30)
    assert result.ok is False


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="docker daemon not available")
def test_annotate_one_happy_path_flask_smoke():
    df = """FROM python:3.12-slim
RUN pip install --no-cache-dir flask==3.0.0
CMD ["python","-c","import flask,sys;print('ok',flask.__version__)"]
"""
    result = annotate_one(df, ["python", "-c", "import flask,sys;print('ok',flask.__version__)"], "ok 3.0.0")
    assert isinstance(result, AnnotateResult)
    assert result.ok is True
    assert result.baseline_size > 50_000_000
    assert result.baseline_build_s > 0


def test_infer_test_cmd_python():
    df = "FROM python:3.12-slim\nRUN pip install flask\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd[0] == "python"
    assert "ok" in expected.lower()


def test_infer_test_cmd_node():
    df = "FROM node:20-alpine\nRUN npm install express\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd[0] == "node"


def test_infer_test_cmd_falls_back_to_none_on_unknown():
    df = "FROM scratch\nCOPY app /app\n"
    cmd, expected = infer_test_cmd(df)
    assert cmd is None and expected is None
