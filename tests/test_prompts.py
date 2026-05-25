"""Tests for prompt template and extractor."""

from __future__ import annotations

from dockermin.reward.prompts import (
    extract_dockerfile,
    format_messages,
)


def test_extract_dockerfile_from_fenced_block() -> None:
    text = """Sure, here is the optimized Dockerfile:

```dockerfile
FROM python:3.12-alpine
RUN pip install flask
```

That should be ~50MB smaller.
"""
    df = extract_dockerfile(text)
    assert df is not None
    assert df.startswith("FROM python:3.12-alpine")
    assert "pip install flask" in df


def test_extract_dockerfile_handles_bare_fence() -> None:
    text = "```\nFROM alpine\nRUN echo hi\n```"
    df = extract_dockerfile(text)
    assert df is not None and "FROM alpine" in df


def test_extract_dockerfile_returns_none_on_no_fence() -> None:
    assert extract_dockerfile("just prose, no code") is None


def test_format_messages_includes_dockerfile_and_test_cmd() -> None:
    msgs = format_messages("FROM python\n", ["python", "-c", "print('ok')"], "ok")
    assert any(m["role"] == "system" for m in msgs)
    assert any("python" in m["content"] for m in msgs)
