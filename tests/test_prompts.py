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


def test_format_messages_includes_dockerfile() -> None:
    msgs = format_messages("FROM python\n")
    assert any(m["role"] == "system" for m in msgs)
    assert any("FROM python" in m["content"] for m in msgs)


def test_prompt_does_not_leak_expected_substring() -> None:
    """The model must NOT see the test_cmd or expected output - that is the
    reward-hack vector (FROM scratch + RUN echo <expected>)."""
    msgs = format_messages("FROM python:3.12\nRUN pip install flask\n")
    blob = " ".join(m["content"] for m in msgs)
    assert "expected" not in blob.lower()
    assert "test_cmd" not in blob.lower()
    assert "ok 3.0.0" not in blob  # a sample expected value must never appear
