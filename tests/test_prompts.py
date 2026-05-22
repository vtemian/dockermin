"""Tests for prompt template and extractor."""
from dockermin.reward.prompts import (
    SYSTEM_PROMPT, USER_TEMPLATE, format_messages, extract_dockerfile,
)

def test_extract_dockerfile_from_fenced_block():
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

def test_extract_dockerfile_handles_bare_fence():
    text = "```\nFROM alpine\nRUN echo hi\n```"
    df = extract_dockerfile(text)
    assert df is not None and "FROM alpine" in df

def test_extract_dockerfile_returns_none_on_no_fence():
    assert extract_dockerfile("just prose, no code") is None

def test_format_messages_includes_dockerfile_and_test_cmd():
    msgs = format_messages("FROM python\n", ["python","-c","print('ok')"], "ok")
    assert any("system" == m["role"] for m in msgs)
    assert any("python" in m["content"] for m in msgs)
