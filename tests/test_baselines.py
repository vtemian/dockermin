"""Tests for the pure Dockerfile transforms in ``dockermin.eval.baselines``.

These cover only the deterministic string-rewriting helpers (no docker, no
model, no network): the hadolint mechanical fixers, the base-image swap, the
consecutive-RUN collapse, and the baseline registry helpers. The model/docker
baselines are intentionally untested here.
"""

from __future__ import annotations

from dockermin.eval import baselines

# ---------------------------------------------------------------------------
# _append_to_line
# ---------------------------------------------------------------------------


def test_append_to_line_plain() -> None:
    assert baselines._append_to_line("RUN x", "rm -rf y") == "RUN x && rm -rf y"


def test_append_to_line_preserves_trailing_backslash() -> None:
    assert baselines._append_to_line("RUN x \\", "rm -rf y") == "RUN x && rm -rf y \\"


# ---------------------------------------------------------------------------
# _append_cleanup_to_matching_lines
# ---------------------------------------------------------------------------


def test_append_cleanup_appends_to_matching_line() -> None:
    out = baselines._append_cleanup_to_matching_lines(
        "RUN yarn install",
        r"\byarn\s+install\b",
        "yarn cache clean",
        "yarn cache clean",
    )
    assert out == "RUN yarn install && yarn cache clean"


def test_append_cleanup_skips_line_with_skip_marker() -> None:
    line = "RUN yarn install && yarn cache clean"
    out = baselines._append_cleanup_to_matching_lines(
        line,
        r"\byarn\s+install\b",
        "yarn cache clean",
        "yarn cache clean",
    )
    assert out == line


def test_append_cleanup_leaves_non_matching_lines_untouched() -> None:
    out = baselines._append_cleanup_to_matching_lines(
        "FROM python:3.12\nWORKDIR /app",
        r"\bapt-get\s+install\b",
        "/var/lib/apt/lists",
        "rm -rf /var/lib/apt/lists/*",
    )
    assert out == "FROM python:3.12\nWORKDIR /app"


# ---------------------------------------------------------------------------
# _fix_dl3009 — apt cache cleanup
# ---------------------------------------------------------------------------


def test_fix_dl3009_appends_apt_cleanup() -> None:
    out = baselines._fix_dl3009("RUN apt-get install -y curl")
    assert out == "RUN apt-get install -y curl && rm -rf /var/lib/apt/lists/*"


def test_fix_dl3009_skips_line_already_cleaning_apt_lists() -> None:
    line = "RUN apt-get install -y curl && rm -rf /var/lib/apt/lists/*"
    assert baselines._fix_dl3009(line) == line


# ---------------------------------------------------------------------------
# _fix_dl3015 — --no-install-recommends
# ---------------------------------------------------------------------------


def test_fix_dl3015_inserts_no_install_recommends() -> None:
    out = baselines._fix_dl3015("RUN apt-get install -y curl")
    assert out == "RUN apt-get install --no-install-recommends -y curl"


def test_fix_dl3015_is_idempotent() -> None:
    once = baselines._fix_dl3015("RUN apt-get install -y curl")
    assert baselines._fix_dl3015(once) == once


# ---------------------------------------------------------------------------
# _fix_dl3019 — apk add --no-cache
# ---------------------------------------------------------------------------


def test_fix_dl3019_inserts_no_cache() -> None:
    out = baselines._fix_dl3019("RUN apk add curl")
    assert out == "RUN apk add --no-cache curl"


def test_fix_dl3019_is_idempotent() -> None:
    once = baselines._fix_dl3019("RUN apk add curl")
    assert baselines._fix_dl3019(once) == once


# ---------------------------------------------------------------------------
# _fix_dl3042 — pip install --no-cache-dir
# ---------------------------------------------------------------------------


def test_fix_dl3042_inserts_no_cache_dir() -> None:
    out = baselines._fix_dl3042("RUN pip install flask")
    assert out == "RUN pip install --no-cache-dir flask"


def test_fix_dl3042_is_idempotent() -> None:
    once = baselines._fix_dl3042("RUN pip install flask")
    assert baselines._fix_dl3042(once) == once


# ---------------------------------------------------------------------------
# _fix_dl3060 — yarn cache clean
# ---------------------------------------------------------------------------


def test_fix_dl3060_appends_yarn_cache_clean() -> None:
    out = baselines._fix_dl3060("RUN yarn install")
    assert out == "RUN yarn install && yarn cache clean"


def test_fix_dl3060_skips_line_already_cleaning_cache() -> None:
    line = "RUN yarn install && yarn cache clean"
    assert baselines._fix_dl3060(line) == line


# ---------------------------------------------------------------------------
# _swap_base_image — known base -> slim/alpine mapping
# ---------------------------------------------------------------------------


def test_swap_base_image_python_to_slim() -> None:
    assert baselines._swap_base_image("FROM python:3.12") == "FROM python:3.12-slim"


def test_swap_base_image_node_to_alpine() -> None:
    assert baselines._swap_base_image("FROM node:20") == "FROM node:20-alpine"


def test_swap_base_image_openjdk_to_temurin() -> None:
    out = baselines._swap_base_image("FROM openjdk:17")
    assert out == "FROM eclipse-temurin:17-jre-alpine"


def test_swap_base_image_preserves_as_stage_alias() -> None:
    out = baselines._swap_base_image("FROM node:20 AS build")
    assert out == "FROM node:20-alpine AS build"


def test_swap_base_image_leaves_unknown_image_untouched() -> None:
    assert baselines._swap_base_image("FROM ubuntu:22.04") == "FROM ubuntu:22.04"


def test_swap_base_image_rewrites_every_matching_from() -> None:
    out = baselines._swap_base_image("FROM python:3.12\nRUN echo hi\nFROM node:20")
    assert out == "FROM python:3.12-slim\nRUN echo hi\nFROM node:20-alpine"


# ---------------------------------------------------------------------------
# _consume_continuations
# ---------------------------------------------------------------------------


def test_consume_continuations_absorbs_backslash_lines() -> None:
    lines = ["RUN a \\", "    b", "RUN c"]
    body_parts = ["a \\"]
    next_i = baselines._consume_continuations(body_parts, lines, 1)
    assert body_parts == ["a", "b"]
    assert next_i == 2


def test_consume_continuations_no_op_without_backslash() -> None:
    lines = ["RUN a", "RUN b"]
    body_parts = ["a"]
    next_i = baselines._consume_continuations(body_parts, lines, 1)
    assert body_parts == ["a"]
    assert next_i == 1


# ---------------------------------------------------------------------------
# _collapse_consecutive_runs
# ---------------------------------------------------------------------------


def test_collapse_two_adjacent_runs() -> None:
    assert baselines._collapse_consecutive_runs("RUN a\nRUN b") == "RUN a \\\n    && b"


def test_collapse_three_adjacent_runs() -> None:
    out = baselines._collapse_consecutive_runs("RUN a\nRUN b\nRUN c")
    assert out == "RUN a \\\n    && b \\\n    && c"


def test_collapse_merges_across_continuations() -> None:
    out = baselines._collapse_consecutive_runs("RUN a \\\n    x\nRUN b")
    assert out == "RUN a \\\n    x \\\n    && b"


def test_collapse_does_not_merge_runs_separated_by_other_directive() -> None:
    df = "RUN a\nWORKDIR /x\nRUN b"
    assert baselines._collapse_consecutive_runs(df) == df


def test_collapse_passes_through_non_run_lines() -> None:
    df = "FROM python:3.12\nWORKDIR /app"
    assert baselines._collapse_consecutive_runs(df) == df


def test_collapse_single_run_unchanged() -> None:
    assert baselines._collapse_consecutive_runs("RUN a") == "RUN a"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_available_baselines_returns_sorted_known_names() -> None:
    names = baselines.available_baselines()
    assert names == sorted(names)
    assert {"hadolint", "manual", "qwen_zs", "gemma_zs", "gpt4o", "sonnet_zs", "slim", "dockermin"} <= set(names)


def test_register_baseline_adds_dispatchable_name() -> None:
    def fake(triple: dict[str, object], **_: object) -> baselines.EvalEntry:
        return baselines.EvalEntry("fake", "x", None, False, 0.0, "")

    baselines.register_baseline("fake_test_baseline", fake)
    assert "fake_test_baseline" in baselines.available_baselines()
