"""Tests for the pure (no-network) scrape helpers."""

from __future__ import annotations

from dockermin.dataset.scrape import (
    _ecosystem_from_dockerfile,
    _infer_ecosystem,
    _is_new_content,
    _is_self_contained_probeable,
    _manifest_raw_url,
    _merge_stanza,
    _parse_manifest_stanzas,
    _parse_official_manifest,
    _parse_stanza_block,
    _split_manifest_blocks,
    _strip_github_repo,
)

# --- _infer_ecosystem ---------------------------------------------------------


def test_infer_ecosystem_known_base_maps_directly() -> None:
    assert _infer_ecosystem("python") == "python"


def test_infer_ecosystem_alias_maps_to_canonical() -> None:
    assert _infer_ecosystem("golang") == "go"
    assert _infer_ecosystem("mariadb") == "mysql"


def test_infer_ecosystem_unknown_name_returns_unknown() -> None:
    assert _infer_ecosystem("totally-made-up-image") == "unknown"


def test_infer_ecosystem_empty_returns_unknown() -> None:
    assert _infer_ecosystem("") == "unknown"


def test_infer_ecosystem_is_case_insensitive_and_strips() -> None:
    assert _infer_ecosystem("  PYTHON  ") == "python"


def test_infer_ecosystem_prefix_match_on_hyphen() -> None:
    # Unknown full key, but the head before the first hyphen is known.
    assert _infer_ecosystem("python-slim") == "python"


def test_infer_ecosystem_no_known_prefix_returns_unknown() -> None:
    assert _infer_ecosystem("frobnicate-xyz") == "unknown"


# --- _split_manifest_blocks ---------------------------------------------------


def test_split_manifest_blocks_separates_on_blank_lines() -> None:
    text = "A: 1\nB: 2\n\nC: 3\n"
    blocks = _split_manifest_blocks(text)
    assert blocks == ["A: 1\nB: 2", "C: 3"]


def test_split_manifest_blocks_treats_whitespace_only_line_as_separator() -> None:
    text = "A: 1\n   \nB: 2"
    blocks = _split_manifest_blocks(text)
    assert blocks == ["A: 1", "B: 2"]


def test_split_manifest_blocks_drops_empty_blocks() -> None:
    # Leading/trailing blank lines yield no empty blocks; only non-empty blocks
    # survive. Blocks are kept verbatim (not stripped), so a trailing blank line
    # can leave a trailing newline on the final block.
    text = "\n\nA: 1\n\nB: 2\n\n"
    blocks = _split_manifest_blocks(text)
    assert blocks == ["A: 1", "B: 2\n"]


# --- _parse_stanza_block ------------------------------------------------------


def test_parse_stanza_block_builds_key_value_map() -> None:
    block = "GitRepo: https://github.com/x/y.git\nGitCommit: abc123"
    stanza = _parse_stanza_block(block)
    assert stanza == {"GitRepo": "https://github.com/x/y.git", "GitCommit": "abc123"}


def test_parse_stanza_block_ignores_comments_and_non_kv_lines() -> None:
    block = "# a comment\nGitRepo: https://github.com/x/y\nnocolon line"
    stanza = _parse_stanza_block(block)
    assert stanza == {"GitRepo": "https://github.com/x/y"}


def test_parse_stanza_block_strips_surrounding_whitespace() -> None:
    block = "  Directory :  3.12/slim  "
    stanza = _parse_stanza_block(block)
    assert stanza == {"Directory": "3.12/slim"}


# --- _parse_manifest_stanzas --------------------------------------------------


def test_parse_manifest_stanzas_returns_one_dict_per_block() -> None:
    text = "Maintainers: x\n\nTags: 3.12\nGitCommit: aaa\n\nTags: 3.11\nGitCommit: bbb"
    stanzas = _parse_manifest_stanzas(text)
    assert len(stanzas) == 3
    assert stanzas[0] == {"Maintainers": "x"}
    assert stanzas[2] == {"Tags": "3.11", "GitCommit": "bbb"}


def test_parse_manifest_stanzas_drops_empty_stanzas() -> None:
    # A block that is all comments produces an empty dict and is dropped.
    text = "# only a comment\n\nTags: 3.12\nGitCommit: aaa"
    stanzas = _parse_manifest_stanzas(text)
    assert stanzas == [{"Tags": "3.12", "GitCommit": "aaa"}]


# --- _merge_stanza ------------------------------------------------------------


def test_merge_stanza_inherits_defaults_when_defaults_lack_tags() -> None:
    defaults = {"GitRepo": "https://github.com/x/y", "Maintainers": "x"}
    stanza = {"Tags": "3.12", "GitCommit": "aaa"}
    merged = _merge_stanza(stanza, defaults)
    assert merged == {
        "GitRepo": "https://github.com/x/y",
        "Maintainers": "x",
        "Tags": "3.12",
        "GitCommit": "aaa",
    }


def test_merge_stanza_override_wins_over_default() -> None:
    defaults = {"GitRepo": "https://github.com/default/repo"}
    stanza = {"GitRepo": "https://github.com/override/repo", "GitCommit": "aaa"}
    merged = _merge_stanza(stanza, defaults)
    assert merged["GitRepo"] == "https://github.com/override/repo"


def test_merge_stanza_with_tags_in_defaults_copies_full_defaults() -> None:
    # When defaults already carry Tags, the whole defaults dict is the base.
    defaults = {"Tags": "latest", "GitRepo": "https://github.com/x/y", "GitCommit": "aaa"}
    stanza = {"Tags": "3.12"}
    merged = _merge_stanza(stanza, defaults)
    assert merged == {"Tags": "3.12", "GitRepo": "https://github.com/x/y", "GitCommit": "aaa"}


# --- _parse_official_manifest -------------------------------------------------

_SAMPLE_MANIFEST = """\
# this is a manifest
Maintainers: Docker (@docker)
GitRepo: https://github.com/docker-library/python.git

Tags: 3.12, 3.12.0, latest
GitCommit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
Directory: 3.12/slim

Tags: 3.11
GitCommit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
Directory: 3.11/slim
"""


def test_parse_official_manifest_inherits_gitrepo_from_defaults() -> None:
    stanzas = _parse_official_manifest(_SAMPLE_MANIFEST)
    assert len(stanzas) == 2
    for stanza in stanzas:
        assert stanza["GitRepo"] == "https://github.com/docker-library/python.git"


def test_parse_official_manifest_extracts_per_tag_commit_and_directory() -> None:
    stanzas = _parse_official_manifest(_SAMPLE_MANIFEST)
    assert stanzas[0]["GitCommit"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert stanzas[0]["Directory"] == "3.12/slim"
    assert stanzas[1]["GitCommit"] == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert stanzas[1]["Directory"] == "3.11/slim"


def test_parse_official_manifest_skips_stanzas_missing_repo_or_commit() -> None:
    text = "GitRepo: https://github.com/x/y\n\nTags: 3.12\nDirectory: foo"
    # The single tag stanza has no GitCommit so it is dropped.
    assert _parse_official_manifest(text) == []


def test_parse_official_manifest_empty_text_returns_empty() -> None:
    assert _parse_official_manifest("") == []


# --- _strip_github_repo -------------------------------------------------------


def test_strip_github_repo_removes_prefix_and_git_suffix() -> None:
    assert _strip_github_repo("https://github.com/docker-library/python.git") == "docker-library/python"


def test_strip_github_repo_without_suffix() -> None:
    assert _strip_github_repo("https://github.com/owner/repo") == "owner/repo"


def test_strip_github_repo_passthrough_when_no_prefix() -> None:
    assert _strip_github_repo("owner/repo") == "owner/repo"


# --- _manifest_raw_url --------------------------------------------------------


def test_manifest_raw_url_builds_raw_githubusercontent_url() -> None:
    url = _manifest_raw_url(_SAMPLE_MANIFEST)
    assert url == (
        "https://raw.githubusercontent.com/docker-library/python/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/3.12/slim/Dockerfile"
    )


def test_manifest_raw_url_uses_custom_file_name() -> None:
    text = "GitRepo: https://github.com/x/y\n\nTags: t\nGitCommit: ccc\nDirectory: dir\nFile: Dockerfile.alpine"
    url = _manifest_raw_url(text)
    assert url == "https://raw.githubusercontent.com/x/y/ccc/dir/Dockerfile.alpine"


def test_manifest_raw_url_without_directory_omits_path_segment() -> None:
    text = "GitRepo: https://github.com/x/y\n\nTags: t\nGitCommit: ccc"
    url = _manifest_raw_url(text)
    assert url == "https://raw.githubusercontent.com/x/y/ccc/Dockerfile"


def test_manifest_raw_url_returns_none_for_empty_manifest() -> None:
    assert _manifest_raw_url("") is None


# --- _ecosystem_from_dockerfile -----------------------------------------------


def test_ecosystem_from_dockerfile_reads_first_from_line() -> None:
    df = "# comment\nFROM python:3.12-slim\nRUN echo hi"
    assert _ecosystem_from_dockerfile(df) == "python"


def test_ecosystem_from_dockerfile_strips_registry_and_tag() -> None:
    df = "FROM docker.io/library/node:20-alpine"
    assert _ecosystem_from_dockerfile(df) == "node"


def test_ecosystem_from_dockerfile_unknown_base_returns_unknown() -> None:
    df = "FROM mystery-base:1.0"
    assert _ecosystem_from_dockerfile(df) == "unknown"


def test_ecosystem_from_dockerfile_no_from_returns_unknown() -> None:
    df = "RUN echo nothing-here"
    assert _ecosystem_from_dockerfile(df) == "unknown"


def test_ecosystem_from_dockerfile_bare_from_returns_unknown() -> None:
    # A FROM line with no base ref has too few parts.
    df = "FROM "
    assert _ecosystem_from_dockerfile(df) == "unknown"


def test_ecosystem_from_dockerfile_golang_base() -> None:
    df = "FROM golang:1.22\nRUN go build ./...\n"
    assert _ecosystem_from_dockerfile(df) == "go"


def test_ecosystem_from_dockerfile_resolves_from_pip_run_line() -> None:
    # A generic debian base with a pip install resolves to python, not unknown.
    df = "FROM debian:bookworm\nRUN apt-get install -y python3-pip && pip install flask\n"
    assert _ecosystem_from_dockerfile(df) == "python"


def test_ecosystem_from_dockerfile_resolves_from_npm_run_line() -> None:
    df = "FROM ubuntu:22.04\nRUN npm install express\n"
    assert _ecosystem_from_dockerfile(df) == "node"


def test_ecosystem_from_dockerfile_resolves_from_gem_run_line() -> None:
    df = "FROM debian:bookworm\nRUN gem install rails\n"
    assert _ecosystem_from_dockerfile(df) == "ruby"


def test_ecosystem_from_dockerfile_resolves_from_composer_run_line() -> None:
    df = "FROM debian:bookworm\nRUN composer install\n"
    assert _ecosystem_from_dockerfile(df) == "php"


def test_ecosystem_from_dockerfile_resolves_from_go_build_run_line() -> None:
    df = "FROM alpine:3.20\nRUN go build ./...\n"
    assert _ecosystem_from_dockerfile(df) == "go"


def test_ecosystem_from_dockerfile_from_takes_precedence_when_known() -> None:
    # A known runtime FROM wins; the RUN-line fallback only kicks in for
    # generic/unknown bases.
    df = "FROM golang:1.22\nRUN pip install something\n"
    assert _ecosystem_from_dockerfile(df) == "go"


# --- _is_self_contained_probeable ---------------------------------------------


def test_self_contained_clean_pip_dockerfile_is_true() -> None:
    assert _is_self_contained_probeable("FROM python:3.12\nRUN pip install flask\n") is True


def test_self_contained_rejects_copy() -> None:
    df = "FROM python:3.12\nCOPY . /app\nRUN pip install -r reqs\n"
    assert _is_self_contained_probeable(df) is False


def test_self_contained_rejects_local_add() -> None:
    assert _is_self_contained_probeable("FROM scratch\nADD x.tar /\n") is False


def test_self_contained_allows_add_http_url() -> None:
    # A remote ADD fetches over the network, so it does not need a local build
    # context. The Dockerfile must still resolve to a known ecosystem.
    df = "FROM python:3.12\nADD https://example.com/x.tar /tmp/\nRUN pip install flask\n"
    assert _is_self_contained_probeable(df) is True


def test_self_contained_rejects_unknown_ecosystem() -> None:
    df = "FROM mystery-base:1.0\nRUN make\n"
    assert _is_self_contained_probeable(df) is False


def test_self_contained_copy_match_is_anchored_to_instruction() -> None:
    # A 'COPY' appearing inside a comment or word should not trip the gate.
    df = "FROM python:3.12\n# we deliberately avoid COPYing the repo\nRUN pip install flask\n"
    assert _is_self_contained_probeable(df) is True


# --- _is_new_content ----------------------------------------------------------


def test_is_new_content_records_first_occurrence() -> None:
    seen: set[str] = set()
    assert _is_new_content("FROM python", seen) is True
    assert len(seen) == 1


def test_is_new_content_detects_duplicate() -> None:
    seen: set[str] = set()
    _is_new_content("FROM python", seen)
    assert _is_new_content("FROM python", seen) is False


def test_is_new_content_distinct_text_is_new() -> None:
    seen: set[str] = set()
    _is_new_content("FROM python", seen)
    assert _is_new_content("FROM node", seen) is True
    assert len(seen) == 2


# --- _INSTALL_PATTERN_QUERIES -------------------------------------------------


def test_install_pattern_queries_cover_new_ecosystems() -> None:
    """v3 must query for rust/ruby/go bases to expand FROM diversity.

    Dotnet was originally on this list but removed in fix/v3-cubic-review-followups:
    without a `dotnet` ecosystem entry in _INSTALLER_SIGNALS and a `_dotnet_probe` in
    annotate.py, the query yields zero usable rows (cubic finding on PR #11). Restore
    the assertion when dotnet support lands in v4.
    """
    from dockermin.dataset.scrape import _INSTALL_PATTERN_QUERIES

    queries = set(_INSTALL_PATTERN_QUERIES)
    must_have = {
        "cargo language:Dockerfile filename:Dockerfile size:<2500",
        '"bundle install" language:Dockerfile filename:Dockerfile size:<2500',
        '"go mod" language:Dockerfile filename:Dockerfile size:<2500',
    }
    missing = must_have - queries
    assert not missing, f"v3 must query: {sorted(missing)}"


# --- fetch_chainguard_images --------------------------------------------------


def test_fetch_chainguard_images_is_callable() -> None:
    """Smoke: chainguard fetcher exists and accepts a limit kwarg."""
    import inspect

    from dockermin.dataset.scrape import fetch_chainguard_images

    sig = inspect.signature(fetch_chainguard_images)
    assert "limit" in sig.parameters
