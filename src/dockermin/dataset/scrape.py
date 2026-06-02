"""Scrape candidate Dockerfiles from official-images, awesome-compose, and GitHub code search."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Small pause between gh calls to keep us under secondary rate limits.
GH_SLEEP_S = 0.5
RAW_SLEEP_S = 0.1

# GitHub search returns at most this many items per page.
_SEARCH_PAGE_SIZE = 100
# A FROM line needs at least image keyword plus a base ref to be usable.
_MIN_FROM_PARTS = 2

# Exceptions raised by transient fetch failures we skip the candidate on.
_FETCH_ERRORS = (
    urllib.error.URLError,
    subprocess.SubprocessError,
    OSError,
    json.JSONDecodeError,
    TimeoutError,
)


@dataclass(frozen=True)
class Candidate:
    source: str
    url: str
    dockerfile: str
    ecosystem: str
    license: str | None


# Mapping of common official-images / base-image names to ecosystems.
_ECOSYSTEM_MAP: dict[str, str] = {
    "python": "python",
    "pypy": "python",
    "node": "node",
    "iojs": "node",
    "golang": "go",
    "rust": "rust",
    "ruby": "ruby",
    "openjdk": "java",
    "eclipse-temurin": "java",
    "amazoncorretto": "java",
    "ibm-semeru-runtimes": "java",
    "php": "php",
    "perl": "perl",
    "elixir": "elixir",
    "erlang": "erlang",
    "haskell": "haskell",
    "clojure": "clojure",
    "swift": "swift",
    "alpine": "alpine",
    "debian": "debian",
    "ubuntu": "ubuntu",
    "fedora": "fedora",
    "centos": "centos",
    "redis": "redis",
    "mongo": "mongo",
    "postgres": "postgres",
    "mysql": "mysql",
    "mariadb": "mysql",
    "nginx": "nginx",
    "httpd": "httpd",
    "traefik": "traefik",
    "caddy": "caddy",
    "memcached": "memcached",
    "rabbitmq": "rabbitmq",
    "elasticsearch": "elasticsearch",
    "kibana": "elasticsearch",
    "logstash": "elasticsearch",
}


# RUN-line installer signals -> ecosystem. Mirrors the breadth the annotate
# probes recognise, so the scraper does not reject go/ruby/php/node candidates
# that sit on a generic (debian/ubuntu/alpine) base just because the FROM line
# is not itself a runtime image.
_INSTALLER_SIGNALS: tuple[tuple[str, str], ...] = (
    ("pip install", "python"),
    ("pip3 install", "python"),
    ("poetry install", "python"),
    ("npm install", "node"),
    ("npm ci", "node"),
    ("yarn install", "node"),
    ("gem install", "ruby"),
    ("bundle install", "ruby"),
    ("composer install", "php"),
    ("composer require", "php"),
    ("go build", "go"),
    ("go install", "go"),
    ("cargo build", "rust"),
    ("cargo install", "rust"),
)

# Bases that are runtime images: their FROM line alone pins the ecosystem.
# A generic OS base (debian/ubuntu/alpine/...) is NOT here, so we fall through
# to the RUN-line installer signals for those.
_RUNTIME_ECOSYSTEMS: frozenset[str] = frozenset(
    {
        "python",
        "node",
        "go",
        "rust",
        "ruby",
        "java",
        "php",
        "perl",
        "elixir",
        "erlang",
        "haskell",
        "clojure",
        "swift",
    }
)


def _infer_ecosystem(name: str) -> str:
    """Map an image / repo name to a coarse ecosystem label, defaulting to 'unknown'."""
    if not name:
        return "unknown"
    key = name.lower().strip()
    if key in _ECOSYSTEM_MAP:
        return _ECOSYSTEM_MAP[key]
    head = key.split("-", 1)[0]
    if head in _ECOSYSTEM_MAP:
        return _ECOSYSTEM_MAP[head]
    return "unknown"


def _gh_api(path: str, extra_args: list[str] | None = None) -> Any:  # noqa: ANN401
    """Run `gh api <path>` and return parsed JSON. Sleep briefly between calls."""
    cmd = ["gh", "api", path]
    if extra_args:
        cmd.extend(extra_args)
    # gh is a trusted local CLI invoked with our own constructed arguments.
    out = subprocess.check_output(cmd, text=True)  # noqa: S603
    time.sleep(GH_SLEEP_S)
    return json.loads(out)


def _fetch_raw(url: str) -> str:
    """Fetch a raw text URL and return the body as a string."""
    # URLs come from the GitHub API (https raw/download links), not arbitrary schemes.
    req = urllib.request.Request(url, headers={"User-Agent": "dockermin-scrape/0.1"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw: bytes = resp.read()
    body = raw.decode("utf-8", errors="replace")
    time.sleep(RAW_SLEEP_S)
    return body


def _split_manifest_blocks(text: str) -> list[str]:
    """Split manifest text into blank-line-separated blocks of non-empty lines.

    A line that is empty after stripping acts as a stanza separator.
    """
    normalized = "\n".join("" if not line.strip() else line for line in text.splitlines())
    blocks = normalized.split("\n\n")
    return [block for block in blocks if block.strip()]


def _parse_stanza_block(block: str) -> dict[str, str]:
    """Parse one manifest block into a `Key: value` mapping, ignoring comments."""
    stanza: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        stanza[key.strip()] = value.strip()
    return stanza


def _parse_manifest_stanzas(text: str) -> list[dict[str, str]]:
    """Split a manifest into blank-line-separated `Key: value` stanzas."""
    stanzas = [_parse_stanza_block(block) for block in _split_manifest_blocks(text)]
    return [stanza for stanza in stanzas if stanza]


def _merge_stanza(stanza: dict[str, str], defaults: dict[str, str]) -> dict[str, str]:
    """Merge a tag stanza with the manifest defaults."""
    merged: dict[str, str] = {}
    if "Tags" in defaults:
        merged.update(defaults)
    else:
        for k, v in defaults.items():
            merged.setdefault(k, v)
    merged.update(stanza)
    return merged


def _parse_official_manifest(text: str) -> list[dict[str, str]]:
    """Parse a docker-library/official-images library/<name> manifest into per-tag stanzas.

    Manifest format: blank-line-separated stanzas of `Key: value` lines. The first stanza
    holds defaults; later stanzas inherit them and may override. We only need GitRepo,
    GitCommit, Directory, and File.
    """
    stanzas = _parse_manifest_stanzas(text)
    if not stanzas:
        return []

    defaults = stanzas[0]
    tag_stanzas = stanzas[1:] if "Tags" not in defaults else stanzas

    merged: list[dict[str, str]] = []
    for stanza in tag_stanzas:
        m = _merge_stanza(stanza, defaults)
        if not m.get("GitRepo") or not m.get("GitCommit"):
            continue
        merged.append(m)
    return merged


def _strip_github_repo(git_repo: str) -> str:
    """Reduce a GitRepo URL to its `owner/repo` path."""
    repo_path = git_repo
    if repo_path.startswith("https://github.com/"):
        repo_path = repo_path[len("https://github.com/") :]
    if repo_path.endswith(".git"):
        repo_path = repo_path[: -len(".git")]
    return repo_path


def _fetch_dockerfile(url: str, context: str) -> str | None:
    """Fetch a Dockerfile body, returning None on fetch failure or empty content."""
    try:
        text = _fetch_raw(url)
    except _FETCH_ERRORS as exc:
        logger.debug("Dockerfile fetch failed (%s): %s", context, exc)
        return None
    return text if text.strip() else None


def _manifest_raw_url(manifest_text: str) -> str | None:
    """Resolve the raw.githubusercontent URL of a manifest's first usable stanza."""
    stanzas = _parse_official_manifest(manifest_text)
    if not stanzas:
        return None
    chosen = stanzas[0]
    git_repo = chosen.get("GitRepo", "").rstrip("/")
    git_commit = chosen.get("GitCommit", "")
    if not git_repo or not git_commit:
        return None
    directory = chosen.get("Directory", "").strip("/")
    dockerfile_name = chosen.get("File", "Dockerfile").strip() or "Dockerfile"
    repo_path = _strip_github_repo(git_repo)
    path_parts = [p for p in [directory, dockerfile_name] if p]
    return f"https://raw.githubusercontent.com/{repo_path}/{git_commit}/{'/'.join(path_parts)}"


def _manifest_dockerfile_url(name: str, manifest_url: str) -> str | None:
    """Fetch an official-images manifest and resolve its first Dockerfile's raw URL."""
    manifest_text = _fetch_dockerfile(manifest_url, f"official-images manifest {name}")
    if manifest_text is None:
        return None
    return _manifest_raw_url(manifest_text)


def _candidate_from_manifest_entry(entry: dict[str, Any]) -> Candidate | None:
    """Build a Candidate from one official-images library/<name> manifest entry."""
    name = entry.get("name")
    download_url = entry.get("download_url")
    if not name or not download_url:
        return None
    raw_url = _manifest_dockerfile_url(name, download_url)
    if raw_url is None:
        return None
    dockerfile_text = _fetch_dockerfile(raw_url, f"official-images Dockerfile {name}")
    if dockerfile_text is None:
        return None
    return Candidate(
        source=f"official-images:{name}",
        url=raw_url,
        dockerfile=dockerfile_text,
        ecosystem=_infer_ecosystem(name),
        license="best-effort",
    )


def fetch_official_images(limit: int = 100) -> Iterable[Candidate]:
    """Walk docker-library/official-images library/ and yield one Candidate per image name."""
    listing = _gh_api("repos/docker-library/official-images/contents/library")
    if not isinstance(listing, list):
        return

    yielded = 0
    for entry in listing:
        if yielded >= limit:
            return
        candidate = _candidate_from_manifest_entry(entry)
        if candidate is None:
            continue
        yield candidate
        yielded += 1


def _candidate_from_compose_file(child: dict[str, Any], source_prefix: str) -> Candidate | None:
    """Build a Candidate from an awesome-compose Dockerfile file entry."""
    name = child.get("name", "")
    if name != "Dockerfile" and not name.endswith(".Dockerfile"):
        return None
    download_url = child.get("download_url")
    if not download_url:
        return None
    dockerfile_text = _fetch_dockerfile(download_url, f"awesome-compose {download_url}")
    if dockerfile_text is None:
        return None
    return Candidate(
        source=source_prefix,
        url=download_url,
        dockerfile=dockerfile_text,
        ecosystem=_infer_ecosystem(source_prefix.split(":", 1)[-1]),
        license="Apache-2.0",
    )


def _list_awesome_compose_dir(path: str) -> list[dict[str, Any]] | None:
    """List the contents of an awesome-compose subdir, or None on failure."""
    try:
        children = _gh_api(f"repos/docker/awesome-compose/contents/{path}")
    except subprocess.CalledProcessError as exc:
        logger.debug("awesome-compose dir listing failed for %s: %s", path, exc)
        return None
    return children if isinstance(children, list) else None


def _walk_compose_child(child: dict[str, Any], source_prefix: str, max_depth: int) -> Iterable[Candidate]:
    """Yield Candidates for one directory-tree entry (a file or a subdir to recurse into)."""
    ctype = child.get("type")
    if ctype == "file":
        candidate = _candidate_from_compose_file(child, source_prefix)
        if candidate is not None:
            yield candidate
        return
    sub_path = child.get("path")
    if ctype != "dir" or max_depth <= 0 or not sub_path:
        return
    sub_children = _list_awesome_compose_dir(sub_path)
    if sub_children is not None:
        yield from _walk_awesome_compose_dir(sub_children, source_prefix, max_depth=max_depth - 1)


def _walk_awesome_compose_dir(
    children: list[dict[str, Any]], source_prefix: str, max_depth: int = 2
) -> Iterable[Candidate]:
    """Yield Candidates for any Dockerfile entries in a directory tree.

    awesome-compose stores Dockerfiles in nested subdirs (e.g. angular/angular/Dockerfile).
    Walk up to max_depth levels deep.
    """
    for child in children:
        yield from _walk_compose_child(child, source_prefix, max_depth)


def _compose_candidates_for_entry(entry: dict[str, Any]) -> Iterable[Candidate]:
    """Yield every Dockerfile Candidate under one top-level awesome-compose dir entry."""
    if entry.get("type") != "dir":
        return
    dir_name = entry.get("name")
    if not dir_name:
        return
    children = _list_awesome_compose_dir(dir_name)
    if children is None:
        return
    yield from _walk_awesome_compose_dir(children, source_prefix=f"awesome-compose:{dir_name}")


def fetch_awesome_compose(limit: int = 50) -> Iterable[Candidate]:
    """List subdirs of docker/awesome-compose and yield one Candidate per Dockerfile found."""
    top = _gh_api("repos/docker/awesome-compose/contents")
    if not isinstance(top, list):
        return

    def _all() -> Iterable[Candidate]:
        for entry in top:
            yield from _compose_candidates_for_entry(entry)

    yield from itertools.islice(_all(), limit)


_CHAINGUARD_REPO = "chainguard-images/images"


def _list_github_dir(repo: str, path: str) -> list[dict[str, Any]] | None:
    """List the contents of ``repos/{repo}/contents/{path}`` or None on failure."""
    try:
        children = _gh_api(f"repos/{repo}/contents/{path}")
    except subprocess.CalledProcessError as exc:
        logger.debug("github dir listing failed for %s/%s: %s", repo, path, exc)
        return None
    return children if isinstance(children, list) else None


def _walk_chainguard_child(repo: str, child: dict[str, Any], max_depth: int) -> Iterable[str]:
    """Yield file paths for one chainguard tree entry (file or subdir to recurse)."""
    ctype = child.get("type")
    path = child.get("path")
    if not path:
        return
    if ctype == "file":
        yield path
        return
    if ctype != "dir" or max_depth <= 0:
        return
    # Recurse directly; _walk_chainguard_dir does its own _list_github_dir call.
    # Earlier code fetched the dir listing here too (cubic flagged on PR #11) —
    # one fetch per directory is enough.
    yield from _walk_chainguard_dir(repo, path, max_depth=max_depth - 1)


def _walk_chainguard_dir(repo: str, root: str, max_depth: int) -> Iterable[str]:
    """Yield file paths from ``repo`` under ``root`` via the GitHub contents API.

    Mirrors the awesome-compose walker but yields raw paths so the caller can
    filter by filename (Dockerfile / *.Dockerfile) before fetching content.
    """
    children = _list_github_dir(repo, root)
    if children is None:
        return
    for child in children:
        yield from _walk_chainguard_child(repo, child, max_depth)


def _github_file_download_url(repo: str, path: str) -> str | None:
    """Resolve the raw download URL for a file path inside ``repo``."""
    try:
        meta = _gh_api(f"repos/{repo}/contents/{path}")
    except subprocess.CalledProcessError as exc:
        logger.debug("github file meta failed for %s/%s: %s", repo, path, exc)
        return None
    if not isinstance(meta, dict):
        return None
    return meta.get("download_url") or None


def _candidate_from_github_path(repo: str, path: str, license_name: str | None) -> Candidate | None:
    """Resolve a file path inside ``repo`` to a Candidate via the contents API."""
    download_url = _github_file_download_url(repo, path)
    if not download_url:
        return None
    dockerfile_text = _fetch_dockerfile(download_url, f"{repo}/{path}")
    if dockerfile_text is None:
        return None
    return Candidate(
        source=f"{repo}:{path}",
        url=download_url,
        dockerfile=dockerfile_text,
        ecosystem=_ecosystem_from_dockerfile(dockerfile_text),
        license=license_name,
    )


def _is_dockerfile_path(path: str) -> bool:
    """Match awesome-compose's `Dockerfile` and `*.Dockerfile` filename convention."""
    return path.endswith(("/Dockerfile", ".Dockerfile"))


def _chainguard_candidate(path: str, seen_hashes: set[str]) -> Candidate | None:
    """Probe one chainguard file path and return a Candidate if it survives the filters."""
    candidate = _candidate_from_github_path(_CHAINGUARD_REPO, path, license_name="Apache-2.0")
    if candidate is None:
        return None
    if not _is_self_contained_probeable(candidate.dockerfile):
        return None
    if not _is_new_content(candidate.dockerfile, seen_hashes):
        return None
    return candidate


def fetch_chainguard_images(limit: int = 50) -> Iterable[Candidate]:
    """Walk chainguard-images/images on GitHub for Wolfi-derived Dockerfiles.

    Chainguard images use ``apk`` (Wolfi/Alpine) instead of apt — entirely
    unrepresented in dockermin-v0. Yields candidates the same shape as
    ``fetch_awesome_compose``. License: Apache-2.0 per the repo.
    """
    seen_hashes: set[str] = set()
    yielded = 0
    for path in _walk_chainguard_dir(_CHAINGUARD_REPO, "images", max_depth=2):
        if yielded >= limit:
            return
        if not _is_dockerfile_path(path):
            continue
        candidate = _chainguard_candidate(path, seen_hashes)
        if candidate is None:
            continue
        yield candidate
        yielded += 1


def _search_code_page(query: str, page: int) -> list[dict[str, Any]]:
    """Fetch one page of GitHub code-search results, returning [] on failure or end."""
    cmd = [
        "gh",
        "api",
        "-X",
        "GET",
        "search/code",
        "-f",
        f"q={query}",
        "-F",
        f"per_page={_SEARCH_PAGE_SIZE}",
        "-F",
        f"page={page}",
    ]
    try:
        # gh is a trusted local CLI invoked with our own constructed arguments.
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        logger.warning("gh api search/code failed (rc=%s, query=%r, page=%s): %s", exc.returncode, query, page, stderr)
        return []
    time.sleep(GH_SLEEP_S)
    payload = json.loads(out)
    return payload.get("items", []) if isinstance(payload, dict) else []


def _search_code_query(query: str) -> list[dict[str, Any]]:
    """Page through one query variant up to GitHub's 10-page limit."""
    items: list[dict[str, Any]] = []
    for page in range(1, 11):  # GitHub search caps at 10 pages
        page_items = _search_code_page(query, page)
        items.extend(page_items)
        if len(page_items) < _SEARCH_PAGE_SIZE:
            break
    return items


def _collect_search_items(query_variants: list[str], target: int) -> list[dict[str, Any]]:
    """Page through each query variant until we have enough items or exhaust pages."""
    items: list[dict[str, Any]] = []
    for query in query_variants:
        if len(items) >= target:
            break
        items.extend(_search_code_query(query))
    return items


def _ecosystem_from_from_line(dockerfile_text: str) -> str:
    """Infer the ecosystem from the first FROM line of a Dockerfile."""
    for line in dockerfile_text.splitlines():
        ls = line.strip().lower()
        if ls.startswith("from "):
            parts = ls.split()
            if len(parts) >= _MIN_FROM_PARTS:
                base = parts[1]
                base_name = base.split(":", 1)[0].split("/")[-1]
                return _infer_ecosystem(base_name)
            return "unknown"
    return "unknown"


def _ecosystem_from_run_lines(dockerfile_text: str) -> str:
    """Infer the ecosystem from RUN-line installer signals (pip/npm/gem/...)."""
    text = dockerfile_text.lower()
    for signal, ecosystem in _INSTALLER_SIGNALS:
        if signal in text:
            return ecosystem
    return "unknown"


def _ecosystem_from_dockerfile(dockerfile_text: str) -> str:
    """Resolve the ecosystem from the FROM runtime base, else RUN installer lines.

    A known runtime base (python/node/golang/...) pins the ecosystem directly.
    For a generic OS base (debian/ubuntu/alpine) or an unknown base, we fall
    through to the RUN-line installer signals so a `debian + pip install`
    candidate is recognised as python rather than rejected as unknown.
    """
    from_ecosystem = _ecosystem_from_from_line(dockerfile_text)
    if from_ecosystem in _RUNTIME_ECOSYSTEMS:
        return from_ecosystem
    return _ecosystem_from_run_lines(dockerfile_text)


def _instruction_keyword(line: str) -> str:
    """Return the upper-cased Dockerfile instruction keyword of a line, or ''."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return stripped.split(maxsplit=1)[0].upper()


def _add_source_is_remote(line: str) -> bool:
    """Return True if every source of an ADD instruction is a remote URL.

    A remote ADD fetches over the network and needs no local build context;
    a local ADD (e.g. `ADD app.tar /`) does and so is not self-contained.
    """
    tokens = line.strip().split()[1:]  # drop the ADD keyword
    sources = [t for t in tokens if not t.startswith("--")][:-1]  # last token is the dest
    if not sources:
        return False
    return all(src.startswith(("http://", "https://")) for src in sources)


def _is_self_contained_probeable(dockerfile: str) -> bool:
    """Return True if a Dockerfile can be built and probed with no local context.

    Rejects bodies that need files from the build context (a `COPY` instruction
    or a local, non-URL `ADD`) and bodies whose ecosystem does not resolve to a
    known runtime - those are the candidates the empty-context build gate and the
    runtime probe would drop anyway.
    """
    for line in dockerfile.splitlines():
        keyword = _instruction_keyword(line)
        if keyword == "COPY":
            return False
        if keyword == "ADD" and not _add_source_is_remote(line):
            return False
    return _ecosystem_from_dockerfile(dockerfile) != "unknown"


def _code_hit_download_url(full_name: str, path: str) -> str | None:
    """Resolve the raw download URL for a code-search hit's file path."""
    try:
        file_meta = _gh_api(f"repos/{full_name}/contents/{path}")
    except subprocess.CalledProcessError as exc:
        logger.debug("code-search file meta failed for %s/%s: %s", full_name, path, exc)
        return None
    if not isinstance(file_meta, dict):
        return None
    return file_meta.get("download_url") or None


def _is_new_content(dockerfile_text: str, seen_hashes: set[str]) -> bool:
    """Record a content hash; return False if it was already seen."""
    digest = hashlib.sha256(dockerfile_text.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)
    return True


def _candidate_from_code_hit(item: dict[str, Any], seen_hashes: set[str]) -> Candidate | None:
    """Build a Candidate from one code-search hit, deduping by content hash."""
    repo = item.get("repository", {}) or {}
    full_name = repo.get("full_name") or ""
    path = item.get("path") or ""
    if not full_name or not path:
        return None
    download_url = _code_hit_download_url(full_name, path)
    if not download_url:
        return None
    dockerfile_text = _fetch_dockerfile(download_url, f"code-search {full_name}/{path}")
    if dockerfile_text is None or not _is_new_content(dockerfile_text, seen_hashes):
        return None
    return Candidate(
        source=f"gh:{full_name}:{path}",
        url=item.get("html_url") or download_url,
        dockerfile=dockerfile_text,
        ecosystem=_ecosystem_from_dockerfile(dockerfile_text),
        license=None,
    )


# Per-ecosystem install-pattern queries. GitHub code-search legacy syntax has
# no NOT/OR, so we use positive installer terms server-side and reject COPY /
# unknown-ecosystem hits client-side via _is_self_contained_probeable. Targeting
# the install pattern (not raw file size) is what surfaces self-contained,
# probeable Dockerfiles instead of repo-coupled ones.
_INSTALL_PATTERN_QUERIES: tuple[str, ...] = (
    "pip language:Dockerfile filename:Dockerfile size:<2500",
    "npm language:Dockerfile filename:Dockerfile size:<2500",
    "gem language:Dockerfile filename:Dockerfile size:<2500",
    "composer language:Dockerfile filename:Dockerfile size:<2500",
    '"go build" language:Dockerfile filename:Dockerfile size:<2500',
    # v3 expansion — broaden FROM-base diversity per the failure analysis
    # in docs/decisions/2026-06-02-v2-headline-result.md.
    "cargo language:Dockerfile filename:Dockerfile size:<2500",
    '"bundle install" language:Dockerfile filename:Dockerfile size:<2500',
    '"go mod" language:Dockerfile filename:Dockerfile size:<2500',
    # `"dotnet restore"` deferred to v4: needs an annotate probe (_dotnet_probe
    # in annotate.py's _PROBE_BUILDERS) before scraped candidates can survive
    # annotation. Without it, hits would resolve to ecosystem='unknown' and be
    # filtered out (cubic flagged on PR #11).
)


def fetch_github_search(limit: int = 100) -> Iterable[Candidate]:
    """Search GitHub code for self-contained, probeable Dockerfiles by install pattern.

    Queries target per-ecosystem installer terms (pip/npm/gem/composer/go build)
    rather than raw file size, then every fetched candidate is gated through
    `_is_self_contained_probeable` - the client-side filter rejects COPY/local-ADD
    and unknown-ecosystem hits that the server-side legacy syntax cannot exclude.

    Pages through search/code 100 items at a time (GitHub caps total search
    results at 1000 across all pages). Dedupe by content hash.
    """
    seen_hashes: set[str] = set()

    items = _collect_search_items(list(_INSTALL_PATTERN_QUERIES), target=limit * 3)

    yielded = 0
    for item in items:
        if yielded >= limit:
            return
        candidate = _candidate_from_code_hit(item, seen_hashes)
        if candidate is None:
            continue
        if not _is_self_contained_probeable(candidate.dockerfile):
            continue
        yield candidate
        yielded += 1
