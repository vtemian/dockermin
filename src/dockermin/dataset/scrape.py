"""Scrape candidate Dockerfiles from official-images, awesome-compose, and GitHub code search."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable

# Small pause between gh calls to keep us under secondary rate limits.
GH_SLEEP_S = 0.5
RAW_SLEEP_S = 0.1


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


def _gh_api(path: str, extra_args: list[str] | None = None) -> object:
    """Run `gh api <path>` and return parsed JSON. Sleep briefly between calls."""
    cmd = ["gh", "api", path]
    if extra_args:
        cmd.extend(extra_args)
    out = subprocess.check_output(cmd, text=True)
    time.sleep(GH_SLEEP_S)
    return json.loads(out)


def _fetch_raw(url: str) -> str:
    """Fetch a raw text URL and return the body as a string."""
    req = urllib.request.Request(url, headers={"User-Agent": "dockermin-scrape/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    time.sleep(RAW_SLEEP_S)
    return body


def _parse_official_manifest(text: str) -> list[dict]:
    """Parse a docker-library/official-images library/<name> manifest into per-tag stanzas.

    Manifest format: blank-line-separated stanzas of `Key: value` lines. The first stanza
    holds defaults; later stanzas inherit them and may override. We only need GitRepo,
    GitCommit, Directory, and File.
    """
    stanzas: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current:
                stanzas.append(current)
                current = {}
            continue
        if line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip()
    if current:
        stanzas.append(current)
    if not stanzas:
        return []

    defaults = stanzas[0]
    tag_stanzas = stanzas[1:] if "Tags" not in defaults else stanzas

    merged: list[dict] = []
    for s in tag_stanzas:
        m: dict[str, str] = {}
        if "Tags" in defaults:
            m.update(defaults)
        else:
            for k, v in defaults.items():
                m.setdefault(k, v)
        m.update(s)
        if not m.get("GitRepo") or not m.get("GitCommit"):
            continue
        merged.append(m)
    return merged


def fetch_official_images(limit: int = 100) -> Iterable[Candidate]:
    """Walk docker-library/official-images library/ and yield one Candidate per image name."""
    listing = _gh_api("repos/docker-library/official-images/contents/library")
    if not isinstance(listing, list):
        return

    yielded = 0
    for entry in listing:
        if yielded >= limit:
            return
        name = entry.get("name")
        download_url = entry.get("download_url")
        if not name or not download_url:
            continue
        try:
            manifest_text = _fetch_raw(download_url)
        except Exception:
            continue
        stanzas = _parse_official_manifest(manifest_text)
        if not stanzas:
            continue

        chosen = stanzas[0]
        git_repo = chosen.get("GitRepo", "").rstrip("/")
        git_commit = chosen.get("GitCommit", "")
        directory = chosen.get("Directory", "").strip("/")
        dockerfile_name = chosen.get("File", "Dockerfile").strip() or "Dockerfile"
        if not git_repo or not git_commit:
            continue

        repo_path = git_repo
        if repo_path.startswith("https://github.com/"):
            repo_path = repo_path[len("https://github.com/"):]
        if repo_path.endswith(".git"):
            repo_path = repo_path[: -len(".git")]

        path_parts = [p for p in [directory, dockerfile_name] if p]
        raw_url = f"https://raw.githubusercontent.com/{repo_path}/{git_commit}/{'/'.join(path_parts)}"
        try:
            dockerfile_text = _fetch_raw(raw_url)
        except Exception:
            continue
        if not dockerfile_text.strip():
            continue

        yield Candidate(
            source=f"official-images:{name}",
            url=raw_url,
            dockerfile=dockerfile_text,
            ecosystem=_infer_ecosystem(name),
            license="best-effort",
        )
        yielded += 1


def _walk_awesome_compose_dir(children: list[dict], source_prefix: str,
                              max_depth: int = 2) -> Iterable[Candidate]:
    """Yield Candidates for any Dockerfile entries in a directory tree.

    awesome-compose stores Dockerfiles in nested subdirs (e.g. angular/angular/Dockerfile).
    Walk up to max_depth levels deep.
    """
    for child in children:
        ctype = child.get("type")
        if ctype == "file":
            name = child.get("name", "")
            if name != "Dockerfile" and not name.endswith(".Dockerfile"):
                continue
            download_url = child.get("download_url")
            if not download_url:
                continue
            try:
                dockerfile_text = _fetch_raw(download_url)
            except Exception:
                continue
            if not dockerfile_text.strip():
                continue
            yield Candidate(
                source=source_prefix,
                url=download_url,
                dockerfile=dockerfile_text,
                ecosystem=_infer_ecosystem(source_prefix.split(":", 1)[-1]),
                license="Apache-2.0",
            )
        elif ctype == "dir" and max_depth > 0:
            sub_path = child.get("path")
            if not sub_path:
                continue
            try:
                sub_children = _gh_api(f"repos/docker/awesome-compose/contents/{sub_path}")
            except subprocess.CalledProcessError:
                continue
            if not isinstance(sub_children, list):
                continue
            yield from _walk_awesome_compose_dir(sub_children, source_prefix,
                                                 max_depth=max_depth - 1)


def fetch_awesome_compose(limit: int = 50) -> Iterable[Candidate]:
    """List subdirs of docker/awesome-compose and yield one Candidate per Dockerfile found."""
    top = _gh_api("repos/docker/awesome-compose/contents")
    if not isinstance(top, list):
        return

    yielded = 0
    for entry in top:
        if yielded >= limit:
            return
        if entry.get("type") != "dir":
            continue
        dir_name = entry.get("name")
        if not dir_name:
            continue
        try:
            children = _gh_api(f"repos/docker/awesome-compose/contents/{dir_name}")
        except subprocess.CalledProcessError:
            continue
        if not isinstance(children, list):
            continue
        for cand in _walk_awesome_compose_dir(children, source_prefix=f"awesome-compose:{dir_name}"):
            yield cand
            yielded += 1
            if yielded >= limit:
                return


def fetch_github_search(limit: int = 100) -> Iterable[Candidate]:
    """Search GitHub code for small Dockerfiles. Dedupe by content hash.

    Note: code search does NOT accept the `stars:` qualifier (that is for repo
    search). Using language:Dockerfile + size:<5000 for volume; quality
    filtering happens at annotation time via parse+build+test gates.
    """
    cmd = [
        "gh", "api", "-X", "GET", "search/code",
        "-f", "q=filename:Dockerfile language:Dockerfile size:<5000",
        "-F", "per_page=100",
    ]
    out = subprocess.check_output(cmd, text=True)
    time.sleep(GH_SLEEP_S)
    payload = json.loads(out)
    items = payload.get("items", []) if isinstance(payload, dict) else []

    seen_hashes: set[str] = set()
    yielded = 0
    for item in items:
        if yielded >= limit:
            return
        repo = item.get("repository", {}) or {}
        full_name = repo.get("full_name") or ""
        path = item.get("path") or ""
        html_url = item.get("html_url") or ""
        if not full_name or not path:
            continue
        try:
            file_meta = _gh_api(f"repos/{full_name}/contents/{path}")
        except subprocess.CalledProcessError:
            continue
        if not isinstance(file_meta, dict):
            continue
        download_url = file_meta.get("download_url")
        if not download_url:
            continue
        try:
            dockerfile_text = _fetch_raw(download_url)
        except Exception:
            continue
        if not dockerfile_text.strip():
            continue
        digest = hashlib.sha256(dockerfile_text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        ecosystem = "unknown"
        for line in dockerfile_text.splitlines():
            ls = line.strip().lower()
            if ls.startswith("from "):
                parts = ls.split()
                if len(parts) >= 2:
                    base = parts[1]
                    base_name = base.split(":", 1)[0].split("/")[-1]
                    ecosystem = _infer_ecosystem(base_name)
                break

        yield Candidate(
            source=f"gh:{full_name}:{path}",
            url=html_url or download_url,
            dockerfile=dockerfile_text,
            ecosystem=ecosystem,
            license=None,
        )
        yielded += 1
