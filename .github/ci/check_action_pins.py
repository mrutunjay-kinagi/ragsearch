"""
Check that GitHub Actions pinned by SHA in .github/workflows/ are pinned to *commits* (0.2.0rc2).

0.2.0rc1 never reached TestPyPI because pypa/gh-action-pypi-publish was pinned to the SHA of the
annotated tag object v1.14.2 (a892a5a...) instead of the commit it points to (dc37677...). GitHub
still ran the action's code, but the action pulls its Docker image as
ghcr.io/pypa/gh-action-pypi-publish:<pinned ref>, and images exist only for commits and version
tags, so the upload failed with "manifest unknown".

For every `uses: owner/repo@<40-hex sha>` this script checks, via the GitHub API, that the SHA is a
commit (and, if it is an annotated tag object, prints the commit to pin instead). For
pypa/gh-action-pypi-publish it also checks that the Docker image for that SHA exists on ghcr.io.
No provider APIs are called. Set GITHUB_TOKEN to avoid API rate limits.

Usage: python .github/ci/check_action_pins.py [workflow files...]
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PIN = re.compile(r"uses:\s*([\w.-]+)/([\w.-]+)(?:/[\w./-]+)?@([0-9a-f]{40})\b")
DOCKER_IMAGE_ACTIONS = {("pypa", "gh-action-pypi-publish"): "ghcr.io/pypa/gh-action-pypi-publish"}


def _get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _github(path):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ragsearch-ci"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return _get(f"https://api.github.com{path}", headers)


def _ghcr_manifest_exists(image, tag):
    repository = image.split("/", 1)[1]
    status, token = _get(f"https://ghcr.io/token?scope=repository:{repository}:pull")
    if status != 200:
        return False
    accept = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/manifests/{tag}",
        headers={"Authorization": f"Bearer {token['token']}", "Accept": accept},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False


def check(files):
    problems = []
    checked = 0
    for path in files:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            match = PIN.search(line)
            if not match:
                continue
            owner, repo, sha = match.groups()
            where = f"{path}:{line_number} {owner}/{repo}@{sha}"
            checked += 1
            status, _ = _github(f"/repos/{owner}/{repo}/git/commits/{sha}")
            if status != 200:
                tag_status, tag = _github(f"/repos/{owner}/{repo}/git/tags/{sha}")
                if tag_status == 200 and tag and tag.get("object", {}).get("type") == "commit":
                    problems.append(f"{where}: this is the annotated tag object {tag.get('tag')!r}, not a commit; pin {tag['object']['sha']}")
                else:
                    problems.append(f"{where}: not a commit in {owner}/{repo} (HTTP {status})")
                continue
            image = DOCKER_IMAGE_ACTIONS.get((owner, repo))
            if image and not _ghcr_manifest_exists(image, sha):
                problems.append(f"{where}: no Docker image {image}:{sha}")
                continue
            print(f"ok: {where}")
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    print(f"{checked} SHA-pinned action(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(str(p) for p in Path(".github/workflows").glob("*.yml"))
    sys.exit(check(targets))
