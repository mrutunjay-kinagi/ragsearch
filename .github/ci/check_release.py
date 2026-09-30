"""
Release gate for .github/workflows/python-publish.yml (#107).

Checks, before anything is built or published:
- the release tag is "v" + the version in pyproject.toml;
- the version is a final release (X.Y.Z) or a release candidate (X.Y.Z{a,b,rc}N);
- the GitHub pre-release flag matches: candidates must be pre-releases, finals must not.

Writes "target=testpypi" (candidates) or "target=pypi" (finals) to $GITHUB_OUTPUT.
Standard library only (tomllib is not available on Python 3.10).

Usage:
  check_release.py --tag v0.2.0 --prerelease false [--pyproject pyproject.toml]
  check_release.py --print-version [--pyproject pyproject.toml]
"""

import argparse
import os
import re
import sys
from pathlib import Path

FINAL = re.compile(r"^\d+\.\d+\.\d+$")
CANDIDATE = re.compile(r"^\d+\.\d+\.\d+(a|b|rc)\d+$")


def read_version(pyproject: Path) -> str:
    section = pyproject.read_text(encoding="utf-8").split("[tool.poetry]", 1)
    match = re.search(r'^version\s*=\s*"([^"]+)"', section[1] if len(section) == 2 else "", re.MULTILINE)
    if not match:
        raise SystemExit(f"error: no [tool.poetry] version found in {pyproject}")
    return match.group(1)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag")
    parser.add_argument("--prerelease", choices=["true", "false"])
    parser.add_argument("--pyproject", default="pyproject.toml", type=Path)
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()

    version = read_version(args.pyproject)
    if args.print_version:
        print(version)
        return
    if args.tag is None or args.prerelease is None:
        parser.error("--tag and --prerelease are required")

    if not args.tag.startswith("v"):
        fail(f"release tag {args.tag!r} must start with 'v' (expected 'v{version}')")
    if args.tag[1:] != version:
        fail(f"release tag {args.tag!r} does not match the pyproject.toml version {version!r} (expected 'v{version}')")

    is_candidate = bool(CANDIDATE.match(version))
    if not is_candidate and not FINAL.match(version):
        fail(f"{version!r} is not a supported release version: use X.Y.Z, or X.Y.Z(a|b|rc)N for candidates")
    prerelease = args.prerelease == "true"
    if is_candidate and not prerelease:
        fail(f"{version!r} is a release candidate and must be published as a GitHub pre-release (it goes to TestPyPI)")
    if not is_candidate and prerelease:
        fail(f"{version!r} is a final version but the GitHub release is marked as a pre-release")

    target = "testpypi" if is_candidate else "pypi"
    print(f"release {args.tag}: version {version} -> {target}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"target={target}\n")


if __name__ == "__main__":
    main()
