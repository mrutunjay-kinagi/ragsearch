"""
Packaging metadata checks (#105: Python 3.10+).

These read pyproject.toml and poetry.lock as text (tomllib is not available on Python 3.10).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
LOCK = (REPO_ROOT / "poetry.lock").read_text(encoding="utf-8")


def _python_constraint():
    match = re.search(r'^python = "([^"]+)"', PYPROJECT, re.MULTILINE)
    assert match, "pyproject.toml must declare a python constraint"
    return match.group(1)


def test_requires_python_310_or_newer():
    # "<4.0" is needed because cohere declares requires-python "<4.0,>=3.9" (Poetry needs the
    # project range to fit inside every dependency's range); the old "^3.9" also meant "<4.0".
    assert _python_constraint() == ">=3.10,<4.0"


def test_lock_file_matches_the_python_constraint():
    match = re.search(r'^python-versions = "([^"]+)"', LOCK.split("[metadata]", 1)[1], re.MULTILINE)
    assert match and match.group(1) == _python_constraint()


def test_lock_has_no_python_39_only_resolutions():
    assert 'python_version < "3.10"' not in LOCK
    assert "python_version < '3.10'" not in LOCK
    names = re.findall(r'^\[\[package\]\]\nname = "([^"]+)"', LOCK, re.MULTILINE)
    duplicated = sorted({name for name in names if names.count(name) > 1})
    # numpy is split on purpose (pyproject.toml): 2.2.x for Python 3.10, 2.3+ (with 3.14 wheels) for 3.11+.
    assert duplicated in ([], ["numpy"]), f"unexpected packages locked at several versions: {duplicated}"


def test_no_dependency_marker_mentions_python_39():
    settings = [line for line in PYPROJECT.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in settings if ">=3.9" in line or "^3.9" in line]


def test_numpy_split_gives_every_supported_python_a_release_with_wheels():
    blocks = [block for block in LOCK.split("[[package]]")[1:] if re.search(r'^name = "numpy"$', block, re.MULTILINE)]
    wheels = set()
    for block in blocks:
        wheels |= set(re.findall(r"-cp(3\d+)-", block))
    assert {"310", "311", "312", "313", "314"} <= wheels, sorted(wheels)


# --- PyPI metadata (#108): frozen at publish time, so pinned by tests -------------------------------


def _poetry_value(key):
    section = PYPROJECT.split("[tool.poetry]", 1)[1].split("\n[", 1)[0]
    match = re.search(rf'^{key} = "([^"]*)"', section, re.MULTILINE)
    return match.group(1) if match else None


def test_description_is_one_honest_line():
    description = _poetry_value("description")
    assert description and "\n" not in description and len(description) <= 120, description
    for claim in ("ChromaDB", "scalable", "trust", "abstain"):
        assert claim.lower() not in description.lower(), f"description must not claim {claim!r}"


def test_author_email_is_well_formed():
    authors = re.search(r"^authors = \[(.*)\]$", PYPROJECT, re.MULTILINE).group(1)
    assert re.fullmatch(r'"Mrutunjay Kinagi <[^\s<>@]+@[^\s<>@]+\.[a-z]+>"', authors), authors


def test_project_urls_are_declared():
    assert _poetry_value("homepage") and _poetry_value("repository") and _poetry_value("documentation")
    urls = PYPROJECT.split("[tool.poetry.urls]", 1)[1].split("\n[", 1)[0]
    assert re.search(r'^"?Changelog"? = "https://', urls, re.MULTILINE), urls
    assert re.search(r'^"?Issues"? = "https://', urls, re.MULTILINE), urls


def test_keywords_and_classifiers_are_declared():
    keywords = re.search(r"^keywords = \[(.*?)\]", PYPROJECT, re.MULTILINE | re.DOTALL)
    assert keywords and "rag" in keywords.group(1)
    classifiers = re.search(r"^classifiers = \[(.*?)\]", PYPROJECT, re.MULTILINE | re.DOTALL)
    assert classifiers and "Development Status ::" in classifiers.group(1)


# --- Release preparation (#109) ---------------------------------------------------------------------


def test_changelog_has_a_section_for_the_release_being_prepared():
    version = _poetry_value("version")
    final = re.match(r"^(\d+\.\d+\.\d+)", version).group(1)  # 0.2.0rc1 -> 0.2.0
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert re.search(rf"^## \[{re.escape(final)}\]", changelog, re.MULTILINE), f"CHANGELOG.md has no '## [{final}]' section"
    unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    assert not re.search(r"^### ", unreleased, re.MULTILINE), "entries should move from Unreleased into the release section"
