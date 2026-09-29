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
