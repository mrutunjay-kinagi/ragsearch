"""
Tests for .github/ci/check_release.py, the publish workflow's release gate (#107).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "ci" / "check_release.py"


def _run(tmp_path, version, tag, prerelease, github_output=None):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[tool.poetry]\nname = "ragsearch"\nversion = "{version}"\n', encoding="utf-8")
    env = {"PATH": ""}
    if github_output is not None:
        env["GITHUB_OUTPUT"] = str(github_output)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--tag", tag, "--prerelease", prerelease, "--pyproject", str(pyproject)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_final_release_matching_the_version_publishes_to_pypi(tmp_path):
    output = tmp_path / "github_output"
    result = _run(tmp_path, "0.2.0", "v0.2.0", "false", github_output=output)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == "target=pypi"


@pytest.mark.parametrize("version", ["0.2.0rc1", "0.2.0a1", "0.2.0b2", "1.0.0rc10"])
def test_prerelease_candidate_publishes_to_testpypi(tmp_path, version):
    output = tmp_path / "github_output"
    result = _run(tmp_path, version, f"v{version}", "true", github_output=output)

    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8").strip() == "target=testpypi"


@pytest.mark.parametrize(
    "version, tag, prerelease, message",
    [
        ("0.2.0", "v0.2.1", "false", "does not match"),
        ("0.2.0", "0.2.0", "false", "must start with 'v'"),
        ("0.2.0rc1", "v0.2.0", "true", "does not match"),
        ("0.2.0", "v0.2.0", "true", "marked as a pre-release"),
        ("0.2.0rc1", "v0.2.0rc1", "false", "must be published as a GitHub pre-release"),
        ("0.2.0.dev1", "v0.2.0.dev1", "true", "not a supported release version"),
        ("0.2", "v0.2", "false", "not a supported release version"),
    ],
)
def test_mismatches_fail_before_anything_is_built(tmp_path, version, tag, prerelease, message):
    result = _run(tmp_path, version, tag, prerelease)

    assert result.returncode != 0
    assert message in result.stderr


def test_real_pyproject_version_is_supported(tmp_path):
    version = subprocess.run(
        [sys.executable, str(SCRIPT), "--print-version", "--pyproject", str(REPO_ROOT / "pyproject.toml")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip()
