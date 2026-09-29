"""
Regression tests for the public import surface and README code examples (#91).

Import checks run in a fresh subprocess: in-process tests are contaminated by
modules that earlier tests already imported.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "libs"
README = REPO_ROOT / "README.md"

_PYTHON_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_RAGSEARCH_IMPORT = re.compile(r"^\s*(from ragsearch[\w.]* import .+|import ragsearch[\w.]*.*)$", re.MULTILINE)


def _readme_python_blocks():
    return _PYTHON_BLOCK.findall(README.read_text(encoding="utf-8"))


def _run_fresh(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=tmp_path,
        env={"PYTHONPATH": str(PACKAGE_ROOT), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )


_ASSERT_PUBLIC_NAMES = """
import inspect
import ragsearch
from ragsearch import setup, RagSearchEngine
assert inspect.isfunction(setup), type(setup)
assert inspect.isclass(RagSearchEngine), type(RagSearchEngine)
assert inspect.isfunction(ragsearch.setup), type(ragsearch.setup)
"""


@pytest.mark.parametrize(
    "prelude",
    [
        pytest.param("", id="fresh"),
        pytest.param("import ragsearch.setup", id="submodule-setup-imported-first"),
        pytest.param("import ragsearch.engine", id="submodule-engine-imported-first"),
        pytest.param("from ragsearch.setup import setup as _s", id="from-submodule-first"),
        pytest.param("from ragsearch import setup as _s", id="repeated-import"),
    ],
)
def test_public_names_resolve_to_function_and_class(prelude, tmp_path):
    result = _run_fresh(prelude + "\n" + _ASSERT_PUBLIC_NAMES, tmp_path)

    assert result.returncode == 0, result.stderr


def test_setup_submodule_import_keeps_working(tmp_path):
    code = (
        "import inspect, sys\n"
        "from ragsearch.setup import setup\n"
        "assert inspect.isfunction(setup), type(setup)\n"
        "assert inspect.ismodule(sys.modules['ragsearch.setup'])\n"
    )

    result = _run_fresh(code, tmp_path)

    assert result.returncode == 0, result.stderr


def test_readme_import_lines_run_in_fresh_interpreter(tmp_path):
    import_lines = sorted(
        {line.strip() for block in _readme_python_blocks() for line in _RAGSEARCH_IMPORT.findall(block)}
    )
    assert "from ragsearch import setup" in import_lines

    code = "\n".join(import_lines) + "\nimport inspect\nassert inspect.isfunction(setup), type(setup)\n"
    result = _run_fresh(code, tmp_path)

    assert result.returncode == 0, result.stderr


def test_readme_python_blocks_compile():
    blocks = _readme_python_blocks()
    assert blocks, "README.md has no ```python code blocks"

    for number, block in enumerate(blocks, start=1):
        try:
            compile(block, f"README.md python block {number}", "exec")
        except SyntaxError as exc:
            pytest.fail(f"README.md python block {number} does not compile: {exc}\n{block}")
