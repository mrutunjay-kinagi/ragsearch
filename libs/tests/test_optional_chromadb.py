"""
chromadb is an optional extra (#125): ``pip install ragsearch[chromadb]``.

The base install must work without it, import it only in ChromaDB mode, and explain how to
install it when it is missing.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from libs.ragsearch.errors import MissingOptionalDependencyError, RagSearchError
from libs.ragsearch.setup import setup
from libs.ragsearch.vector_db import query_chromadb
from libs.tests.fakes import CannedLLMClient, KeywordEmbeddingModel

# Patch the module object: ``libs.ragsearch.setup`` as an attribute path is the setup() function (#91).
ragsearch_setup_module = importlib.import_module("libs.ragsearch.setup")

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_HINT = "pip install 'ragsearch[chromadb]'"


@pytest.fixture
def without_chromadb(monkeypatch):
    """Make ``import chromadb`` fail, as in a base install without the extra."""
    for name in [name for name in sys.modules if name == "chromadb" or name.startswith("chromadb.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "chromadb", None)


@pytest.fixture
def offline_providers(monkeypatch):
    monkeypatch.setattr(ragsearch_setup_module, "CohereClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(ragsearch_setup_module, "create_embedding_model", lambda **kwargs: KeywordEmbeddingModel(["claim"]))
    monkeypatch.setattr(ragsearch_setup_module, "create_llm_client", lambda **kwargs: CannedLLMClient("ok [1]"))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "claims.csv"
    path.write_text("claim_id,description\nC1,roof claim\n", encoding="utf-8")
    return path


def test_importing_ragsearch_does_not_import_chromadb(tmp_path):
    code = (
        "import sys\n"
        "import ragsearch, ragsearch.setup, ragsearch.engine, ragsearch.vector_db\n"
        "from ragsearch import setup, RagSearchEngine\n"
        "assert 'chromadb' not in sys.modules, 'chromadb was imported eagerly'\n"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT / "libs"), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def test_default_faiss_setup_works_without_chromadb(csv_path, tmp_path, without_chromadb, offline_providers):
    engine = setup(csv_path, llm_api_key="offline-key", embeddings_dir=str(tmp_path / "cache"))

    assert engine.answer("roof claim", top_k=1)["answer"] == "ok [1]"


def test_chromadb_mode_without_the_extra_raises_a_clear_error(csv_path, tmp_path, without_chromadb, offline_providers):
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        setup(
            csv_path,
            llm_api_key="offline-key",
            use_chromadb=True,
            chromadb_sqlite_path=str(tmp_path / "chroma"),
            chromadb_collection_name="claims",
        )

    assert INSTALL_HINT in str(excinfo.value)
    assert isinstance(excinfo.value, RagSearchError) and isinstance(excinfo.value, ImportError)


def test_querying_chromadb_without_the_extra_raises_a_clear_error(tmp_path, without_chromadb):
    with pytest.raises(MissingOptionalDependencyError, match=re.escape(INSTALL_HINT)):
        query_chromadb(str(tmp_path / "chroma"), "claims", "roof", n_results=1)


def test_chromadb_is_an_optional_extra_in_pyproject():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    main = pyproject.split("[tool.poetry.dependencies]", 1)[1].split("\n[", 1)[0]

    assert re.search(r'^chromadb = \{[^}]*optional = true[^}]*\}$', main, re.MULTILINE), main
    extras = pyproject.split("[tool.poetry.extras]", 1)[1].split("\n[", 1)[0]
    assert re.search(r'^chromadb = \["chromadb"\]$', extras, re.MULTILINE), extras
