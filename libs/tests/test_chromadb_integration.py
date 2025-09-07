"""
Integration test for ChromaDB backend in ragsearch package.
"""
import os
import pytest
from pathlib import Path
from libs.ragsearch.setup import setup

@pytest.mark.skipif(
    not os.path.exists("/tmp/chroma.sqlite3"),
    reason="ChromaDB SQLite file not found. Provide a test file at /tmp/chroma.sqlite3."
)
def test_chromadb_setup_and_search():
    # These should be set to valid test values
    data_path = Path("/workspaces/ragsearch/libs/tests/sample_data.csv")
    llm_api_key = "test-api-key"  # Use a valid key or mock
    chromadb_sqlite_path = "/tmp/chroma.sqlite3"
    chromadb_collection_name = "test_collection"

    # Initialize engine with ChromaDB backend
    engine = setup(
        data_path,
        llm_api_key,
        use_chromadb=True,
        chromadb_sqlite_path=chromadb_sqlite_path,
        chromadb_collection_name=chromadb_collection_name
    )

    # Run a test query
    query = "Find recipes with chicken"
    results = engine.chromadb_search(query, top_k=3)
    assert isinstance(results, dict)
    assert "ids" in results
    assert len(results["ids"]) <= 3
    # Optionally print results for manual inspection
    print(results)
