"""
Test querying a real ChromaDB SQLite file and collection.
"""
import os
import pytest
from pathlib import Path
from libs.ragsearch.setup import setup

@pytest.mark.skipif(
    not os.path.exists("/workspaces/ragsearch/chroma.sqlite3"),
    reason="ChromaDB SQLite file not found. Run create_chromadb.py first."
)
def test_chromadb_real_query():
    data_path = Path("libs/tests/sample_data.csv")
    llm_api_key = "test-api-key"  # Use a valid key or mock
    chromadb_sqlite_path = "/workspaces/ragsearch/chroma.sqlite3"
    chromadb_collection_name = "test_collection"

    engine = setup(
        data_path,
        llm_api_key,
        use_chromadb=True,
        chromadb_sqlite_path=chromadb_sqlite_path,
        chromadb_collection_name=chromadb_collection_name
    )

    queries = [
        "chicken",
        "vegetarian",
        "dessert",
        "cakes",
        "food"
    ]
    for query in queries:
        results = engine.chromadb_search(query, top_k=3)
        print(f"Query: '{query}'\nResults: {results}\n")
        assert isinstance(results, dict)
        assert "ids" in results
