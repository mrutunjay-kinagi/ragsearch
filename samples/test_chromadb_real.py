import os
import pytest
import chromadb

chromadb_sqlite_path = os.environ.get("CHROMADB_SQLITE_PATH", "chroma.sqlite3")
collection_name = os.environ.get("CHROMADB_COLLECTION_NAME", "test_collection")

@pytest.mark.skipif(
    not os.path.exists(chromadb_sqlite_path),
    reason=f"ChromaDB SQLite file not found at '{chromadb_sqlite_path}'. Set CHROMADB_SQLITE_PATH or run create_chromadb.py first."
)
def test_chromadb_real_query():
    client = chromadb.PersistentClient(path=chromadb_sqlite_path)
    collection = client.get_collection(name=collection_name)
    queries = [
        "chicken",
        "vegetarian",
        "dessert",
        "cakes",
        "food"
    ]
    for query in queries:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"Query: '{query}'\nResults: {results}\n")
        assert isinstance(results, dict)
        assert "ids" in results