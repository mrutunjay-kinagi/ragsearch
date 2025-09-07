"""
Sample: Querying a real ChromaDB SQLite file and collection using ragsearch.

This example demonstrates how to use the ragsearch package to query a ChromaDB collection with natural language queries.
Ensure you have run create_chromadb.py to generate the test database.
"""
import os
from pathlib import Path
from ragsearch import setup

def main():
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

if __name__ == "__main__":
    main()
