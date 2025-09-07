import os
import chromadb

chromadb_sqlite_path = os.environ.get("CHROMADB_SQLITE_PATH", "chroma.sqlite3")
collection_name = os.environ.get("CHROMADB_COLLECTION_NAME", "test_collection")

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