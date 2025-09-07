import os
import chromadb
from chromadb.utils import embedding_functions

chromadb_sqlite_path = os.environ.get("CHROMADB_SQLITE_PATH", "chroma.sqlite3")
client = chromadb.PersistentClient(path=chromadb_sqlite_path)

collection_name = os.environ.get("CHROMADB_COLLECTION_NAME", "test_collection")
collection = client.get_or_create_collection(name=collection_name)

documents = [
    "Chicken curry recipe",
    "Vegetarian pasta recipe",
    "Chocolate cake recipe"
]
metadatas = [
    {"type": "main"},
    {"type": "main"},
    {"type": "dessert"}
]
ids = ["doc1", "doc2", "doc3"]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Collection '{collection_name}' created at '{chromadb_sqlite_path}' with sample data.")