import chromadb

# Create a persistent client (new API)
client = chromadb.PersistentClient(path="/workspaces/ragsearch/chroma.sqlite3")

# Create a new collection
collection = client.create_collection(name="test_collection")

# Add some sample documents
documents = [
    "This is a test document about chicken recipes.",
    "Another document about vegetarian food.",
    "A third document about desserts and cakes."
]
metadatas = [
    {"category": "Non-veg"},
    {"category": "Veg"},
    {"category": "Dessert"}
]
ids = ["doc1", "doc2", "doc3"]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("ChromaDB SQLite file created at: /workspaces/ragsearch/chroma.sqlite3")
