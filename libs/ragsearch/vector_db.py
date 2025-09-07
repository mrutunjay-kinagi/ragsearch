"""
This module contains the VectorDB class which is responsible for
managing the FAISS index and associated metadata.
"""
import faiss
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



import chromadb
from chromadb.config import Settings

class VectorDB:
    def __init__(self, embedding_dim: int = 1024):
        """
        Initializes the FAISS vector database with an in-memory index.

        Args:
            embedding_dim (int): The dimension of the embeddings to be stored.
        Raises:
            ValueError: If the embedding dimension is not a positive integer
        """
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        # Use IndexFlatIP for cosine similarity (requires normalized embeddings)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata_store = {}  # Dictionary to store metadata
        self.current_id = 0  # Incremental ID to track embeddings
        logging.info(f"FAISS VectorDB initialized with dimension: {embedding_dim}")

def get_chromadb_collection(sqlite_path: str, collection_name: str):
    """
    Connects to a ChromaDB SQLite file and returns the specified collection using the new PersistentClient API.
    """
    client = chromadb.PersistentClient(path=sqlite_path)
    return client.get_collection(collection_name)

def query_chromadb(sqlite_path: str, collection_name: str, query_text: str, n_results: int = 5):
    """
    Query the ChromaDB collection for similar documents to the query_text.
    """
    collection = get_chromadb_collection(sqlite_path, collection_name)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

    @staticmethod
    def _normalize_embedding(embedding: list) -> np.ndarray:
        """
        Normalizes the embedding to a unit vector.

        Args:
            embedding (list): The embedding to normalize.
        Returns:
            np.ndarray: The normalized embedding.
        """
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return embedding / norm

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """
        Searches for the top-k most similar embeddings in the FAISS index.

        Args:
            query_embedding (list): The query embedding to search for.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of dictionaries containing the results with similarity scores and metadata.
        """
        try:
            if self.index.ntotal == 0:
                raise ValueError("The FAISS index is empty. Add embeddings before searching.")

            # Normalize the query embedding
            normalized_query = self._normalize_embedding(query_embedding)
            normalized_query = np.array([normalized_query], dtype=np.float32)

            # Perform the search
            distances, indices = self.index.search(normalized_query, top_k)

            # Map indices to metadata
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:  # Check if a valid result is returned
                    metadata = self.metadata_store.get(idx, {})
                    results.append({"index": idx, "similarity": dist, "metadata": metadata})

            logging.info(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logging.error(f"Failed to search in vector database: {e}")
            raise
