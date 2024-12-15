"""
This module contains the VectorDB class which is responsible for
managing the FAISS index and associated metadata.
"""
import faiss
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

    def insert(self, embedding: list, metadata: dict):
        """
        Inserts an embedding and associated metadata into the FAISS index.

        Args:
            embedding (list): The embedding to insert.
            metadata (dict): The metadata associated with the embedding.
        Raises:
            ValueError: If the embedding dimension does not match the index dimension
        """
        try:
            # Check if the embedding has the correct shape
            if len(embedding) != self.index.d:
                raise ValueError(f"Embedding dimension mismatch: expected {self.index.d}, got {len(embedding)}")

            # Normalize the embedding
            normalized_embedding = self._normalize_embedding(embedding)
            normalized_embedding = np.array([normalized_embedding], dtype=np.float32)

            # Add to FAISS index
            self.index.add(normalized_embedding)
            self.metadata_store[self.current_id] = metadata
            logging.info(f"Embedding inserted successfully with ID: {self.current_id}")
            self.current_id += 1  # Increment the ID for the next embedding
        except Exception as e:
            logging.error(f"Failed to insert embedding: {e}")
            raise

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
