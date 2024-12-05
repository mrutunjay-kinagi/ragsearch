"""
This module contains the VectorDB class which is responsible for managing the FAISS index.
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
        self.index = faiss.IndexFlatL2(embedding_dim)
        logging.info(f"FAISS VectorDB initialized with dimension: {embedding_dim}")

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

            embedding = np.array([embedding], dtype=np.float32)
            self.index.add(embedding)
            logging.info("Embedding inserted successfully.")
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
            list: A list of dictionaries containing the results with similarity scores.
        """
        try:
            query_embedding = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_embedding, top_k)
            results = [{"index": idx, "distance": dist} for idx, dist in zip(indices[0], distances[0])]
            logging.info(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logging.error(f"Failed to search in vector database: {e}")
            raise
