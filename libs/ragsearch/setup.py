"""
This module contains the setup function for the RAG search engine.

Supports both FAISS and ChromaDB as vector database backends.

Example usage (FAISS, default):
    from ragsearch import setup
    engine = setup(data_path, llm_api_key)

Example usage (ChromaDB):
    engine = setup(
        data_path,
        llm_api_key,
        use_chromadb=True,
        chromadb_sqlite_path="/path/to/chroma.sqlite3",
        chromadb_collection_name="your_collection_name"
    )

The returned RagSearchEngine instance will use the selected backend for queries.
"""
import os
from pathlib import Path
import pandas as pd
from cohere import Client as CohereClient
from .vector_db import VectorDB, query_chromadb
from .engine import RagSearchEngine

def setup(data_path: Path,
          llm_api_key: str,
          use_chromadb: bool = False,
          chromadb_sqlite_path: str = None,
          chromadb_collection_name: str = None):
    """
    Initializes the RAG search engine.

    Args:
        data_path (Path): The path to the data file.
        llm_api_key (str): The API key for the Cohere client.
    Returns:
        RagSearchEngine: The initialized RAG search engine.
    Raises:
        FileNotFoundError: If the data path does not exist.
        ValueError: If the file type is not supported.
        RuntimeError: If there is an error loading the data,
        initializing the Cohere client, or connecting to the vector database.
    """
    print("Starting setup of the RAG Search Engine...")

    # Validate data path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Load data
    try:
        # Get file name of the data_path
        file_name = data_path.name
        if data_path.suffix == '.csv':
            data = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            data = pd.read_json(data_path)
        elif data_path.suffix in ['.parquet', '.pq']:
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # Initialize Cohere client
    try:
        llm_client = CohereClient(api_key=llm_api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Cohere client: {e}")

    # Connect to vector database or ChromaDB
    if use_chromadb:
        if not chromadb_sqlite_path or not chromadb_collection_name:
            raise ValueError("ChromaDB path and collection name must be provided when use_chromadb is True.")
        engine = RagSearchEngine(
            data=data,
            embedding_model=llm_client,
            llm_client=llm_client,
            vector_db=None,
            file_name=file_name,
            chromadb_sqlite_path=chromadb_sqlite_path,
            chromadb_collection_name=chromadb_collection_name
        )
    else:
        try:
            vector_db = VectorDB(embedding_dim=4096)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vector database: {e}")
        engine = RagSearchEngine(
            data=data,
            embedding_model=llm_client,
            llm_client=llm_client,
            vector_db=vector_db,
            file_name=file_name
        )

    print("Setup complete.")
    return engine
