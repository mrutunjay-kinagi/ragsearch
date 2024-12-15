"""
Utility functions for the ragsearch package.
"""
import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_textual_columns(data: pd.DataFrame) -> list:
    """
    Extract columns containing textual data from a DataFrame.
    """
    try:
        textual_columns = data.select_dtypes(include=['object']).columns.to_list()
        logging.info(f"Extracted textual columns: {textual_columns}")
        return textual_columns
    except Exception as e:
        logging.error(f"Failed to extract textual columns: {e}")
        raise

def preprocess_search_text(text: str) -> str:
    """
    Preprocess text by stripping whitespace and converting to lowercase.

    Args:
        text (str): The input text string.
    Returns:
        str: The preprocessed text string.
    """
    return text.strip().lower()

def preprocess_text(row, columns):
    """
    Preprocess text data in the specified columns of a DataFrame row.
    """
    return " | ".join(str(row[col]) if pd.notna(row[col]) else "" for col in columns)


def batch_generate_embeddings(embedding_model, texts: list) -> list:
    """
    Generate embeddings for a batch of text data using the embedding model.
    """
    try:
        response = embedding_model.embed(texts=texts)
        embeddings = response.embeddings
        logging.info("Generated embeddings successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise

def insert_embeddings_to_vector_db(vector_db, data, metadata_columns):
    """
    Inserts embeddings and associated metadata into the vector database.
    """
    try:
        for _, row in data.iterrows():
            embedding = row["embedding"]
            metadata = {col: row[col] for col in metadata_columns}
            vector_db.insert(embedding=embedding, metadata=metadata)

        logging.info("Embeddings and metadata successfully stored in the vector database.")
    except Exception as e:
        logging.error(f"Failed to store embeddings in vector database: {e}")
        raise

def search_vector_db(vector_db, query_embedding: list, top_k: int = 5) -> list:
    """
    Search for the top-k most relevant results in the vector database for a given query embedding.
    """
    try:
        results = vector_db.search(query_embedding, top_k=top_k)
        logging.info(f"Search completed. Found {len(results)} results.")
        return results
    except Exception as e:
        logging.error(f"Failed to search in vector database: {e}")
        raise

def log_data_summary(data: pd.DataFrame):
    """
    Log a summary of the DataFrame, including its shape and data types.
    """
    logging.info(f"Data Summary:\nShape: {data.shape}\nData Types:\n{data.dtypes}")
