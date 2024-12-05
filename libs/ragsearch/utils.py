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

    Args:
        data (pd.DataFrame): The input DataFrame.
    Returns:
        list: A list of column names containing textual data.
    Raises:
        Exception: If an error occurs during column extraction.
    """
    try:
        textual_columns = data.select_dtypes(include=['object']).columns.to_list()
        logging.info(f"Extracted textual columns: {textual_columns}")
        return textual_columns
    except Exception as e:
        logging.error(f"Failed to extract textual columns: {e}")
        raise

def preprocess_text(text: str) -> str:
    """
    Preprocess text by stripping whitespace and converting to lowercase.

    Args:
        text (str): The input text string.
    Returns:
        str: The preprocessed text string.
    """
    return text.strip().lower()

def dataframe_to_text(data: pd.DataFrame, columns: list) -> list:
    """
    Convert specified columns of a DataFrame to a list of text strings.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list): The list of column names to convert to text.
    Returns:
        list: A list of text strings.
    Raises:
        Exception: If an error occurs during text conversion.
    """
    try:
        text_list = data[columns].apply(lambda row: ' '.join(row.astype(str).values), axis=1).tolist()
        logging.info("Converted DataFrame columns to text successfully.")
        return text_list
    except Exception as e:
        logging.error(f"Failed to convert DataFrame columns to text: {e}")
        raise

def batch_generate_embeddings(embedding_model, texts: list) -> list:
    """
    Generate embeddings for a batch of text data using the embedding model.

    Args:
        embedding_model: The embedding model to use for generating embeddings.
        texts (list): A list of text strings.
    Returns:
        list: A list of embeddings corresponding to the input text strings.
    Raises:
        Exception: If an error occurs during embedding generation
    """
    try:
        response = embedding_model.embed(texts=texts)
        embeddings = response.embeddings
        logging.info("Generated embeddings successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        raise

def insert_embeddings_to_vector_db(vector_db, embeddings: list, metadata: list):
    """
    Insert generated embeddings into the vector database.

    Args:
        vector_db: The vector database to store the embeddings.
        embeddings (list): A list of embeddings to store.
        metadata (list): A list of metadata associated with the embeddings.
    Raises:
        Exception: If an error occurs during insertion of embeddings
    """
    try:
        for idx, embedding in enumerate(embeddings):
            vector_db.insert(embedding=embedding, metadata=metadata[idx])
        logging.info("Embeddings stored in the vector database successfully.")
    except Exception as e:
        logging.error(f"Failed to store embeddings in vector database: {e}")
        raise

def search_vector_db(vector_db, query_embedding: list, top_k: int = 5) -> list:
    """
    Search for the top-k most relevant results in the vector database for a given query embedding.

    Args:
        vector_db: The vector database to search.
        query_embedding (list): The query embedding to search for.
        top_k (int): The number of top results to return.
    Returns:
        list: A list of dictionaries containing the search results.
    Raises:
        Exception: If an error occurs during the search process.
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

    Args:
        data (pd.DataFrame): The input DataFrame to summarize
    """
    logging.info(f"Data Summary:\nShape: {data.shape}\nData Types:\n{data.dtypes}")
