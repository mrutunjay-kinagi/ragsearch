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

def preprocess_search_text(text: str) -> str:
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

def preprocess_text(row, columns):
    """
    Preprocess text data in the specified columns of a DataFrame row.

    Args:
        row (pd.Series): The input DataFrame row.
        columns (list): The list of column names containing text data.
    Returns:
        str: The preprocessed text string.
    """
    return " | ".join(str(row[col]) if pd.notna(row[col]) else "" for col in columns)


def prepare_embeddings(dataframe, textual_columns, embedding_model):
    """
    Prepares the DataFrame by combining textual fields and generating embeddings.

    Args:
        dataframe (pd.DataFrame): Input data with structured fields.
        textual_columns (list): List of columns to combine for text embeddings.
        embedding_model: Preloaded embedding model for generating embeddings.

    Returns:
        pd.DataFrame: Updated DataFrame with 'combined_text' and 'embedding' columns.
    """
    try:
        # Combine textual fields into a single text column
        dataframe["combined_text"] = dataframe.apply(lambda row: preprocess_text(row, textual_columns), axis=1)

        # Generate embeddings using the embedding model
        embeddings_response = embedding_model.embed(texts=dataframe["combined_text"].tolist())

        # Extract the actual embeddings from the response
        embeddings = embeddings_response.embeddings  # Adjust this to match Cohere's API

        # Add embeddings to the DataFrame
        dataframe["embedding"] = embeddings

        logging.info("Generated embeddings and updated the DataFrame successfully.")
        return dataframe
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
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

def insert_embeddings_to_vector_db(vector_db, data, metadata_columns):
    """
    Inserts embeddings and associated metadata into the vector database.

    Args:
        vector_db (VectorDB): The vector database instance.
        data (pd.DataFrame): DataFrame containing 'embedding' and metadata columns.
        metadata_columns (list): Columns to use as metadata for each embedding.

    Returns:
        None
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
