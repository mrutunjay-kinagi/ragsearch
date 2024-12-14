"""
This module contains the RAGSearchEngine class,
which is responsible for initializing the RAG Search Engine
"""
import logging
from typing import List, Dict
import pandas as pd
from cohere import Client as CohereClient
from .utils import (extract_textual_columns,
                    preprocess_search_text,
                    dataframe_to_text,
                    batch_generate_embeddings,
                    insert_embeddings_to_vector_db,
                    search_vector_db,
                    log_data_summary,
                    prepare_embeddings)
from .vector_db import VectorDB
from flask import Flask, request, jsonify, render_template
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RagSearchEngine:
    def __init__(self, data: pd.DataFrame,
                 embedding_model: CohereClient,
                 llm_client: CohereClient,
                 vector_db: VectorDB):
        """
        Initializes the RAG Search Engine with data, an LLM client, and a vector database.

        Args:
            data (pd.DataFrame): The input data containing structured information.
            embedding_model (CohereClient): The client for generating text embeddings.
            llm_client (CohereClient): The client for interacting with the LLM.
            vector_db (VectorDB): The vector database for storing and querying embeddings.
        """
        logging.info("Initializing RAG Search Engine...")
        self.data = data
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.vector_db = vector_db

        # Log data summary
        log_data_summary(self.data)

        # Extract textual columns and preprocess text
        textual_columns = extract_textual_columns(data)
        prepared_data = prepare_embeddings(data, textual_columns, self.embedding_model)

        # Extract Metadata
        metadata_columns = data.select_dtypes(exclude=['object']).columns.to_list()
        insert_embeddings_to_vector_db(self.vector_db, prepared_data, metadata_columns)

        logging.info("RAG Search Engine initialized successfully.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches the vector database for the top-k most relevant results for a given query.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict]: A list of dictionaries containing metadata (excluding embeddings) and similarity scores for each result.
        """
        try:
            logging.info(f"Processing search query: '{query}'")

            # Generate the query embedding
            query_embedding = self.embedding_model.embed(texts=[preprocess_search_text(query)]).embeddings[0]

            # Search the vector database
            results = search_vector_db(self.vector_db, query_embedding, top_k=top_k)
            logging.info(f"Search completed. Found {len(results)} results.")

            # Map indices to metadata and include similarity scores, excluding 'embedding'
            enriched_results = []
            for result in results:
                index = result["index"]
                metadata = self.data.iloc[index].to_dict()

                # Remove the embedding from metadata if it exists
                if "embedding" in metadata:
                    del metadata["embedding"]

                enriched_results.append({
                    "metadata": metadata
                })

            logging.info(f"Found {len(enriched_results)} results for the query.")
            return enriched_results
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise

    def run(self):
        """
        Launches an interactive search interface where users can input queries and see results.
        """
        logging.info("Launching browser-based search interface...")

        # Initialize Flask app
        app = Flask(__name__, template_folder="templates")

        # Route for the index page
        @app.route('/')
        def index():
            return render_template('index.html')  # Serves the HTML web interface

        # Route for handling search queries
        @app.route('/query', methods=['POST'])
        def query():
            request_data = request.get_json()
            query = request_data.get('query')
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400  # Return error if query is missing

            top_k = int(request_data.get('top_k', 5))
            results = self.search(query, top_k=top_k)
            return jsonify({"results": [res['metadata'] for res in results]})

        # Run the Flask app on a separate thread
        threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8080, "use_reloader": False}).start()
