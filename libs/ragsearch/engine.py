"""
This module contains the RAGSearchEngine class,
which is responsible for initializing the RAG Search Engine
"""
import logging
from typing import List, Dict
import pandas as pd
from cohere import Client as CohereClient
from .utils import (extract_textual_columns,
                    preprocess_text,
                    dataframe_to_text,
                    batch_generate_embeddings,
                    insert_embeddings_to_vector_db,
                    search_vector_db,
                    log_data_summary)
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
        preprocessed_texts = dataframe_to_text(data, textual_columns)

        # Generate and store embeddings
        embeddings = batch_generate_embeddings(self.embedding_model, preprocessed_texts)
        insert_embeddings_to_vector_db(self.vector_db, embeddings, metadata=data.to_dict(orient='records'))

        logging.info("RAG Search Engine initialized successfully.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches the vector database for the top-k most relevant results for a given query.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        Returns:
            List[Dict]: A list of dictionaries containing the search results.
        """
        logging.info(f"Processing search query: '{query}'")
        query_embedding = self.embedding_model.embed(texts=[preprocess_text(query)]).embeddings[0]
        results = search_vector_db(self.vector_db, query_embedding, top_k=top_k)

        logging.info(f"Found {len(results)} results for the query.")
        return results

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
        threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5000, "use_reloader": False}).start()
