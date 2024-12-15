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
                    preprocess_text,
                    insert_embeddings_to_vector_db,
                    search_vector_db,
                    log_data_summary)
from .vector_db import VectorDB
from flask import Flask, request, jsonify, render_template
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RagSearchEngine:
    def __init__(self,
                 data: pd.DataFrame,
                 embedding_model: CohereClient,
                 llm_client: CohereClient,
                 vector_db: VectorDB,
                 batch_size: int = 100,
                 save_dir: str = "embeddings",
                 file_name: str = "data.csv"):
        """
        Initializes the RAG Search Engine with data, an LLM client, and a vector database.

        Args:
            data (pd.DataFrame): The input data containing structured information.
            embedding_model (CohereClient): The client for generating text embeddings.
            llm_client (CohereClient): The client for interacting with the LLM.
            vector_db (VectorDB): The vector database for storing and querying embeddings.
            batch_size (int): Number of rows to process in each batch.
            save_dir (str): Directory to save intermediate embeddings.
        """
        logging.info("Initializing RAG Search Engine...")
        self.data = data
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.vector_db = vector_db
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.file_name = file_name

        # Ensure the embeddings directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Log data summary
        log_data_summary(self.data)

        # Extract textual columns
        textual_columns = extract_textual_columns(data)

        # Process embeddings in batches and save results
        self._process_and_store_embeddings(textual_columns)

        logging.info("RAG Search Engine initialized successfully.")

    def _process_and_store_embeddings(self, textual_columns: list):
        """
        Processes and stores embeddings in batches, saving to the vector database incrementally.

        Args:
            textual_columns (list): The list of columns to combine for text embeddings.
        """
        # Combine textual fields into a single text column
        self.data["combined_text"] = self.data.apply(lambda row: preprocess_text(row, textual_columns), axis=1)

        # Split data into batches
        batches = [self.data.iloc[i:i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        logging.info(f"Data split into {len(batches)} batches (batch size: {self.batch_size})")

        for batch_idx, batch in enumerate(batches):
            try:
                logging.info(f"Processing batch {batch_idx + 1} with {len(batch)} records...")

                # Generate embeddings
                response = self.embedding_model.embed(texts=batch["combined_text"].tolist())
                embeddings = response.embeddings

                # Add embeddings to the batch DataFrame
                batch["embedding"] = embeddings

                # Insert embeddings and metadata into the vector database
                metadata_columns = self.data.columns.difference(["embedding"]).tolist()
                insert_embeddings_to_vector_db(self.vector_db, batch, metadata_columns)

                logging.info(f"Batch {batch_idx + 1} successfully stored in the vector database.")
            except Exception as e:
                logging.error(f"Failed to process batch {batch_idx + 1}: {e}")

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

        @app.route('/data-info', methods=['GET'])
        def data_info():
            num_records = len(self.data)
            columns = list(self.data.columns)
            return jsonify({
                "file_name": self.file_name,
                "num_records": num_records,
                "columns": columns
            })

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
