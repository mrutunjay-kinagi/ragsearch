"""
This module contains the RAGSearchEngine class,
which is responsible for initializing the RAG Search Engine
"""
import logging
import hashlib
import copy
import json
from time import perf_counter
from typing import Any, Dict, List, Optional
import pandas as pd
from .errors import NoDataFoundError
from .embedding_models import EmbeddingModel, extract_embeddings
from .llm_clients import LLMClient
from .chunking import ChunkingStrategy, RowChunkingStrategy
from .reranking import NoOpReranker, Reranker
from .utils import (extract_textual_columns,
                    preprocess_search_text,
                    preprocess_text,
                    insert_embeddings_to_vector_db,
                    search_vector_db,
                    log_data_summary)
from .vector_backends import VectorBackend
from flask import Flask, request, jsonify, render_template
import threading
from pathlib import Path

class RagSearchEngine:
    @staticmethod
    def _normalize_optional_text(value) -> str:
        """Normalize optional metadata text fields to stable API strings."""
        if value is None:
            return ""
        # Handle pandas missing markers without introducing stringified 'nan'.
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text.lower() in {"none", "nan"}:
            return ""
        return text

    def __init__(self,
                 data: pd.DataFrame,
                 embedding_model: EmbeddingModel,
                 llm_client: LLMClient,
                 vector_db: VectorBackend = None,
                 batch_size: int = 100,
                 save_dir: str = "embeddings",
                 file_name: str = "data.csv",
                 chunking_strategy: Optional[ChunkingStrategy] = None,
                 reranker: Optional[Reranker] = None,
                 observability_max_events: Optional[int] = 1000,
                 chromadb_sqlite_path: str = None,
                 chromadb_collection_name: str = None):
        """
        Initializes the RAG Search Engine with data, an LLM client, and a vector database.

        Args:
            data (pd.DataFrame): The input data containing structured information.
            embedding_model (EmbeddingModel): Embedding provider implementing the embedding contract.
            llm_client (LLMClient): Baseline generation client implementing the LLM contract.
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
        self.chunking_strategy = chunking_strategy or RowChunkingStrategy()
        self.reranker = reranker or NoOpReranker()
        if observability_max_events is not None and observability_max_events <= 0:
            raise ValueError("observability_max_events must be > 0 when provided")
        self.observability_max_events = observability_max_events
        self.chromadb_sqlite_path = chromadb_sqlite_path
        self.chromadb_collection_name = chromadb_collection_name
        self.index_data = data
        self.observability_events: List[Dict[str, Any]] = []
        self.indexing_diagnostics = {
            "manifest_version": 1,
            "manifest_path": "",
            "total_records": 0,
            "embedded_records": 0,
            "reused_records": 0,
            "new_records": 0,
            "changed_records": 0,
        }

        if self.data.empty:
            raise NoDataFoundError("No data found in the provided DataFrame.")

        # Ensure the embeddings directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Log data summary
        log_data_summary(self.data)

        # Extract textual columns
        textual_columns = extract_textual_columns(data)

        # Build index rows before embedding so chunking strategy can expand records.
        if self.vector_db is not None:
            self.index_data = self._build_index_data(textual_columns)

        # Only process embeddings if using FAISS
        if self.vector_db is not None:
            self._process_and_store_embeddings()

        logging.info("RAG Search Engine initialized successfully.")
    def chromadb_search(self, query: str, top_k: int = 5):
        """
        Query the ChromaDB collection for similar documents to the query text.
        """
        from .vector_db import query_chromadb
        if not self.chromadb_sqlite_path or not self.chromadb_collection_name:
            raise ValueError("ChromaDB path and collection name must be set for chromadb_search.")
        return query_chromadb(self.chromadb_sqlite_path, self.chromadb_collection_name, query, n_results=top_k)

    def _process_and_store_embeddings(self):
        """
        Processes and stores embeddings in batches, saving to the vector database incrementally.

        Args:
            textual_columns (list): The list of columns to combine for text embeddings.
        """
        manifest_path = self.save_dir / f"{self.file_name}.embedding_manifest.json"
        manifest = self._load_embedding_manifest(manifest_path)

        total_records = len(self.index_data)
        embedded_records = 0
        reused_records = 0
        new_records = 0
        changed_records = 0

        # Split data into batches
        batches = [self.index_data.iloc[i:i + self.batch_size] for i in range(0, len(self.index_data), self.batch_size)]
        logging.info(f"Data split into {len(batches)} batches (batch size: {self.batch_size})")

        batch_errors = []
        for batch_idx, batch in enumerate(batches):
            try:
                logging.info(f"Processing batch {batch_idx + 1} with {len(batch)} records...")

                resolved_embeddings = []
                pending_positions = []
                pending_texts = []
                pending_keys = []
                pending_hashes = []

                for _, row in batch.iterrows():
                    record_key = self._record_cache_key(row)
                    content_hash = self._content_hash(str(row.get("combined_text", "")))
                    cached = manifest["records"].get(record_key)

                    if cached and cached.get("content_hash") == content_hash:
                        reused_records += 1
                        resolved_embeddings.append(cached.get("embedding", []))
                        continue

                    pending_positions.append(len(resolved_embeddings))
                    pending_texts.append(str(row.get("combined_text", "")))
                    pending_keys.append(record_key)
                    pending_hashes.append(content_hash)
                    if cached:
                        changed_records += 1
                    else:
                        new_records += 1
                    resolved_embeddings.append(None)

                if pending_texts:
                    response = self.embedding_model.embed(texts=pending_texts)
                    new_embeddings = extract_embeddings(response)
                    embedded_records += len(new_embeddings)

                    for offset, embedding in enumerate(new_embeddings):
                        position = pending_positions[offset]
                        resolved_embeddings[position] = embedding
                        manifest["records"][pending_keys[offset]] = {
                            "content_hash": pending_hashes[offset],
                            "embedding": [float(value) for value in embedding],
                        }

                # Add embeddings to the batch DataFrame
                batch = batch.copy()
                batch["embedding"] = resolved_embeddings

                # Insert embeddings and metadata into the vector database
                metadata_columns = self.index_data.columns.difference(["embedding"]).tolist()
                insert_embeddings_to_vector_db(self.vector_db, batch, metadata_columns)

                logging.info(f"Batch {batch_idx + 1} successfully stored in the vector database.")
            except Exception as e:
                logging.error(f"Failed to process batch {batch_idx + 1}: {e}")
                batch_errors.append((batch_idx + 1, e))

        if batch_errors:
            failed = ", ".join(f"batch {i}" for i, _ in batch_errors)
            raise RuntimeError(
                f"Embedding indexing failed for {len(batch_errors)} batch(es): {failed}. "
                f"First error: {batch_errors[0][1]}"
            ) from batch_errors[0][1]

        self._save_embedding_manifest(manifest_path, manifest)
        self.indexing_diagnostics = {
            "manifest_version": int(manifest.get("version", 1)),
            "manifest_path": str(manifest_path),
            "total_records": int(total_records),
            "embedded_records": int(embedded_records),
            "reused_records": int(reused_records),
            "new_records": int(new_records),
            "changed_records": int(changed_records),
        }
        self._emit_observability_event(
            stage="indexing",
            event="indexing_completed",
            payload=self.indexing_diagnostics,
        )

    def _emit_observability_event(self, stage: str, event: str, payload: Dict[str, Any]):
        record = {
            "stage": stage,
            "event": event,
            # Snapshot payload to keep historical events immutable for callers.
            "payload": copy.deepcopy(payload),
        }
        self.observability_events.append(record)
        if self.observability_max_events is not None and len(self.observability_events) > self.observability_max_events:
            self.observability_events = self.observability_events[-self.observability_max_events:]
        logging.info("OBSERVABILITY %s", json.dumps(record, sort_keys=True))

    def _build_index_data(self, textual_columns: list) -> pd.DataFrame:
        rows = []
        is_default_chunking = isinstance(self.chunking_strategy, RowChunkingStrategy)

        for source_record_id, row in self.data.iterrows():
            combined_text = preprocess_text(row, textual_columns)
            chunks = self.chunking_strategy.chunk_text(combined_text)
            if not isinstance(chunks, list):
                raise ValueError("chunking strategy must return a list of text chunks")

            normalized_chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
            if not normalized_chunks:
                normalized_chunks = [combined_text]

            for chunk_index, chunk_text in enumerate(normalized_chunks):
                payload = row.to_dict()
                payload["combined_text"] = chunk_text
                if not is_default_chunking:
                    payload["source_record_id"] = int(source_record_id)
                    payload["chunk_index"] = int(chunk_index)
                rows.append(payload)

        if not rows:
            raise NoDataFoundError("No indexable text chunks generated from input data")
        return pd.DataFrame(rows)

    @staticmethod
    def _content_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _record_cache_key(row: pd.Series) -> str:
        source_path = str(row.get("source_path", "")).strip()
        parser_name = str(row.get("parser_name", "")).strip()
        source_record_id = int(row.get("source_record_id", row.name))
        chunk_index = int(row.get("chunk_index", 0))
        if source_path:
            return f"{source_path}::{parser_name}::{source_record_id}::{chunk_index}"
        return f"row::{source_record_id}::{chunk_index}"

    @staticmethod
    def _load_embedding_manifest(manifest_path: Path) -> Dict[str, Any]:
        if not manifest_path.exists():
            return {"version": 1, "records": {}}

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "records": {}}

        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, dict):
            records = {}

        normalized_records: Dict[str, Dict[str, Any]] = {}
        for key, value in records.items():
            if not isinstance(value, dict):
                continue
            content_hash = value.get("content_hash")
            embedding = value.get("embedding")
            if not isinstance(content_hash, str) or not isinstance(embedding, list):
                continue
            try:
                normalized_embedding = [float(item) for item in embedding]
            except (TypeError, ValueError):
                continue
            normalized_records[str(key)] = {
                "content_hash": content_hash,
                "embedding": normalized_embedding,
            }

        version = payload.get("version", 1) if isinstance(payload, dict) else 1
        return {"version": int(version), "records": normalized_records}

    @staticmethod
    def _save_embedding_manifest(manifest_path: Path, manifest: Dict[str, Any]):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Searches the vector database for the top-k most relevant results for a given query.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.

        Returns:
            List[Dict]: A list of dictionaries containing metadata (excluding embeddings) and similarity scores for each result.
        """
        search_started = perf_counter()
        try:
            logging.info(f"Processing search query: '{query}'")

            # Generate the query embedding
            query_response = self.embedding_model.embed(texts=[preprocess_search_text(query)])
            query_embedding = extract_embeddings(query_response)[0]

            # Search the vector database
            results = search_vector_db(self.vector_db, query_embedding, top_k=top_k)
            logging.info(f"Search completed. Found {len(results)} results.")

            # Map indices to metadata and include similarity scores, excluding 'embedding'
            enriched_results = []
            index_frame = self.index_data if self.vector_db is not None else self.data

            for result in results:
                index = result["index"]
                metadata = index_frame.iloc[index].to_dict()

                # Remove the embedding from metadata if it exists
                if "embedding" in metadata:
                    del metadata["embedding"]

                source_path = self._normalize_optional_text(metadata.get("source_path", ""))
                parser_name = self._normalize_optional_text(metadata.get("parser_name", ""))

                excerpt_source = metadata.get("text") or metadata.get("combined_text") or ""
                excerpt = "" if excerpt_source is None else str(excerpt_source)[:200]

                citation = {
                    "record_id": int(index),
                    "source_path": source_path,
                    "parser_name": parser_name,
                    "excerpt": excerpt,
                }

                enriched_results.append({
                    "metadata": metadata,
                    "citation": citation,
                    "similarity": float(result.get("similarity", 0.0)),
                })

            reranked = self.reranker.rerank(query, enriched_results)
            if not isinstance(reranked, list):
                raise ValueError("reranker must return a list of retrieval results")

            latency_ms = round((perf_counter() - search_started) * 1000.0, 3)
            self._emit_observability_event(
                stage="retrieval",
                event="search_completed",
                payload={
                    "query": query,
                    "top_k": int(top_k),
                    "results_count": int(len(reranked[:top_k])),
                    "latency_ms": latency_ms,
                },
            )

            logging.info(f"Found {len(reranked)} results for the query after reranking.")
            return reranked[:top_k]
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise

    @staticmethod
    def _serialize_query_results(results: List[Dict], include_details: bool = False) -> List[Dict]:
        """
        Serialize search results for API consumers.

        Backward compatibility:
        - Default returns metadata-only entries (legacy behavior).
        - When include_details=True, returns full enriched results including citation and similarity.
        """
        if include_details:
            return results
        return [res.get("metadata", {}) for res in results]

    @staticmethod
    def _build_answer_context(results: List[Dict]) -> str:
        """Build a numbered retrieval context block for answer generation."""
        if not results:
            return ""

        blocks = []
        for position, result in enumerate(results, start=1):
            citation = result.get("citation", {})
            metadata = result.get("metadata", {})
            excerpt = citation.get("excerpt") or metadata.get("text") or metadata.get("combined_text") or ""
            source_path = citation.get("source_path", "")
            parser_name = citation.get("parser_name", "")
            similarity = result.get("similarity", 0.0)

            blocks.append(
                "\n".join(
                    [
                        f"[{position}] source_path: {source_path}",
                        f"parser_name: {parser_name}",
                        f"similarity: {similarity:.4f}",
                        f"excerpt: {excerpt}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _build_answer_prompt(query: str, results: List[Dict]) -> str:
        """Construct a deterministic prompt for answer generation."""
        context = RagSearchEngine._build_answer_context(results)
        return "\n".join(
            [
                "You are a retrieval-augmented assistant.",
                "Answer only from the provided sources.",
                "If the sources are insufficient, say you do not know.",
                "Cite sources inline using bracketed numbers like [1] or [1][2].",
                "Keep the answer concise and grounded in the sources.",
                "",
                f"Question: {query}",
                "",
                "Sources:",
                context or "(no sources retrieved)",
            ]
        )

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate a grounded answer with preserved retrieval citations."""
        generation_started = perf_counter()
        results = self.search(query, top_k=top_k)
        prompt = self._build_answer_prompt(query, results)
        answer_text = self.llm_client.generate(prompt)
        latency_ms = round((perf_counter() - generation_started) * 1000.0, 3)

        self._emit_observability_event(
            stage="generation",
            event="answer_completed",
            payload={
                "query": query,
                "top_k": int(top_k),
                "results_count": int(len(results)),
                "citations_count": int(len(results)),
                "latency_ms": latency_ms,
            },
        )

        return {
            "question": query,
            "answer": answer_text,
            "results": results,
            "citations": [result.get("citation", {}) for result in results],
            "context": self._build_answer_context(results),
        }

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
            include_details_raw = request_data.get('include_details', False)
            if isinstance(include_details_raw, str):
                include_details = include_details_raw.strip().lower() in {'1', 'true', 'yes', 'on'}
            else:
                include_details = bool(include_details_raw)
            results = self.search(query, top_k=top_k)
            serialized = self._serialize_query_results(results, include_details=include_details)
            return jsonify({"results": serialized})

        @app.route('/answer', methods=['POST'])
        def answer():
            request_data = request.get_json()
            query = request_data.get('query')
            if not query:
                return jsonify({"error": "Query parameter is required"}), 400

            top_k = int(request_data.get('top_k', 5))
            return jsonify(self.answer(query, top_k=top_k))

        # Run the Flask app on a separate thread
        threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8080, "use_reloader": False}).start()
