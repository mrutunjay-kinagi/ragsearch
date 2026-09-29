"""
This module contains the RAGSearchEngine class,
which is responsible for initializing the RAG Search Engine
"""
import logging
import hashlib
import copy
import json
import math
from time import perf_counter
from typing import Any, Dict, List, Optional
import pandas as pd
from .errors import NoDataFoundError
from .embedding_models import EmbeddingModel, describe_embedding_model, extract_embeddings
from .llm_clients import LLMClient
from .chunking import ChunkingStrategy, RowChunkingStrategy, describe_chunking_strategy
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

# v2 adds the embedding model identity and dimension the cached vectors were produced with;
# v3 adds the chunking strategy and its settings.
EMBEDDING_MANIFEST_VERSION = 3

# Default cap on the estimated tokens of the numbered sources that answer() sends to the LLM.
DEFAULT_MAX_CONTEXT_TOKENS = 3000

# Characters per token used by estimate_tokens(). Measured with Cohere's tokenizer on the claim
# documents and an 80-page NIST PDF at 3.55-4.87 (4.27 overall), so 3.5 errs on the high side.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Approximate token count of ``text``, erring on the high side (no tokenizer dependency)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class RagSearchEngine:
    @staticmethod
    def _normalize_optional_text(value) -> str:
        """Normalize optional metadata text fields to stable API strings."""
        if value is None:
            return ""
        # pd.isna on list/array values is element-wise, so stringify non-scalars directly.
        if not pd.api.types.is_scalar(value):
            return str(value).strip()
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
                 chromadb_collection_name: str = None,
                 max_context_tokens: Optional[int] = DEFAULT_MAX_CONTEXT_TOKENS):
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
        self.chunking_id = describe_chunking_strategy(self.chunking_strategy)
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0 when provided")
        self.max_context_tokens = max_context_tokens
        self.reranker = reranker or NoOpReranker()
        if observability_max_events is not None and observability_max_events <= 0:
            raise ValueError("observability_max_events must be > 0 when provided")
        self.observability_max_events = observability_max_events
        self.chromadb_sqlite_path = chromadb_sqlite_path
        self.chromadb_collection_name = chromadb_collection_name
        self.index_data = data
        self.observability_events: List[Dict[str, Any]] = []
        self.embedding_model_id = describe_embedding_model(embedding_model)
        self.indexing_diagnostics = {
            "manifest_version": EMBEDDING_MANIFEST_VERSION,
            "manifest_path": "",
            "total_records": 0,
            "embedded_records": 0,
            "reused_records": 0,
            "new_records": 0,
            "changed_records": 0,
            "cache_invalidated_reason": "",
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
        expected_dim = getattr(self.vector_db, "embedding_dim", None)
        cache_invalidated_reason = self._embedding_cache_mismatch(manifest, expected_dim)
        if cache_invalidated_reason:
            logging.warning(
                "Embedding cache %s does not match the current embedding model (%s). "
                "One-time re-embedding of %d records; later runs reuse the rebuilt cache.",
                manifest_path,
                cache_invalidated_reason,
                len(self.index_data),
            )
            manifest["records"] = {}
        manifest["version"] = EMBEDDING_MANIFEST_VERSION
        manifest["embedding_model"] = self.embedding_model_id
        manifest["embedding_dim"] = expected_dim
        manifest["chunking"] = self.chunking_id

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

                    if (
                        cached
                        and cached.get("content_hash") == content_hash
                        and (expected_dim is None or len(cached["embedding"]) == expected_dim)
                    ):
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
                    if manifest["embedding_dim"] is None and new_embeddings:
                        manifest["embedding_dim"] = len(new_embeddings[0])

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
            "cache_invalidated_reason": cache_invalidated_reason,
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

    def _embedding_cache_mismatch(self, manifest: Dict[str, Any], expected_dim: Optional[int]) -> str:
        """Return why cached embeddings cannot be reused with the current model, or "" if they can."""
        if not manifest["records"]:
            return ""
        cached_model = manifest.get("embedding_model")
        if not cached_model:
            return (
                "the cache has no embedding model metadata (written by ragsearch 0.1.5 or earlier); "
                f"current model is {self.embedding_model_id}"
            )
        if cached_model != self.embedding_model_id:
            return f"embedding model changed from {cached_model} to {self.embedding_model_id}"
        cached_dim = manifest.get("embedding_dim")
        if expected_dim is not None and cached_dim is not None and cached_dim != expected_dim:
            return f"embedding dimension changed from {cached_dim} to {expected_dim}"
        # Manifests written before chunking was recorded always used row-level chunking.
        cached_chunking = manifest.get("chunking") or describe_chunking_strategy(RowChunkingStrategy())
        if cached_chunking != self.chunking_id:
            return f"chunking changed from {cached_chunking} to {self.chunking_id}"
        return ""

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
        embedding_model = payload.get("embedding_model") if isinstance(payload, dict) else None
        embedding_dim = payload.get("embedding_dim") if isinstance(payload, dict) else None
        chunking = payload.get("chunking") if isinstance(payload, dict) else None
        return {
            "version": int(version),
            "embedding_model": embedding_model if isinstance(embedding_model, str) else None,
            "embedding_dim": embedding_dim if isinstance(embedding_dim, int) else None,
            "chunking": chunking if isinstance(chunking, str) else None,
            "records": normalized_records,
        }

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

                # Excerpt is a short display snippet; answer context uses the full chunk text.
                excerpt = self._result_text(metadata)[:200]

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
    def _result_text(metadata: Dict) -> str:
        """Return the full indexed text for a retrieval result.

        Chunked rows carry the chunk in ``combined_text`` while ``text`` still holds the
        whole source document, so prefer the chunk whenever the row was chunked.
        """
        if "chunk_index" in metadata:
            candidates = (metadata.get("combined_text"), metadata.get("text"))
        else:
            candidates = (metadata.get("text"), metadata.get("combined_text"))
        for candidate in candidates:
            text = RagSearchEngine._normalize_optional_text(candidate)
            if text:
                return text
        return ""

    @staticmethod
    def _build_answer_context(results: List[Dict]) -> str:
        """Build a numbered retrieval context block for answer generation."""
        if not results:
            return ""

        blocks = []
        for position, result in enumerate(results, start=1):
            citation = result.get("citation", {})
            metadata = result.get("metadata", {})
            excerpt = RagSearchEngine._result_text(metadata) or citation.get("excerpt") or ""
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
    def _assemble_budgeted_context(results: List[Dict], max_context_tokens: Optional[int]) -> tuple:
        """Build the numbered sources block within a token budget.

        Sources are added in score order until the next one would exceed the budget. If even the
        first source does not fit, a prefix of its text is used so the model still gets context.
        Returns ``(context, sources_used, truncated)``.
        """
        if max_context_tokens is None:
            return RagSearchEngine._build_answer_context(results), len(results), False

        max_chars = int(max_context_tokens * CHARS_PER_TOKEN)
        context = ""
        used = 0
        for result in results:
            block = RagSearchEngine._build_answer_context([result]).replace("[1]", f"[{used + 1}]", 1)
            candidate = f"{context}\n\n{block}" if context else block
            if len(candidate) <= max_chars:
                context, used = candidate, used + 1
                continue
            if used == 0:
                marker = " [truncated]"
                context = block[: max(0, max_chars - len(marker))] + marker
                context = context[:max_chars]
                used = 1
            break
        truncated = used < len(results) or (used == 1 and context.endswith("[truncated]"))
        return context, used, truncated

    @staticmethod
    def _build_answer_prompt(query: str, results: List[Dict], context: Optional[str] = None) -> str:
        """Construct a deterministic prompt for answer generation."""
        if context is None:
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

    def answer(self, query: str, top_k: int = 5, max_context_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate a grounded answer with preserved retrieval citations.

        The numbered sources sent to the LLM are capped at ``max_context_tokens`` estimated tokens
        (default: the engine's ``max_context_tokens``; ``None`` on the engine means no cap). Sources
        are added in score order; ``context_truncated`` reports whether any were left out or cut,
        and ``citations`` lists only the sources the LLM actually received.
        """
        generation_started = perf_counter()
        results = self.search(query, top_k=top_k)
        budget = self.max_context_tokens if max_context_tokens is None else max_context_tokens
        if budget is not None and budget <= 0:
            raise ValueError("max_context_tokens must be > 0 when provided")
        context, sources_used, truncated = self._assemble_budgeted_context(results, budget)
        prompt = self._build_answer_prompt(query, results, context=context)
        answer_text = self.llm_client.generate(prompt)
        latency_ms = round((perf_counter() - generation_started) * 1000.0, 3)

        self._emit_observability_event(
            stage="generation",
            event="answer_completed",
            payload={
                "query": query,
                "top_k": int(top_k),
                "results_count": int(len(results)),
                "citations_count": int(sources_used),
                "context_truncated": bool(truncated),
                "latency_ms": latency_ms,
            },
        )

        return {
            "question": query,
            "answer": answer_text,
            "results": results,
            "citations": [result.get("citation", {}) for result in results[:sources_used]],
            "context": context,
            "context_sources": int(sources_used),
            "context_truncated": bool(truncated),
            "context_tokens_estimate": estimate_tokens(context),
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
