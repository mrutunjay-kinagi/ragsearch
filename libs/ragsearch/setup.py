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
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from cohere import Client as CohereClient
from .errors import NoDataFoundError, ParsingError, RagSearchError
from .embedding_models import create_embedding_model, infer_embedding_dimension
from .llm_clients import create_llm_client
from .parsers import FallbackParser, LiteParseAdapter, get_parser
from .vector_db import VectorDB
from .engine import RagSearchEngine


# File types loaded directly via pandas (no parser dispatch needed)
STRUCTURED_EXTENSIONS = {".csv", ".json", ".parquet", ".pq"}
SUPPORTED_EMBEDDING_PROVIDERS = {"cohere", "openai", "ollama", "sentence_transformers"}
SUPPORTED_LLM_PROVIDERS = {"cohere", "openai", "ollama"}
logger = logging.getLogger(__name__)


def _normalize_provider_name(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def _validate_provider(provider: str, supported: set[str], kind: str) -> str:
    normalized = _normalize_provider_name(provider)
    if normalized not in supported:
        raise ValueError(
            f"Unsupported {kind} provider: {provider}. Supported providers: {', '.join(sorted(supported))}."
        )
    return normalized


def build_vector_backend(*, embedding_dim: int):
    """Build the default vector backend while preserving legacy setup behavior."""
    return VectorDB(embedding_dim=embedding_dim)


def _load_structured_data(data_path: Path) -> pd.DataFrame:
    if data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    if data_path.suffix == '.json':
        return pd.read_json(data_path)
    if data_path.suffix in ['.parquet', '.pq']:
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported file type: {data_path.suffix}")


def _parser_label(parser) -> str:
    if isinstance(parser, LiteParseAdapter):
        return "liteparse"
    if isinstance(parser, FallbackParser):
        return "fallback"
    return parser.__class__.__name__.lower()


def _load_unstructured_data(data_path: Path, diagnostics: Optional[dict] = None) -> pd.DataFrame:
    """
    Load unstructured data (text, PDF, DOCX, etc.) by dispatching to parser via get_parser().
    
    Returns DataFrame with columns: text, metadata, source_path, parser_name.
    - text: str, non-empty after stripping whitespace
    - metadata: dict, defaults to empty dict if parser returns None
    - source_path: str, normalized to string format
    - parser_name: str, identifies which parser was used
    
    Raises:
        NoDataFoundError: If no parseable content found or all content is empty.
        ValueError: If document structure is invalid (bad text or metadata types).
    
    Integration point: parser contract (see docs/adr/ADR-0002-document-parsing-pipeline.md).
    """
    parser = get_parser(data_path)
    if diagnostics is not None:
        diagnostics["selected_parser"] = _parser_label(parser)
    try:
        documents = list(parser.parse(data_path))
    except ParsingError as exc:
        # If LiteParse fails at runtime, attempt supported fallback parsing.
        if isinstance(parser, LiteParseAdapter):
            fallback = FallbackParser()
            if fallback.supports(data_path):
                logger.warning("LiteParse parsing failed; using fallback parser: %s", exc)
                if diagnostics is not None:
                    diagnostics["status"] = "recovered_with_fallback"
                    diagnostics["failure_reason"] = str(exc)
                    diagnostics["selected_parser"] = _parser_label(fallback)
                try:
                    documents = list(fallback.parse(data_path))
                except ParsingError as fallback_exc:
                    raise exc from fallback_exc
            else:
                raise
        else:
            raise
    if not documents:
        raise NoDataFoundError(f"No data found in parsed input file: {data_path}")

    rows = []
    for document in documents:
        # Validate and normalize text
        if not isinstance(document.text, (str, type(None))):
            raise ValueError(f"Invalid document.text type: {type(document.text).__name__}")
        text = (document.text.strip() if isinstance(document.text, str) else "").strip()
        if not text:
            continue
        
        # Validate and normalize metadata (ensure dict)
        if document.metadata is not None and not isinstance(document.metadata, dict):
            raise ValueError(f"Invalid document.metadata type: {type(document.metadata).__name__}")
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        
        # Normalize source_path to string
        source_path = str(document.source_path) if document.source_path else ""
        
        # Ensure parser_name is string
        parser_name = document.parser_name or "unknown"
        
        rows.append({
            "text": text,
            "metadata": metadata,
            "source_path": source_path,
            "parser_name": parser_name,
        })

    if not rows:
        raise NoDataFoundError(f"No data found in parsed input file: {data_path}")
    return pd.DataFrame(rows)

def setup(data_path: Path,
          llm_api_key: str,
          use_chromadb: bool = False,
          chromadb_sqlite_path: str = None,
          chromadb_collection_name: str = None,
          embedding_provider: str = "cohere",
          embedding_model_name: Optional[str] = None,
          embedding_api_key: Optional[str] = None,
          embedding_base_url: Optional[str] = None,
          llm_provider: str = "cohere",
          llm_model_name: Optional[str] = None,
          llm_base_url: Optional[str] = None):
    """
    Initializes the RAG search engine from structured or unstructured data.

    Data loading strategy (Slice 2 integration):
    - Structured (CSV, JSON, Parquet): Loaded directly via pandas, bypassing parser.
    - Unstructured (.txt, .pdf, .docx, etc.): Dispatched to parser via get_parser() contract.

    Args:
        data_path (Path): The path to the data file (structured or unstructured).
        llm_api_key (str): The API key for the Cohere client.
        use_chromadb (bool): Whether to use ChromaDB instead of FAISS (default: False).
        chromadb_sqlite_path (str): Path to ChromaDB SQLite database (required if use_chromadb=True).
        chromadb_collection_name (str): ChromaDB collection name (required if use_chromadb=True).
        embedding_provider (str): Embedding provider identifier (default: "cohere").
        embedding_model_name (str): Optional provider-specific model name.
        embedding_api_key (str): Optional API key for embedding provider; defaults to llm_api_key.
        embedding_base_url (str): Optional base URL for provider endpoints (for OpenAI-compatible or Ollama hosts).
        llm_provider (str): LLM provider identifier (default: "cohere").
        llm_model_name (str): Optional provider-specific chat model name.
        llm_base_url (str): Optional base URL for provider endpoints (for OpenAI-compatible or Ollama hosts).
    Returns:
        RagSearchEngine: The initialized RAG search engine.
    Raises:
        TypeError: If data_path is not a Path object.
        FileNotFoundError: If the data path does not exist.
        NoDataFoundError: If no data found:
            - Structured: Empty file
            - Unstructured: Parser returns no documents or all content is empty
        ValueError: If CSV/JSON/Parquet file format is invalid (structured path only).
        RagSearchError: If unstructured parser fails (UnsupportedFileTypeError, ParsingError, etc.).
        RuntimeError: For other data loading, Cohere client, or vector database errors.
    """
    print("Starting setup of the RAG Search Engine...")

    # Validate data_path type
    if not isinstance(data_path, Path):
        raise TypeError(f"data_path must be Path object, got {type(data_path).__name__}")
    
    # Validate data path exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    ingestion_diagnostics = {
        "source_path": str(data_path),
        "selected_parser": "",
        "status": "success",
        "failure_reason": "",
    }

    # Load data with specific error handling
    try:
        if data_path.suffix in STRUCTURED_EXTENSIONS:
            ingestion_diagnostics["selected_parser"] = "structured/pandas"
            data = _load_structured_data(data_path)
        else:
            data = _load_unstructured_data(data_path, diagnostics=ingestion_diagnostics)
    except RagSearchError:
        # Preserve RagSearchError from parser (NoDataFoundError, ParsingError, etc.)
        raise
    except ValueError as e:
        # ValueError from _load_structured_data or invalid document structure
        raise ValueError(f"Failed to load data: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}") from e

    if data.empty:
        raise NoDataFoundError(f"No data found in input file: {data_path}")

    # Get file name for logging/engine initialization
    file_name = data_path.name

    try:
        llm_provider_name = _validate_provider(llm_provider, SUPPORTED_LLM_PROVIDERS, "LLM")
    except ValueError as e:
        raise RuntimeError(f"Failed to initialize LLM client: {e}") from e

    try:
        embedding_provider_name = _validate_provider(embedding_provider, SUPPORTED_EMBEDDING_PROVIDERS, "embedding")
    except ValueError as e:
        raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

    cohere_client = None
    if llm_provider_name == "cohere" or embedding_provider_name == "cohere":
        try:
            cohere_client = CohereClient(api_key=llm_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cohere client: {e}")

    try:
        llm_client = create_llm_client(
            provider=llm_provider_name,
            api_key=llm_api_key,
            model=llm_model_name,
            base_url=llm_base_url,
            cohere_client=cohere_client,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM client: {e}") from e

    try:
        embedding_model = create_embedding_model(
            provider=embedding_provider_name,
            api_key=embedding_api_key or llm_api_key,
            model=embedding_model_name,
            base_url=embedding_base_url,
            cohere_client=cohere_client,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding model: {e}") from e

    # Connect to vector database or ChromaDB
    if use_chromadb:
        if not chromadb_sqlite_path or not chromadb_collection_name:
            raise ValueError("ChromaDB path and collection name must be provided when use_chromadb is True.")
        engine = RagSearchEngine(
            data=data,
            embedding_model=embedding_model,
            llm_client=llm_client,
            vector_db=None,
            file_name=file_name,
            chromadb_sqlite_path=chromadb_sqlite_path,
            chromadb_collection_name=chromadb_collection_name
        )
    else:
        try:
            embedding_dim = infer_embedding_dimension(embedding_model)
        except Exception as exc:
            # Preserve legacy fallback behavior when probe-time inference fails.
            logger.warning("Falling back to legacy embedding dimension 4096: %s", exc)
            embedding_dim = 4096
        try:
            vector_db = build_vector_backend(embedding_dim=embedding_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to vector database: {e}")
        engine = RagSearchEngine(
            data=data,
            embedding_model=embedding_model,
            llm_client=llm_client,
            vector_db=vector_db,
            file_name=file_name
        )

    print("Setup complete.")
    engine.ingestion_diagnostics = ingestion_diagnostics
    return engine
