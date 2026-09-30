"""Embedding model protocols and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, runtime_checkable

from .errors import ModelNotFoundError, is_model_not_found_error, provider_error_message


# Cohere model IDs: https://docs.cohere.com/docs/models
DEFAULT_COHERE_EMBEDDING_MODEL = "embed-v4.0"
# v3+ Cohere embed models require input_type: https://docs.cohere.com/v1/reference/embed
DEFAULT_COHERE_EMBEDDING_INPUT_TYPE = "search_document"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal contract required by retrieval/indexing paths."""

    def embed(self, texts: Sequence[str]) -> Any:
        ...


@dataclass
class EmbeddingResponse:
    """Normalized embedding response payload used by adapters."""

    embeddings: List[List[float]]


@dataclass
class CohereEmbeddingAdapter:
    """Adapter that presents a stable embedding contract over Cohere-like clients."""

    client: Any
    model: str = DEFAULT_COHERE_EMBEDDING_MODEL
    input_type: str = DEFAULT_COHERE_EMBEDDING_INPUT_TYPE

    def embed(self, texts: Sequence[str]) -> Any:
        try:
            return self.client.embed(texts=list(texts), model=self.model, input_type=self.input_type)
        except Exception as exc:
            if is_model_not_found_error(exc):
                raise ModelNotFoundError(
                    f"Cohere embedding model '{self.model}' is not available: {provider_error_message(exc)} "
                    f"Set embedding_model_name to a current Cohere embedding model "
                    f"(default: '{DEFAULT_COHERE_EMBEDDING_MODEL}'); see https://docs.cohere.com/docs/models.",
                    cause=exc,
                ) from exc
            raise


@dataclass
class OpenAIEmbeddingAdapter:
    """Adapter for OpenAI embedding clients."""

    client: Any
    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        raw_vectors = []
        for item in getattr(response, "data", []):
            if isinstance(item, dict):
                raw_vectors.append(item.get("embedding"))
            else:
                raw_vectors.append(getattr(item, "embedding", None))
        return EmbeddingResponse(embeddings=extract_embeddings(EmbeddingResponse(embeddings=raw_vectors)))


@dataclass
class SentenceTransformersEmbeddingAdapter:
    """Adapter for sentence-transformers models."""

    model: Any
    model_name: str | None = None

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        encoded = self.model.encode(list(texts))
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if isinstance(encoded, list) and encoded and not isinstance(encoded[0], list):
            encoded = [encoded]
        return EmbeddingResponse(embeddings=extract_embeddings(EmbeddingResponse(embeddings=encoded)))


def _response_field(payload: Any, name: str) -> Any:
    """Read ``name`` from a dict response or a typed (attribute-based) response object.

    The ollama client returns typed pydantic responses (e.g. ``ollama.EmbedResponse``), which are
    not dicts; older clients and raw HTTP payloads are plain dicts (#129).
    """
    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


@dataclass
class OllamaEmbeddingAdapter:
    """Adapter for Ollama embedding clients."""

    client: Any
    model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL

    def embed(self, texts: Sequence[str]) -> EmbeddingResponse:
        if hasattr(self.client, "embed"):
            payload = self.client.embed(model=self.model, input=list(texts))
            raw_vectors = _response_field(payload, "embeddings")
            return EmbeddingResponse(embeddings=extract_embeddings(EmbeddingResponse(embeddings=raw_vectors)))

        if hasattr(self.client, "embeddings"):
            vectors: List[List[float]] = []
            for text in texts:
                payload = self.client.embeddings(model=self.model, prompt=text)
                vectors.append(_response_field(payload, "embedding"))
            return EmbeddingResponse(embeddings=extract_embeddings(EmbeddingResponse(embeddings=vectors)))

        raise ValueError("Ollama client must provide either 'embed' or 'embeddings'.")


def create_embedding_model(
    provider: str = "cohere",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    cohere_client: Any | None = None,
) -> EmbeddingModel:
    """Build an embedding model adapter from provider configuration."""

    normalized_provider = provider.strip().lower().replace("-", "_")

    if normalized_provider == "cohere":
        client = cohere_client
        if client is None:
            if not api_key:
                raise ValueError("Cohere embedding provider requires api_key.")
            try:
                from cohere import Client as CohereClient
            except ImportError as exc:
                raise RuntimeError("Cohere SDK is not installed. Install package 'cohere'.") from exc
            client = CohereClient(api_key=api_key)
        return CohereEmbeddingAdapter(client, model=model or DEFAULT_COHERE_EMBEDDING_MODEL)

    if normalized_provider in {"sentence_transformers", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install package 'sentence-transformers'."
            ) from exc
        model_name = model or DEFAULT_SENTENCE_TRANSFORMERS_MODEL
        return SentenceTransformersEmbeddingAdapter(SentenceTransformer(model_name), model_name=model_name)

    if normalized_provider == "openai":
        if not api_key:
            raise ValueError("OpenAI embedding provider requires api_key.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Install package 'openai'.") from exc
        return OpenAIEmbeddingAdapter(
            client=OpenAI(api_key=api_key, base_url=base_url),
            model=model or DEFAULT_OPENAI_EMBEDDING_MODEL,
        )

    if normalized_provider == "ollama":
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Ollama SDK is not installed. Install package 'ollama'.") from exc
        client = ollama.Client(host=base_url) if base_url else ollama.Client()
        return OllamaEmbeddingAdapter(client=client, model=model or DEFAULT_OLLAMA_EMBEDDING_MODEL)

    raise ValueError(
        "Unsupported embedding provider: "
        f"{provider}. Supported providers: cohere, sentence_transformers, openai, ollama."
    )


def extract_embeddings(response: Any) -> List[List[float]]:
    """Normalize embedding responses to a list-of-lists float shape."""
    if not hasattr(response, "embeddings"):
        raise ValueError("Embedding response must contain an 'embeddings' attribute.")

    embeddings = getattr(response, "embeddings")
    if not isinstance(embeddings, (list, tuple)) or not embeddings:
        raise ValueError("Embedding response must provide a non-empty embeddings sequence.")

    normalized: List[List[float]] = []
    for embedding in embeddings:
        if not isinstance(embedding, (list, tuple)) or not embedding:
            raise ValueError("Each embedding must be a non-empty numeric sequence.")
        try:
            normalized.append([float(value) for value in embedding])
        except (TypeError, ValueError) as exc:
            raise ValueError("Each embedding must be a numeric sequence.") from exc

    return normalized


def describe_embedding_model(embedding_model: Any) -> str:
    """Return a stable identity for an embedding model, used to validate cached embeddings.

    The identity is the class name plus the model name, taken from a string
    ``model_name`` or ``model`` attribute when the object has one
    (for example ``"OpenAIEmbeddingAdapter:text-embedding-3-small"``).
    """
    identity = type(embedding_model).__name__
    for attribute in ("model_name", "model"):
        name = getattr(embedding_model, attribute, None)
        if isinstance(name, str) and name.strip():
            return f"{identity}:{name.strip()}"
    return identity


def infer_embedding_dimension(embedding_model: EmbeddingModel, probe_text: str = "dimension probe") -> int:
    """Infer embedding dimension from a single deterministic probe embedding."""
    embeddings = extract_embeddings(embedding_model.embed(texts=[probe_text]))
    return len(embeddings[0])
