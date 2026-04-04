"""Embedding model protocols and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Minimal contract required by retrieval/indexing paths."""

    def embed(self, texts: Sequence[str]) -> Any:
        ...


@dataclass
class CohereEmbeddingAdapter:
    """Adapter that presents a stable embedding contract over Cohere-like clients."""

    client: Any

    def embed(self, texts: Sequence[str]) -> Any:
        return self.client.embed(texts=list(texts))


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


def infer_embedding_dimension(embedding_model: EmbeddingModel, probe_text: str = "dimension probe") -> int:
    """Infer embedding dimension from a single deterministic probe embedding."""
    embeddings = extract_embeddings(embedding_model.embed(texts=[probe_text]))
    return len(embeddings[0])
