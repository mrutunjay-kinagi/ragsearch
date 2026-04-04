"""Vector backend abstraction and factory helpers for retrieval storage."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Contract for vector storage/query backends consumed by the engine."""

    def insert(self, embedding: list, metadata: dict) -> None:
        ...

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        ...