"""Reranking interfaces and defaults for retrieval quality hooks."""

from typing import Protocol


class Reranker(Protocol):
    """Post-processes retrieval results before returning to callers."""

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Return reordered retrieval results for a query."""


class NoOpReranker:
    """Default reranker that preserves legacy retrieval ordering."""

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        return results
