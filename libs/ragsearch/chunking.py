"""Chunking strategy interfaces and defaults for retrieval quality hooks."""

from typing import Protocol


class ChunkingStrategy(Protocol):
    """Defines how a text record is split into retrieval chunks."""

    def chunk_text(self, text: str) -> list[str]:
        """Return a deterministic list of text chunks for a record."""


class RowChunkingStrategy:
    """Default strategy that preserves legacy row-level indexing behavior."""

    def chunk_text(self, text: str) -> list[str]:
        return [text]


class FixedWordChunkingStrategy:
    """Split text into deterministic chunks with a fixed number of words."""

    def __init__(self, words_per_chunk: int = 200):
        if words_per_chunk <= 0:
            raise ValueError("words_per_chunk must be a positive integer")
        self.words_per_chunk = words_per_chunk

    def chunk_text(self, text: str) -> list[str]:
        words = str(text).split()
        if not words:
            return []

        chunks = []
        for start in range(0, len(words), self.words_per_chunk):
            chunk_words = words[start:start + self.words_per_chunk]
            chunks.append(" ".join(chunk_words))
        return chunks
