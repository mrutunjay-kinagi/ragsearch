"""Chunking strategy interfaces and defaults for retrieval quality hooks."""

from typing import Any, Protocol

# Default chunking for unstructured files (PDF, DOCX, HTML, text) used by setup(). Chosen from a
# retrieval experiment on an 80-page PDF (14 questions, pages 5-80; see ADR-0010): 150 words with
# 20% overlap gave the best hit@1/hit@3 balance and found every late-document answer in the top 5,
# while sending ~1,500 estimated tokens for top_k=5. Larger chunks lowered hit@1 and grew the context.
DEFAULT_WORDS_PER_CHUNK = 150
DEFAULT_OVERLAP_WORDS = 30


class ChunkingStrategy(Protocol):
    """Defines how a text record is split into retrieval chunks."""

    def chunk_text(self, text: str) -> list[str]:
        """Return a deterministic list of text chunks for a record."""


class RowChunkingStrategy:
    """Default strategy that preserves legacy row-level indexing behavior."""

    def chunk_text(self, text: str) -> list[str]:
        return [text]


class FixedWordChunkingStrategy:
    """Split text into deterministic word windows, optionally overlapping.

    Each chunk holds up to ``words_per_chunk`` words; consecutive chunks share
    ``overlap_words`` words so a passage cut at a chunk boundary still appears whole in one chunk.
    """

    def __init__(self, words_per_chunk: int = 200, overlap_words: int = 0):
        if words_per_chunk <= 0:
            raise ValueError("words_per_chunk must be a positive integer")
        if overlap_words < 0 or overlap_words >= words_per_chunk:
            raise ValueError("overlap_words must be >= 0 and smaller than words_per_chunk")
        self.words_per_chunk = words_per_chunk
        self.overlap_words = overlap_words

    def chunk_text(self, text: str) -> list[str]:
        words = str(text).split()
        if not words:
            return []

        step = self.words_per_chunk - self.overlap_words
        chunks = []
        for start in range(0, len(words), step):
            chunks.append(" ".join(words[start:start + self.words_per_chunk]))
            if start + self.words_per_chunk >= len(words):
                break
        return chunks


def default_unstructured_chunking_strategy() -> FixedWordChunkingStrategy:
    """Chunking that setup() applies to unstructured files when no strategy is given."""
    return FixedWordChunkingStrategy(words_per_chunk=DEFAULT_WORDS_PER_CHUNK, overlap_words=DEFAULT_OVERLAP_WORDS)


def describe_chunking_strategy(strategy: Any) -> str:
    """Return a stable identity for a chunking strategy and its settings, used in the embedding cache.

    The identity is the class name plus the strategy's public scalar attributes, e.g.
    ``FixedWordChunkingStrategy(overlap_words=50, words_per_chunk=250)``.
    """
    attributes = vars(strategy) if hasattr(strategy, "__dict__") else {}
    settings = {
        name: value
        for name, value in sorted(attributes.items())
        if not name.startswith("_") and isinstance(value, (str, int, float, bool))
    }
    name = type(strategy).__name__
    if not settings:
        return name
    return f"{name}({', '.join(f'{key}={value}' for key, value in settings.items())})"
