"""Tests for chunking strategies (#77)."""

import pytest

from libs.ragsearch.chunking import (
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_WORDS_PER_CHUNK,
    FixedWordChunkingStrategy,
    RowChunkingStrategy,
    default_unstructured_chunking_strategy,
    describe_chunking_strategy,
)


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_fixed_word_chunking_without_overlap_is_unchanged():
    chunks = FixedWordChunkingStrategy(words_per_chunk=4).chunk_text(_words(10))

    assert chunks == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]


def test_fixed_word_chunking_with_overlap_repeats_trailing_words():
    chunks = FixedWordChunkingStrategy(words_per_chunk=4, overlap_words=1).chunk_text(_words(10))

    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_fixed_word_chunking_does_not_emit_a_window_with_no_new_words():
    # 7 words, windows of 4 stepping by 3: [0-3], [3-6]; a third window would only repeat w6.
    chunks = FixedWordChunkingStrategy(words_per_chunk=4, overlap_words=1).chunk_text(_words(7))

    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6"]


def test_fixed_word_chunking_covers_every_word():
    words = _words(1003).split()
    chunks = FixedWordChunkingStrategy(words_per_chunk=250, overlap_words=40).chunk_text(" ".join(words))

    covered = {word for chunk in chunks for word in chunk.split()}
    assert covered == set(words)
    assert all(len(chunk.split()) <= 250 for chunk in chunks)


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_fixed_word_chunking_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap_words"):
        FixedWordChunkingStrategy(words_per_chunk=4, overlap_words=overlap)


def test_fixed_word_chunking_class_defaults_are_backward_compatible():
    strategy = FixedWordChunkingStrategy()

    assert (strategy.words_per_chunk, strategy.overlap_words) == (200, 0)


def test_default_unstructured_chunking_settings():
    strategy = default_unstructured_chunking_strategy()

    assert isinstance(strategy, FixedWordChunkingStrategy)
    assert (strategy.words_per_chunk, strategy.overlap_words) == (DEFAULT_WORDS_PER_CHUNK, DEFAULT_OVERLAP_WORDS)
    assert 0 < DEFAULT_OVERLAP_WORDS < DEFAULT_WORDS_PER_CHUNK


def test_describe_chunking_strategy_includes_settings():
    assert describe_chunking_strategy(RowChunkingStrategy()) == "RowChunkingStrategy"
    assert (
        describe_chunking_strategy(FixedWordChunkingStrategy(words_per_chunk=120, overlap_words=20))
        == "FixedWordChunkingStrategy(overlap_words=20, words_per_chunk=120)"
    )


def test_describe_chunking_strategy_for_custom_strategies():
    class Sentences:
        def __init__(self):
            self.max_sentences = 3
            self._cache = {}

        def chunk_text(self, text):
            return [text]

    assert describe_chunking_strategy(Sentences()) == "Sentences(max_sentences=3)"
