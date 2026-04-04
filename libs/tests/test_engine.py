"""
Unit tests for citation payload behavior in RagSearchEngine.search.
"""

import pandas as pd

from libs.ragsearch.engine import RagSearchEngine
from libs.ragsearch.chunking import FixedWordChunkingStrategy, RowChunkingStrategy
from libs.ragsearch.reranking import NoOpReranker
from libs.ragsearch.vector_db import VectorDB


class DummyEmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class DummyEmbeddingModel:
    """Deterministic embedding model used for citation MVP tests."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return DummyEmbeddingResponse(vectors)


class DummyLLMClient:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "grounded answer"


class CountingEmbeddingModel:
    def __init__(self):
        self.call_sizes = []

    def embed(self, texts):
        self.call_sizes.append(len(texts))
        vectors = []
        for text in texts:
            seed = float((sum(ord(ch) for ch in str(text)) % 10) + 1)
            vectors.append([seed, seed / 2.0, seed / 3.0, seed / 4.0])
        return DummyEmbeddingResponse(vectors)


class ReverseReranker:
    def rerank(self, query, results):
        return list(reversed(results))


def _make_engine():
    data = pd.DataFrame(
        [
            {
                "text": "Alpha document content",
                "source_path": "/docs/alpha.txt",
                "parser_name": "fallback/plain_text",
            },
            {
                "text": "Beta document content",
                "source_path": "/docs/beta.txt",
                "parser_name": "fallback/plain_text",
            },
        ]
    )
    return RagSearchEngine(
        data=data,
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )


def test_search_returns_citation_fields():
    engine = _make_engine()

    results = engine.search("alpha", top_k=1)

    assert len(results) == 1
    assert "citation" in results[0]

    citation = results[0]["citation"]
    assert citation["record_id"] == 0
    assert citation["source_path"] == "/docs/alpha.txt"
    assert citation["parser_name"] == "fallback/plain_text"
    assert "Alpha document" in citation["excerpt"]


def test_search_result_includes_similarity_and_metadata():
    engine = _make_engine()

    result = engine.search("beta", top_k=1)[0]

    assert "metadata" in result
    assert "similarity" in result
    assert isinstance(result["similarity"], float)
    assert result["metadata"]["source_path"] == "/docs/beta.txt"


def test_search_excerpt_truncates_to_200_chars():
    long_text = "x" * 250
    data = pd.DataFrame(
        [
            {
                "text": long_text,
                "source_path": "/docs/long.txt",
                "parser_name": "fallback/plain_text",
            }
        ]
    )
    engine = RagSearchEngine(
        data=data,
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )

    result = engine.search("alpha", top_k=1)[0]
    assert len(result["citation"]["excerpt"]) == 200


def test_search_handles_missing_text_fields():
    data = pd.DataFrame(
        [
            {
                "title": "metadata only",
                "source_path": None,
                "parser_name": None,
            }
        ]
    )
    engine = RagSearchEngine(
        data=data,
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )

    # Force a payload where both excerpt source fields are absent.
    engine.data = pd.DataFrame(
        [
            {
                "text": None,
                "combined_text": None,
                "source_path": None,
                "parser_name": None,
            }
        ]
    )
    engine.index_data = engine.data.copy()

    result = engine.search("alpha", top_k=1)[0]
    assert result["citation"]["excerpt"] == ""
    assert result["citation"]["source_path"] == ""
    assert result["citation"]["parser_name"] == ""


def test_serialize_query_results_keeps_backward_compatibility():
    enriched = [
        {
            "metadata": {"text": "alpha", "source_path": "/docs/alpha.txt"},
            "citation": {
                "record_id": 0,
                "source_path": "/docs/alpha.txt",
                "parser_name": "fallback/plain_text",
                "excerpt": "alpha",
            },
            "similarity": 0.99,
        }
    ]

    legacy_payload = RagSearchEngine._serialize_query_results(enriched)
    assert legacy_payload == [{"text": "alpha", "source_path": "/docs/alpha.txt"}]

    detailed_payload = RagSearchEngine._serialize_query_results(enriched, include_details=True)
    assert detailed_payload == enriched


def test_search_raises_value_error_for_invalid_embedding_response():
    class BadEmbeddingModel:
        def embed(self, texts):
            return object()

    data = pd.DataFrame(
        [
            {
                "text": "alpha",
                "source_path": "/docs/a.txt",
                "parser_name": "fallback/plain_text",
            }
        ]
    )
    engine = RagSearchEngine(
        data=data,
        embedding_model=BadEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )

    try:
        engine.search("alpha", top_k=1)
        raise AssertionError("Expected ValueError for invalid embedding response")
    except ValueError as exc:
        assert "embeddings" in str(exc).lower()


def test_build_answer_context_formats_retrieved_sources():
    results = [
        {
            "metadata": {"text": "Alpha document content"},
            "citation": {"source_path": "/docs/alpha.txt", "parser_name": "fallback/plain_text", "excerpt": "Alpha document content"},
            "similarity": 0.93,
        }
    ]

    context = RagSearchEngine._build_answer_context(results)

    assert "[1] source_path: /docs/alpha.txt" in context
    assert "parser_name: fallback/plain_text" in context
    assert "similarity: 0.9300" in context
    assert "excerpt: Alpha document content" in context


def test_answer_returns_structured_output_with_preserved_citations():
    engine = _make_engine()

    payload = engine.answer("alpha", top_k=1)

    assert payload["question"] == "alpha"
    assert payload["answer"] == "grounded answer"
    assert len(payload["results"]) == 1
    assert payload["citations"] == [payload["results"][0]["citation"]]
    assert "Question: alpha" in engine.llm_client.prompts[0]
    assert "Sources:" in engine.llm_client.prompts[0]
    assert "[1] source_path: /docs/alpha.txt" in engine.llm_client.prompts[0]


def test_build_answer_prompt_mentions_grounding_rules():
    prompt = RagSearchEngine._build_answer_prompt("What is alpha?", [])

    assert "Answer only from the provided sources." in prompt
    assert "If the sources are insufficient, say you do not know." in prompt
    assert "Question: What is alpha?" in prompt
    assert "(no sources retrieved)" in prompt


def test_incremental_indexing_first_run_marks_all_as_new(tmp_path):
    data = pd.DataFrame(
        [
            {"text": "alpha", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )
    model = CountingEmbeddingModel()

    engine = RagSearchEngine(
        data=data,
        embedding_model=model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="incremental.csv",
    )

    assert model.call_sizes == [2]
    assert engine.indexing_diagnostics["total_records"] == 2
    assert engine.indexing_diagnostics["embedded_records"] == 2
    assert engine.indexing_diagnostics["new_records"] == 2
    assert engine.indexing_diagnostics["changed_records"] == 0
    assert engine.indexing_diagnostics["reused_records"] == 0


def test_incremental_indexing_no_change_rerun_reuses_cached_embeddings(tmp_path):
    data = pd.DataFrame(
        [
            {"text": "alpha", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )

    first_model = CountingEmbeddingModel()
    RagSearchEngine(
        data=data.copy(),
        embedding_model=first_model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="incremental.csv",
    )

    second_model = CountingEmbeddingModel()
    second_engine = RagSearchEngine(
        data=data.copy(),
        embedding_model=second_model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="incremental.csv",
    )

    assert second_model.call_sizes == []
    assert second_engine.indexing_diagnostics["embedded_records"] == 0
    assert second_engine.indexing_diagnostics["new_records"] == 0
    assert second_engine.indexing_diagnostics["changed_records"] == 0
    assert second_engine.indexing_diagnostics["reused_records"] == 2


def test_incremental_indexing_reindexes_only_changed_records(tmp_path):
    baseline = pd.DataFrame(
        [
            {"text": "alpha", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )
    updated = pd.DataFrame(
        [
            {"text": "alpha updated", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )

    first_model = CountingEmbeddingModel()
    RagSearchEngine(
        data=baseline,
        embedding_model=first_model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="incremental.csv",
    )

    second_model = CountingEmbeddingModel()
    second_engine = RagSearchEngine(
        data=updated,
        embedding_model=second_model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="incremental.csv",
    )

    assert second_model.call_sizes == [1]
    assert second_engine.indexing_diagnostics["embedded_records"] == 1
    assert second_engine.indexing_diagnostics["new_records"] == 0
    assert second_engine.indexing_diagnostics["changed_records"] == 1
    assert second_engine.indexing_diagnostics["reused_records"] == 1


def test_default_hook_path_matches_explicit_defaults():
    data = pd.DataFrame(
        [
            {"text": "alpha", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )

    implicit_engine = RagSearchEngine(
        data=data.copy(),
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )
    explicit_engine = RagSearchEngine(
        data=data.copy(),
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
        chunking_strategy=RowChunkingStrategy(),
        reranker=NoOpReranker(),
    )

    implicit = implicit_engine.search("alpha", top_k=2)
    explicit = explicit_engine.search("alpha", top_k=2)
    assert implicit == explicit


def test_chunking_hook_splits_records_deterministically(tmp_path):
    data = pd.DataFrame(
        [
            {"text": "alpha beta gamma"},
        ]
    )
    model = CountingEmbeddingModel()

    engine = RagSearchEngine(
        data=data,
        embedding_model=model,
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir=str(tmp_path / "embeddings"),
        file_name="chunked.csv",
        chunking_strategy=FixedWordChunkingStrategy(words_per_chunk=1),
    )

    assert model.call_sizes == [3]
    assert engine.indexing_diagnostics["total_records"] == 3


def test_search_applies_configured_reranker():
    data = pd.DataFrame(
        [
            {"text": "alpha", "source_path": "/docs/a.txt", "parser_name": "fallback/plain_text"},
            {"text": "beta", "source_path": "/docs/b.txt", "parser_name": "fallback/plain_text"},
        ]
    )

    baseline_engine = RagSearchEngine(
        data=data.copy(),
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
    )
    reranked_engine = RagSearchEngine(
        data=data.copy(),
        embedding_model=DummyEmbeddingModel(),
        llm_client=DummyLLMClient(),
        vector_db=VectorDB(embedding_dim=4),
        save_dir="embeddings/test_engine",
        reranker=ReverseReranker(),
    )

    baseline = baseline_engine.search("alpha", top_k=2)
    reranked = reranked_engine.search("alpha", top_k=2)

    assert len(baseline) == 2
    assert len(reranked) == 2
    assert reranked[0]["citation"]["record_id"] == baseline[1]["citation"]["record_id"]
    assert reranked[1]["citation"]["record_id"] == baseline[0]["citation"]["record_id"]


def test_search_emits_observability_metrics():
    engine = _make_engine()

    engine.search("alpha", top_k=1)

    retrieval_events = [item for item in engine.observability_events if item["stage"] == "retrieval"]
    assert len(retrieval_events) >= 1
    payload = retrieval_events[-1]["payload"]
    assert payload["query"] == "alpha"
    assert payload["top_k"] == 1
    assert payload["results_count"] == 1
    assert payload["latency_ms"] >= 0


def test_answer_emits_generation_observability_metrics():
    engine = _make_engine()

    payload = engine.answer("alpha", top_k=1)

    generation_events = [item for item in engine.observability_events if item["stage"] == "generation"]
    assert len(generation_events) >= 1
    metrics = generation_events[-1]["payload"]
    assert metrics["query"] == "alpha"
    assert metrics["top_k"] == 1
    assert metrics["results_count"] == 1
    assert metrics["citations_count"] == len(payload["citations"])
    assert metrics["latency_ms"] >= 0
