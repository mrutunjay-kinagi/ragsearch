"""
Unit tests for citation payload behavior in RagSearchEngine.search.
"""

import pandas as pd

from libs.ragsearch.engine import RagSearchEngine
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
    pass


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
