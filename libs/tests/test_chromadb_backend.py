"""
Reproduction tests for ChromaDB mode through the public setup()/search()/answer() path (#76).

These are strict xfails: they document the current failure and will flip to
XPASS (failing the suite) once ChromaDB is routed through the VectorBackend
protocol, at which point the xfail markers should be removed.
"""

from pathlib import Path

import pytest

from libs.ragsearch.setup import setup


class DummyEmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class DummyCohereClient:
    """Deterministic stand-in for the Cohere client used by the default providers."""

    def __init__(self, *args, **kwargs):
        self.embedded_texts = []
        self.prompts = []

    def embed(self, texts, **kwargs):
        self.embedded_texts.extend(texts)
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if "chicken" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "salad" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0, 0.0])
        return DummyEmbeddingResponse(vectors)

    def chat(self, message, **kwargs):
        self.prompts.append(message)

        class Response:
            text = "grounded answer [1]"

        return Response()


@pytest.fixture
def chroma_engine(tmp_path, monkeypatch):
    data_path = tmp_path / "recipes.csv"
    data_path.write_text(
        "name,description\n"
        "Roast chicken,Chicken roasted with lemon and thyme\n"
        "Green salad,Salad of lettuce and cucumber\n",
        encoding="utf-8",
    )
    client = DummyCohereClient()
    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", lambda *args, **kwargs: client)

    engine = setup(
        Path(data_path),
        llm_api_key="test-key",
        use_chromadb=True,
        chromadb_sqlite_path=str(tmp_path / "chroma"),
        chromadb_collection_name="recipes",
        embeddings_dir=str(tmp_path / "embeddings"),
    )
    return engine, client


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="#76: ChromaDB mode passes vector_db=None, so search() fails",
)
def test_chromadb_mode_answer_end_to_end(chroma_engine):
    engine, client = chroma_engine

    payload = engine.answer("chicken", top_k=1)

    assert payload["answer"] == "grounded answer [1]"
    assert len(payload["results"]) == 1
    assert "Roast chicken" in payload["context"]
    assert "Roast chicken" in client.prompts[0]


@pytest.mark.xfail(strict=True, reason="#76: ChromaDB mode skips indexing entirely (no embeddings generated)")
def test_chromadb_mode_indexes_with_configured_embedding_model(chroma_engine):
    _, client = chroma_engine

    assert any("Roast chicken" in text for text in client.embedded_texts)
