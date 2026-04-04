"""Tests for embedding provider adapters and factory behavior."""

from types import SimpleNamespace

import pytest

from libs.ragsearch.embedding_models import (
    OllamaEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    SentenceTransformersEmbeddingAdapter,
    create_embedding_model,
    extract_embeddings,
)


class _OpenAIEmbeddingsClient:
    def create(self, model, input):
        assert model == "text-embedding-3-small"
        assert input == ["a", "b"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )


class _OpenAIClient:
    def __init__(self):
        self.embeddings = _OpenAIEmbeddingsClient()


class _SentenceTransformerModel:
    def encode(self, texts):
        assert texts == ["a", "b"]
        return [[0.1, 0.2], [0.3, 0.4]]


class _OllamaClient:
    def embed(self, model, input):
        assert model == "nomic-embed-text"
        assert input == ["a", "b"]
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}


class _CohereLikeClient:
    def embed(self, texts):
        assert texts == ["probe"]
        return SimpleNamespace(embeddings=[[0.9, 0.8]])


def test_openai_adapter_normalizes_response_shape():
    adapter = OpenAIEmbeddingAdapter(client=_OpenAIClient(), model="text-embedding-3-small")

    response = adapter.embed(["a", "b"])

    assert extract_embeddings(response) == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_sentence_transformers_adapter_normalizes_response_shape():
    adapter = SentenceTransformersEmbeddingAdapter(model=_SentenceTransformerModel())

    response = adapter.embed(["a", "b"])

    assert extract_embeddings(response) == [[0.1, 0.2], [0.3, 0.4]]


def test_ollama_adapter_normalizes_response_shape():
    adapter = OllamaEmbeddingAdapter(client=_OllamaClient(), model="nomic-embed-text")

    response = adapter.embed(["a", "b"])

    assert extract_embeddings(response) == [[0.1, 0.2], [0.3, 0.4]]


def test_create_embedding_model_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedding_model(provider="unknown")


def test_create_embedding_model_requires_openai_api_key():
    with pytest.raises(ValueError, match="requires api_key"):
        create_embedding_model(provider="openai")


def test_create_embedding_model_supports_injected_cohere_client():
    model = create_embedding_model(provider="cohere", cohere_client=_CohereLikeClient())

    response = model.embed(["probe"])

    assert extract_embeddings(response) == [[0.9, 0.8]]
