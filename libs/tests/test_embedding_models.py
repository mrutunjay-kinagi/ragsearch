"""Tests for embedding provider adapters and factory behavior."""

from types import SimpleNamespace

import pytest

from libs.ragsearch.errors import ModelNotFoundError
from libs.ragsearch.embedding_models import (
    DEFAULT_COHERE_EMBEDDING_INPUT_TYPE,
    DEFAULT_COHERE_EMBEDDING_MODEL,
    CohereEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    SentenceTransformersEmbeddingAdapter,
    create_embedding_model,
    describe_embedding_model,
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
    def embed(self, texts, **kwargs):
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


class _RecordingCohereClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return SimpleNamespace(embeddings=[[0.1, 0.2]])


class _CohereNotFoundError(Exception):
    """Mimics cohere.errors.NotFoundError (an ApiError with status_code and body)."""

    def __init__(self, message):
        super().__init__(f"status_code: 404, body: {{'message': {message!r}}}")
        self.status_code = 404
        self.body = {"message": message}


class _MissingModelCohereClient:
    def embed(self, texts, **kwargs):
        raise _CohereNotFoundError(f"model '{kwargs.get('model')}' not found")


def test_cohere_embedding_adapter_sends_default_model_and_input_type():
    client = _RecordingCohereClient()

    create_embedding_model(provider="cohere", cohere_client=client).embed(["probe"])

    assert DEFAULT_COHERE_EMBEDDING_MODEL == "embed-v4.0"
    assert DEFAULT_COHERE_EMBEDDING_INPUT_TYPE == "search_document"
    assert client.calls == [
        {"texts": ["probe"], "model": "embed-v4.0", "input_type": "search_document"}
    ]


def test_create_embedding_model_passes_cohere_model_name_through():
    client = _RecordingCohereClient()

    create_embedding_model(
        provider="cohere", model="embed-multilingual-v3.0", cohere_client=client
    ).embed(["probe"])

    assert client.calls[0]["model"] == "embed-multilingual-v3.0"


def test_cohere_embedding_adapter_names_parameter_when_model_not_found():
    adapter = CohereEmbeddingAdapter(_MissingModelCohereClient(), model="embed-nope")

    with pytest.raises(ModelNotFoundError, match="embedding_model_name") as excinfo:
        adapter.embed(["probe"])

    assert "embed-nope" in str(excinfo.value)
    assert isinstance(excinfo.value.cause, _CohereNotFoundError)


def test_cohere_embedding_adapter_reraises_other_errors_unchanged():
    class _FailingClient:
        def embed(self, texts, **kwargs):
            raise RuntimeError("provider temporarily unavailable")

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        CohereEmbeddingAdapter(_FailingClient()).embed(["probe"])


def test_describe_embedding_model_includes_provider_model_name():
    assert describe_embedding_model(
        create_embedding_model(provider="cohere", cohere_client=_RecordingCohereClient())
    ) == "CohereEmbeddingAdapter:embed-v4.0"
    assert describe_embedding_model(
        OpenAIEmbeddingAdapter(client=_OpenAIClient(), model="text-embedding-3-large")
    ) == "OpenAIEmbeddingAdapter:text-embedding-3-large"
    assert describe_embedding_model(
        SentenceTransformersEmbeddingAdapter(model=object(), model_name="all-MiniLM-L6-v2")
    ) == "SentenceTransformersEmbeddingAdapter:all-MiniLM-L6-v2"


def test_describe_embedding_model_falls_back_to_class_name_for_custom_models():
    class CustomEmbedder:
        def embed(self, texts):
            return SimpleNamespace(embeddings=[[1.0]])

    assert describe_embedding_model(CustomEmbedder()) == "CustomEmbedder"
