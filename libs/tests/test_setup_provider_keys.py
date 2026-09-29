"""
Tests for API key handling across LLM and embedding providers in setup().

An embedding provider must only receive llm_api_key when it is the same provider
as the LLM, so one provider's key is never sent to another.
"""

import importlib
from pathlib import Path

import pytest

from libs.ragsearch.setup import setup

# Patch the module object: ``libs.ragsearch.setup`` as an attribute path is the setup() function (#91).
ragsearch_setup_module = importlib.import_module("libs.ragsearch.setup")


class RecordingCohereClient:
    """Stand-in for cohere.Client that records the key each instance was built with."""

    instances = []

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.embed_calls = 0
        self.chat_calls = 0
        RecordingCohereClient.instances.append(self)

    def embed(self, texts, **kwargs):
        self.embed_calls += 1

        class Resp:
            embeddings = [[0.1, 0.2, 0.3] for _ in texts]

        return Resp()

    def chat(self, message, **kwargs):
        self.chat_calls += 1

        class Resp:
            text = "answer"

        return Resp()


class RecordingLLM:
    def generate(self, prompt, **kwargs):
        return "answer"


@pytest.fixture
def llm_keys(monkeypatch):
    """Record the (provider, api_key) of every non-Cohere LLM client setup() builds."""
    RecordingCohereClient.instances = []
    keys = []
    original_create_llm_client = ragsearch_setup_module.create_llm_client

    def fake_create_llm_client(provider, **kwargs):
        if provider == "cohere":
            return original_create_llm_client(provider=provider, **kwargs)
        keys.append((provider, kwargs.get("api_key")))
        return RecordingLLM()

    monkeypatch.setattr(ragsearch_setup_module, "CohereClient", RecordingCohereClient)
    monkeypatch.setattr(ragsearch_setup_module, "create_llm_client", fake_create_llm_client)
    return keys


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("name,description\na,b\n", encoding="utf-8")
    return Path(path)


def test_mixed_providers_send_each_provider_its_own_key(data_path, tmp_path, llm_keys):
    engine = setup(
        data_path,
        llm_api_key="openai-key",
        llm_provider="openai",
        embedding_provider="cohere",
        embedding_api_key="cohere-key",
        embeddings_dir=str(tmp_path / "cache"),
    )
    engine.answer("a", top_k=1)

    assert llm_keys == [("openai", "openai-key")]
    assert [client.api_key for client in RecordingCohereClient.instances] == ["cohere-key"]
    assert RecordingCohereClient.instances[0].embed_calls > 0


def test_mixed_providers_without_embedding_key_fail_before_any_key_is_sent(data_path, llm_keys):
    with pytest.raises(ValueError, match="embedding_api_key"):
        setup(data_path, llm_api_key="openai-key", llm_provider="openai", embedding_provider="cohere")

    assert RecordingCohereClient.instances == []
    assert llm_keys == []


def test_keyless_llm_provider_still_lends_its_key_to_embeddings_with_deprecation_warning(data_path, tmp_path, llm_keys):
    with pytest.warns(DeprecationWarning, match="embedding_api_key"):
        setup(
            data_path,
            llm_api_key="cohere-key",
            llm_provider="ollama",
            embedding_provider="cohere",
            embeddings_dir=str(tmp_path / "cache"),
        )

    assert [client.api_key for client in RecordingCohereClient.instances] == ["cohere-key"]


def test_same_provider_defaults_embedding_key_to_llm_key(data_path, tmp_path, llm_keys):
    engine = setup(data_path, llm_api_key="cohere-key", embeddings_dir=str(tmp_path / "cache"))
    engine.answer("a", top_k=1)

    assert [client.api_key for client in RecordingCohereClient.instances] == ["cohere-key"]
    client = RecordingCohereClient.instances[0]
    assert client.embed_calls > 0 and client.chat_calls == 1


def test_cohere_embeddings_honour_a_separate_embedding_key(data_path, tmp_path, llm_keys):
    engine = setup(
        data_path,
        llm_api_key="cohere-chat-key",
        embedding_api_key="cohere-embed-key",
        embeddings_dir=str(tmp_path / "cache"),
    )
    engine.answer("a", top_k=1)

    clients = {client.api_key: client for client in RecordingCohereClient.instances}
    assert set(clients) == {"cohere-chat-key", "cohere-embed-key"}
    assert clients["cohere-embed-key"].embed_calls > 0 and clients["cohere-embed-key"].chat_calls == 0
    assert clients["cohere-chat-key"].chat_calls == 1 and clients["cohere-chat-key"].embed_calls == 0


def test_keyless_embedding_provider_needs_no_embedding_key(data_path, tmp_path, llm_keys, monkeypatch):
    captured = {}

    def fake_create_embedding_model(provider, **kwargs):
        captured["provider"] = provider
        captured["api_key"] = kwargs.get("api_key")
        return RecordingCohereClient()  # any object with a compatible embed()

    monkeypatch.setattr(ragsearch_setup_module, "create_embedding_model", fake_create_embedding_model)

    setup(
        data_path,
        llm_api_key="openai-key",
        llm_provider="openai",
        embedding_provider="sentence_transformers",
        embeddings_dir=str(tmp_path / "cache"),
    )

    assert captured == {"provider": "sentence_transformers", "api_key": None}
