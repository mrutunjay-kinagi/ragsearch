"""Tests for LLM provider adapters and factory behavior."""

from types import SimpleNamespace

import pytest

from libs.ragsearch.llm_clients import (
    CohereLLMClientAdapter,
    OllamaLLMClientAdapter,
    OpenAILLMClientAdapter,
    DEFAULT_COHERE_CHAT_MODEL,
    create_llm_client,
)
from libs.ragsearch.errors import ModelNotFoundError


class _CohereClient:
    def chat(self, message, **kwargs):
        assert message == "hello"
        return SimpleNamespace(text="cohere answer")


class _OpenAIChatCompletions:
    def create(self, model, messages, **kwargs):
        assert model == "gpt-4o-mini"
        assert messages == [{"role": "user", "content": "hello"}]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai answer"))]
        )


class _OpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_OpenAIChatCompletions())


class _OllamaClient:
    def chat(self, model, messages, **kwargs):
        assert model == "llama3.1"
        assert messages == [{"role": "user", "content": "hello"}]
        return {"message": {"content": "ollama answer"}}


def test_cohere_adapter_returns_text():
    adapter = CohereLLMClientAdapter(_CohereClient())

    assert adapter.generate("hello") == "cohere answer"


def test_openai_adapter_returns_text():
    adapter = OpenAILLMClientAdapter(client=_OpenAIClient())

    assert adapter.generate("hello") == "openai answer"


def test_ollama_adapter_returns_text():
    adapter = OllamaLLMClientAdapter(client=_OllamaClient())

    assert adapter.generate("hello") == "ollama answer"


def test_create_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(provider="unknown")


def test_create_llm_client_requires_openai_api_key():
    with pytest.raises(ValueError, match="requires api_key"):
        create_llm_client(provider="openai")


def test_create_llm_client_supports_injected_cohere_client():
    model = create_llm_client(provider="cohere", cohere_client=_CohereClient())

    assert model.generate("hello") == "cohere answer"


class _RecordingCohereChatClient:
    def __init__(self):
        self.calls = []

    def chat(self, message, **kwargs):
        self.calls.append({"message": message, **kwargs})
        return SimpleNamespace(text="ok")


class _CohereNotFoundError(Exception):
    """Mimics cohere.errors.NotFoundError (an ApiError with status_code and body)."""

    def __init__(self, message):
        super().__init__(f"status_code: 404, body: {{'message': {message!r}}}")
        self.status_code = 404
        self.body = {"message": message}


def test_cohere_adapter_sends_default_model():
    client = _RecordingCohereChatClient()

    create_llm_client(provider="cohere", cohere_client=client).generate("hello")

    assert DEFAULT_COHERE_CHAT_MODEL == "command-a-03-2025"
    assert client.calls == [{"message": "hello", "model": "command-a-03-2025"}]


def test_create_llm_client_passes_cohere_model_name_through():
    client = _RecordingCohereChatClient()

    create_llm_client(provider="cohere", model="command-a-plus-05-2026", cohere_client=client).generate("hello")

    assert client.calls[0]["model"] == "command-a-plus-05-2026"


def test_cohere_adapter_names_parameter_when_model_removed():
    class _RemovedModelClient:
        def chat(self, message, **kwargs):
            raise _CohereNotFoundError("model 'command-r' was removed on September 15, 2025.")

    adapter = CohereLLMClientAdapter(_RemovedModelClient(), model="command-r")

    with pytest.raises(ModelNotFoundError, match="llm_model_name") as excinfo:
        adapter.generate("hello")

    assert "command-r" in str(excinfo.value)
