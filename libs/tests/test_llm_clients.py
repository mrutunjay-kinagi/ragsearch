"""Tests for LLM provider adapters and factory behavior."""

from types import SimpleNamespace

import pytest

from libs.ragsearch.llm_clients import (
    CohereLLMClientAdapter,
    OllamaLLMClientAdapter,
    OpenAILLMClientAdapter,
    create_llm_client,
)


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
