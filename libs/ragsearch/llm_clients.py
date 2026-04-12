"""LLM client abstraction boundary and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_CHAT_MODEL = "llama3.1"


@runtime_checkable
class LLMClient(Protocol):
    """Baseline generation contract used by the engine boundary."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class CohereLLMClientAdapter:
    """Adapter exposing a stable generation surface on top of Cohere client."""

    def __init__(self, client: Any):
        self._client = client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.chat(message=prompt, **kwargs)
        return str(getattr(response, "text", ""))


def _normalize_provider_name(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def _extract_text_from_choice(choice: Any) -> str:
    if isinstance(choice, dict):
        message = choice.get("message") or choice.get("delta") or {}
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", ""))

    message = getattr(choice, "message", None) or getattr(choice, "delta", None)
    if message is not None:
        return str(getattr(message, "content", ""))
    return str(getattr(choice, "text", ""))


def _extract_openai_text(response: Any) -> str:
    if isinstance(response, dict):
        if response.get("output_text"):
            return str(response.get("output_text", ""))
        choices = response.get("choices") or []
    else:
        if getattr(response, "output_text", None):
            return str(getattr(response, "output_text", ""))
        choices = getattr(response, "choices", [])

    if not choices:
        return ""
    return _extract_text_from_choice(choices[0])


def _extract_ollama_text(response: Any) -> str:
    if isinstance(response, dict):
        if isinstance(response.get("message"), dict):
            return str(response["message"].get("content", ""))
        return str(response.get("response", ""))

    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", ""))
    return str(getattr(response, "response", ""))


@dataclass
class OpenAILLMClientAdapter:
    """Adapter for OpenAI chat/completions style clients."""

    client: Any
    model: str = DEFAULT_OPENAI_CHAT_MODEL

    def generate(self, prompt: str, **kwargs: Any) -> str:
        chat = getattr(self.client, "chat", None)
        if chat is not None and hasattr(chat, "completions"):
            response = chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return _extract_openai_text(response)

        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                **kwargs,
            )
            return _extract_openai_text(response)

        raise ValueError("OpenAI client must provide chat.completions.create or responses.create.")


@dataclass
class OllamaLLMClientAdapter:
    """Adapter for Ollama generation clients."""

    client: Any
    model: str = DEFAULT_OLLAMA_CHAT_MODEL

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if hasattr(self.client, "chat"):
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return _extract_ollama_text(response)

        if hasattr(self.client, "generate"):
            response = self.client.generate(model=self.model, prompt=prompt, **kwargs)
            return _extract_ollama_text(response)

        raise ValueError("Ollama client must provide chat() or generate().")


def create_llm_client(
    provider: str = "cohere",
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    cohere_client: Optional[Any] = None,
    openai_client: Optional[Any] = None,
    ollama_client: Optional[Any] = None,
) -> LLMClient:
    """Build an LLM client adapter from provider configuration."""

    normalized_provider = _normalize_provider_name(provider)

    if normalized_provider == "cohere":
        client = cohere_client
        if client is None:
            if not api_key:
                raise ValueError("Cohere LLM provider requires api_key.")
            try:
                from cohere import Client as CohereClient
            except ImportError as exc:
                raise RuntimeError("Cohere SDK is not installed. Install package 'cohere'.") from exc
            client = CohereClient(api_key=api_key)
        return CohereLLMClientAdapter(client)

    if normalized_provider == "openai":
        client = openai_client
        if client is None:
            if not api_key:
                raise ValueError("OpenAI LLM provider requires api_key.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("OpenAI SDK is not installed. Install package 'openai'.") from exc
            client = OpenAI(api_key=api_key, base_url=base_url)
        return OpenAILLMClientAdapter(client=client, model=model or DEFAULT_OPENAI_CHAT_MODEL)

    if normalized_provider == "ollama":
        client = ollama_client
        if client is None:
            try:
                import ollama
            except ImportError as exc:
                raise RuntimeError("Ollama SDK is not installed. Install package 'ollama'.") from exc
            client = ollama.Client(host=base_url) if base_url else ollama.Client()
        return OllamaLLMClientAdapter(client=client, model=model or DEFAULT_OLLAMA_CHAT_MODEL)

    raise ValueError(
        "Unsupported LLM provider: "
        f"{provider}. Supported providers: cohere, openai, ollama."
    )