"""LLM client abstraction boundary and adapters."""

from typing import Any, Protocol, runtime_checkable


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