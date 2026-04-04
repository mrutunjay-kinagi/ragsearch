# ADR-0004: LLM Provider Registry/Factory

- Status: Accepted
- Date: 2026-04-04
- Epic: #50
- Slice: #54 (S3)

## Context

The generation boundary was still coupled to a single Cohere-specific adapter in setup. That made it difficult to configure alternative local or hosted LLM providers while keeping the engine contract stable.

## Decision

Introduce a provider registry/factory for generation clients:

1. `LLMClient` remains the stable engine-facing contract.
- Requires `generate(prompt, **kwargs) -> str`.

2. `create_llm_client(...)` resolves provider configuration into an adapter.
- Phase-1 providers: `cohere` (default), `openai`, `ollama`.
- Unknown provider names fail fast with a deterministic configuration error.

3. Provider-specific adapters normalize response shape.
- `CohereLLMClientAdapter` preserves existing behavior.
- `OpenAILLMClientAdapter` supports OpenAI chat/completions-style responses.
- `OllamaLLMClientAdapter` supports Ollama chat/generate-style responses.

4. `setup()` remains backward-compatible.
- Existing callers can continue passing only `data_path` and `llm_api_key`.
- Provider selection is additive through optional keyword arguments.

## Consequences

Positive:
- Stable engine contract with provider-agnostic generation wiring.
- Additive path for multimodel support without changing query/search call sites.
- Clear failure mode for invalid provider configuration.

Trade-offs:
- Optional provider SDKs become environment-dependent when selected.
- Response normalization assumptions must be kept in sync with provider SDK changes.

## Testing Notes

- Adapter tests verify response normalization and provider selection.
- Setup tests verify factory wiring and invalid provider handling.
- Full suite regression coverage remains required before merge.

## Related

- Issue #54
- `libs/ragsearch/llm_clients.py`
- `libs/ragsearch/setup.py`
