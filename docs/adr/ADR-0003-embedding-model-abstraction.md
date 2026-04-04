# ADR-0003: Embedding Model Abstraction (A1)

- Status: Accepted
- Date: 2026-04-04
- Epic: #45
- Slice: #46 (M1-A1)

## Context

The retrieval/indexing path was tightly coupled to concrete embedding-client behavior.
This made it harder to validate provider response shapes consistently and increased
risk when swapping providers or handling probe-time failures.

## Decision

Introduce an embedding boundary with three parts:

1. `EmbeddingModel` protocol contract
- Requires `embed(texts=[...])` support.
- Response must provide `embeddings` as a non-empty sequence of numeric vectors.

2. Compatibility adapter
- `CohereEmbeddingAdapter` wraps current provider usage without changing public setup/search APIs.

3. Shared normalization utilities
- `extract_embeddings(response)` validates and normalizes embedding payloads.
- `infer_embedding_dimension(model)` probes one deterministic input to infer vector dimension.

Setup behavior:
- Preferred path: infer dimension from probe response.
- Compatibility fallback: if probe inference fails (invalid response shape or runtime provider error), use legacy dimension `4096` and continue.

## Consequences

Positive:
- Clear contract for embedding providers.
- Centralized validation for embedding payload shape.
- Reduced direct coupling to a specific embedding client in engine/setup.

Trade-offs:
- Setup now performs a probe call to infer dimension.
- Fallback to legacy dimension can defer certain dimension mismatch failures to runtime if provider output is inconsistent.

## Validation and Test Mapping

- Invalid embedding response shape raises deterministic `ValueError` in engine search path.
- Setup dimension inference uses adapter probe output when valid.
- Setup falls back to legacy `4096` when probe shape is invalid.
- Setup falls back to legacy `4096` when probe raises runtime/provider exceptions.

## Non-goals (A1)

- Full provider marketplace and dynamic plugin loading.
- Advanced generation/streaming abstractions (handled in later slices).
- Performance tuning beyond compatibility-safe boundary changes.
