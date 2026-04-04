# ADR-0007: Retrieval quality hooks for chunking and reranking

- Status: accepted
- Date: 2026-04-04
- Related issue: #57

## Context

Retrieval quality needs configurable extension points without changing baseline
behavior. The existing path indexes one embedding per row and returns raw
similarity-ranked results. We need additive hooks for chunking and reranking
while preserving deterministic defaults and backward compatibility.

## Decision

Introduce optional retrieval quality hooks:

- Add chunking strategy interface (`chunk_text(text) -> list[str]`) used before indexing.
- Add reranker interface (`rerank(query, results) -> list[dict]`) used after retrieval.
- Keep defaults equivalent to current behavior:
  - `RowChunkingStrategy`: row-level indexing parity.
  - `NoOpReranker`: no change to retrieval ordering.
- Wire hooks through `setup()` into `RagSearchEngine` as optional parameters.
- Keep hook behavior additive and deterministic, with explicit tests for parity and boundaries.

## Consequences

Positive:

- Enables retrieval quality experiments without forcing framework migration.
- Preserves stable behavior for existing users when hooks are not configured.
- Creates a clean contract for future quality upgrades (e.g., semantic chunking, model-based rerankers).

Trade-offs:

- Chunking changes can alter retrieval boundaries and manifest/cache cardinality.
- Misconfigured custom hooks can degrade relevance or latency if not validated.

## Verification

Tests cover:

- default path parity against explicit default hooks
- deterministic chunking boundary behavior
- reranker-enabled ordering behavior
- setup wiring for chunking and reranker parameters
