# ADR-0005: Retrieval-to-Generation Pipeline

- Status: Accepted
- Date: 2026-04-04
- Epic: #50
- Slice: #55 (S4)

## Context

The engine could retrieve relevant records and return citation-rich search results, but there was no first-class path to turn those retrieved records into a grounded natural-language answer while preserving the citation payload.

## Decision

Add an additive retrieval-to-generation pipeline on top of the existing search flow:

1. `RagSearchEngine.answer(query, top_k=5)` generates a grounded answer.
- It reuses `search()` to retrieve results.
- It builds a deterministic prompt from the retrieved results.
- It calls `llm_client.generate(prompt)` and returns a structured payload.

2. Retrieval results remain the source of truth for citations.
- The answer payload includes the original `results` and a flattened `citations` list.
- Existing `search()` behavior remains unchanged.

3. The public HTTP API is additive.
- `/query` continues to serve search-only results.
- `/answer` exposes the answer pipeline without changing legacy search consumers.

## Consequences

Positive:
- Simple grounded-answer path built on existing retrieval primitives.
- Citation payload is preserved and easy to inspect.
- Backward compatibility is maintained for search consumers.

Trade-offs:
- The answer prompt is intentionally conservative and may be terse.
- Any prompt schema changes should be coordinated with tests and documentation.

## Testing Notes

- Tests verify context assembly and prompt construction.
- Tests verify answer output structure and citation preservation.
- Search-only APIs remain covered by existing regression tests.

## Related

- Issue #55
- `libs/ragsearch/engine.py`
- `libs/ragsearch/llm_clients.py`
