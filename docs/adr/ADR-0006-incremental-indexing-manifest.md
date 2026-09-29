# ADR-0006: Incremental indexing manifest and changed-file detection

- Status: accepted
- Date: 2026-04-04
- Related issue: #56

## Context

Full re-embedding on every setup run is wasteful when most source records are unchanged.
For the FAISS path, we need deterministic skip-unchanged behavior while keeping the in-memory
vector index complete for each engine instance.

## Decision

Implement incremental indexing with an on-disk embedding manifest:

- Persist a manifest under the embeddings directory with per-record content hash and embedding.
- Compute deterministic cache keys per record and compare content hash to detect changes.
- Reuse embeddings for unchanged records and generate embeddings only for new/changed records.
- Rebuild the in-memory FAISS index on each run using a mix of reused and newly generated embeddings.
- Expose deterministic indexing counters on `engine.indexing_diagnostics`, and include them in
  `engine.ingestion_diagnostics["indexing"]` for setup consumers.

## Consequences

Positive:

- Repeated setup runs avoid unnecessary embedding API calls for unchanged data.
- Changed/new records are explicitly re-embedded, reducing stale-index risk.
- Diagnostics provide deterministic observability for first-run, no-change rerun,
  and changed-file rerun scenarios.

Trade-offs:

- Manifest files introduce local state management requirements.
- Cache key strategy is tied to record ordering plus available source metadata.

## Amendment 1: embedding model identity in the manifest (proposed, 2026-09-29)

### Context

The cache key and content hash describe the record, not the model that embedded it. After
`embedding_provider`/`embedding_model_name` changed, cached vectors were reused anyway: with a new
dimension `setup()` failed with an empty `Embedding indexing failed` error, and with the same
dimension retrieval silently mixed vectors from two models. PR #90 changed the default Cohere
embedding model (`large`, 4096-d, retired by Cohere, to `embed-v4.0`, 1536-d), so every existing
default cache would break on upgrade.

### Decision

- Manifest schema version 2 adds `embedding_model` (the adapter class plus its model name, e.g.
  `CohereEmbeddingAdapter:embed-v4.0`) and `embedding_dim`.
- If the cached identity or dimension differs from the current engine, the whole cache is
  discarded, a warning is logged, every record is re-embedded, and
  `indexing_diagnostics["cache_invalidated_reason"]` explains why.
- A version 1 manifest (no identity) that still has records is treated as a mismatch: the model
  that produced it cannot be known, so it is re-embedded once and rewritten as version 2.
- As a second guard, a cached vector is never reused if its length differs from the vector
  backend's dimension.

### Consequences

- Upgrading from 0.1.5 or earlier re-embeds each cached corpus once, even if the model did not
  change. That one-time cost was chosen over silently reusing vectors of unknown origin.
  Checking only the dimension would avoid it, but would miss same-dimension model changes.
- Custom embedding models are identified by class name, plus a string `model_name` or `model`
  attribute when present. Custom models without one are only protected by the dimension check
  when switched between two configurations of the same class.

## Verification

Tests cover:

- first-run indexing (all records new)
- no-change rerun (all records reused)
- changed-record rerun (only changed records embedded)
- upgrade from a version 1 manifest written with a different model and dimension (all re-embedded, warning logged)
- same-dimension model change and dimension-only change (all re-embedded)
- unchanged model and dimension (cache reused, no warning)
