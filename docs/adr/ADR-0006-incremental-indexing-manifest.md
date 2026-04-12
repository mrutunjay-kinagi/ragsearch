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

## Verification

Tests cover:

- first-run indexing (all records new)
- no-change rerun (all records reused)
- changed-record rerun (only changed records embedded)
