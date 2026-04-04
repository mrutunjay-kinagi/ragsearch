# ADR-0008: Structured observability and evaluation harness baseline

- Status: accepted
- Date: 2026-04-04
- Related issue: #58

## Context

Issue #58 requires deterministic telemetry and a simple regression harness so
retrieval and answer quality changes can be validated before merge. Existing
engine/setup behavior exposes ingestion diagnostics but does not provide a
uniform event stream across indexing, retrieval, and generation.

## Decision

Introduce a baseline observability and evaluation contract:

- Add structured engine-level events in `rag_engine.observability_events`.
- Emit deterministic payload fields for retrieval and generation stages:
  - retrieval: `query`, `top_k`, `results_count`, `latency_ms`
  - generation: `query`, `top_k`, `results_count`, `citations_count`, `latency_ms`
- Emit indexing completion event with ingestion/indexing diagnostic payload for FAISS/in-process indexing path.
- Extend setup diagnostics with deterministic `observability` metrics:
  - `setup_latency_ms`, `loaded_records`, `selected_parser`, `fallback_recovered`
- Add evaluation harness module `ragsearch/evaluation.py`:
  - `run_regression_gates(engine, cases, thresholds)` returns deterministic summary
  - `EvaluationThresholds` supports minimum retrieval/citation guardrails
- Gate baseline unit behavior in CI with deterministic evaluation tests.

## Consequences

Positive:

- Establishes a stable telemetry contract for test and debugging workflows.
- Enables deterministic quality gates without introducing external services.
- Provides a lightweight foundation for future metric exporters/eval suites.

Trade-offs:

- In-memory event storage grows with engine usage unless capped/configured by caller.
- Baseline thresholds are intentionally simple and do not replace richer quality metrics.

## Verification

Tests cover:

- retrieval and generation event emission contracts in engine tests
- indexing event emission contract for FAISS/in-process indexing path
- setup ingestion diagnostics observability contract
- evaluation harness pass/fail and input-validation behavior with deterministic fake-engine cases
