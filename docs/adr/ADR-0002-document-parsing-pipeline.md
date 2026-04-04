# ADR-0002: Document Parsing Pipeline (LiteParse + Fallback)

## Status
Accepted

## Context
Issue #18 introduced parser boundary abstractions and setup-path integration for unstructured ingestion. We needed a deterministic strategy for:
- selecting parsing backends,
- handling parser unavailability,
- preserving typed errors,
- and protecting downstream indexing from empty content.

## Decision
1. `LiteParseAdapter` is the preferred parser when available.
2. `FallbackParser` is used when LiteParse is unavailable and file type is supported.
3. When LiteParse is selected but fails at runtime, setup retries with `FallbackParser` for fallback-supported file types.
4. Parser selection is centralized in `get_parser()`.
5. `setup()` routes structured files through pandas and unstructured files through parser dispatch.
6. Empty or whitespace-only parsed content is filtered before indexing.
7. Parser failures surface typed `RagSearchError` subclasses for predictable handling.
8. `setup()` publishes deterministic per-file ingestion diagnostics on the engine (`ingestion_diagnostics`) with parser selection, status, and fallback failure reason.

## Consequences
- Better extraction quality for complex formats when LiteParse is available.
- Graceful degradation when LiteParse or optional parser dependencies are missing.
- Improved runtime resiliency when LiteParse fails after selection and fallback can parse the same file type.
- Improved operational clarity through deterministic ingestion diagnostics for each setup input.
- Deterministic error handling for timeout, corruption, unsupported type, and unavailable parser cases.
- Existing structured ingestion flows remain unchanged.

## Testing Notes
- Parser unit tests verify dispatch, timeout behavior, malformed output handling, optional dependency failures, and unsupported extensions.
- Setup integration tests verify structured-path bypass, unstructured-path parser use, whitespace filtering, and exception propagation.
- Full suite regression coverage remains required before merge.

## Related
- Issue #18
- `libs/ragsearch/parsers/_dispatch.py`
- `libs/ragsearch/parsers/_liteparse.py`
- `libs/ragsearch/parsers/_fallback.py`
- `libs/ragsearch/setup.py`
