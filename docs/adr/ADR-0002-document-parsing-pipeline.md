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

## Amendment 1: LiteParse is opt-in; built-in parsers are the default (proposed, 2026-09-29)

### Context
The adapter invoked `npx --no-install @run-llama/liteparse --json <file>`. No package with that name exists on npm. LiteParse is published as `@llamaindex/liteparse`, whose CLI is `lit parse <file> --format json` and returns a `pages[]` structure. The adapter could never work, yet `get_parser()` selected it whenever Node.js and npx were installed. So every PDF, DOCX or HTML file paid for a failed npx call (about 1.4 s) before falling back. And with a plain `pip install`, the optional fallback dependencies were missing, so PDF and DOCX files failed entirely (#100).

### Decision
- `pypdf`, `python-docx` and `beautifulsoup4` become regular dependencies, so the built-in parser works after a plain install. If one is missing at runtime anyway, `ParserUnavailableError` names the `pip install` command.
- `LiteParseAdapter.available()` is true only when `RAGSEARCH_LITEPARSE_CLI` points at a compatible CLI. It is no longer inferred from `node`/`npx`, and the adapter never calls npx.
- Supporting the official LiteParse CLI is deferred to #102 (0.3.0), which requires a version-pinned install step and no download-on-demand.

### Consequences
- Unstructured parsing works out of the box, with no Node.js dependency and no per-file npx delay.
- `.doc` and image files are unsupported by default until #102.
- The fallback runtime-recovery path (decision 3) now applies only when a user-configured CLI fails.

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
