# Top 10 Architecture & Technical Questions

**Author:** Architect Agent  
**Status:** Proposed  
**Date:** 2026-03-28  
**Related issues:** [#21 — Gather & prioritize top 10 architectural questions](https://github.com/mrutunjay-kinagi/ragsearch/issues/21), [#19 — Enterprise-grade RAG library epic](https://github.com/mrutunjay-kinagi/ragsearch/issues/19)

---

This document captures the top 10 prioritized architecture and technical questions that must be decided before ragsearch can reach enterprise-grade quality (epic #19). Each question includes why it matters, concrete decision options, and the artifact that will capture the accepted decision.

---

## Q1 — 🔌 Provider Abstraction (LLM, Embedding & Vector Store Plug-in Interfaces)

**Priority: P0 — Critical foundation for all other work**

### Why it matters

`setup()` hard-wires `CohereClient` for both embedding and LLM generation, and `VectorDB` (`vector_db.py`) conflates a concrete FAISS index with module-level ChromaDB helper functions — there is no shared interface. Any user wanting OpenAI, Azure, Anthropic, or a local model must fork core files. This creates an unmaintainable vendor monoculture and blocks enterprise adoption.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | `Protocol`-based interfaces (`EmbeddingProvider`, `LLMProvider`, `VectorStoreProvider`) via `typing.Protocol` (structural subtyping) | Zero base-class overhead; easy mocking | Requires Python ≥ 3.8; no enforced `__init__` signature |
| B | Abstract base classes (`abc.ABC`) with `@abstractmethod` | Explicit contract; `isinstance` checks work | Slightly more boilerplate; inheritance coupling |
| C | Callable duck-typing + validation at `setup()` entry point only | Minimal code change | No IDE support; poor discoverability; no formal contract |

**Recommended:** Option B (ABC) for public-facing interfaces — clear contracts are essential for external contributors; optionally exposed as `Protocol` aliases for type-check-only use.

**Artifact:** `docs/adr/ADR-0001-provider-abstraction.md` → new module `ragsearch/providers/base.py`

---

## Q2 — 📄 Unstructured Document Ingestion Pipeline

**Priority: P1 — Enables the dominant enterprise use case**

### Why it matters

`setup()` only calls `pd.read_csv / read_json / read_parquet`, limiting ragsearch to tabular data. PDFs, DOCX files, HTML pages, audio transcripts, and images represent the vast majority of enterprise knowledge-base content. Without an ingestion pipeline the library cannot compete with LangChain, LlamaIndex, or Haystack.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Built-in loaders for each type (PyMuPDF, python-docx, BeautifulSoup, Whisper, etc.) bundled in core | Single install; consistent API | Heavy transitive dependencies; slow cold-start |
| B | `DocumentLoader` protocol with optional built-in loaders as extras (`pip install ragsearch[pdf]`) | Lean core; user pays for what they use | More packaging complexity |
| C | Delegate entirely to users — accept `List[Document]` as input | Zero new dependencies | Pushes all complexity to users; worse DX |

**Recommended:** Option B — follow the LlamaIndex extras pattern; define the `DocumentLoader` interface in `ragsearch/ingestion/base.py`.

**Artifact:** `docs/adr/ADR-0002-document-ingestion-pipeline.md` → `ragsearch/ingestion/` package

---

## Q3 — 🤖 Runtime & LLM Selection (Cohere, OpenAI, vLLM, Local Models)

**Priority: P1 — Enterprise deployments require LLM choice**

### Why it matters

`RagSearchEngine.__init__` accepts `embedding_model: CohereClient` and `llm_client: CohereClient` with explicit Cohere types in the signature. This makes the type system itself a barrier to runtime swappability. Enterprise deployments require air-gapped / on-premise options (vLLM, Ollama, HuggingFace) and multi-cloud providers (OpenAI, Azure OpenAI, Bedrock).

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Replace typed params with `LLMProvider` / `EmbeddingProvider` ABC (from Q1); ship adapters for Cohere, OpenAI, vLLM in `ragsearch/providers/` | Clean API; adapter pattern is well understood | Breaking change in `setup()` and `RagSearchEngine.__init__` |
| B | Accept any callable/object; validate duck-typed methods at runtime | Non-breaking | No static analysis help; cryptic runtime errors |
| C | Configuration-driven provider selection (string keys + registry) | Enables YAML/env-var switching | Magic strings; harder to test custom providers |

**Recommended:** Option A combined with a `ProviderRegistry` for string-based convenience (Option C used as a thin layer over A).

**Artifact:** `docs/adr/ADR-0003-runtime-llm-selection.md` → `ragsearch/providers/{cohere,openai,vllm}.py`

---

## Q4 — ✂️ Chunking Strategy & Context-Window Management

**Priority: P1 — Silent data loss with current implementation**

### Why it matters

`_process_and_store_embeddings()` in `engine.py` embeds entire rows as single strings via `preprocess_text(row, textual_columns)`. There is no splitting, no overlap, and the embedding dimension is hard-coded to `4096` in `setup()` (`VectorDB(embedding_dim=4096)`). Long documents are silently truncated by the model's token limit, causing invisible data loss and degraded retrieval quality.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Fixed-size token-aware chunker with configurable `chunk_size` / `chunk_overlap` (via `tiktoken` or provider tokenizer) | Predictable; provider-agnostic | Token counting adds latency |
| B | Semantic chunker (split on sentence/paragraph boundaries using NLTK or spaCy) | Higher retrieval quality | Heavier deps; harder to tune |
| C | Recursive character text splitter (LangChain-style heuristic, no NLP deps) | Lightweight; good default | Less semantically aware |

**Recommended:** Ship Option C as the default `Chunker` implementation; expose a `Chunker` ABC so users can plug in Option B. Make `chunk_size`, `chunk_overlap`, and `embedding_dim` first-class config fields — not hard-coded in `setup()`.

**Artifact:** `docs/adr/ADR-0004-chunking-strategy.md` → `ragsearch/ingestion/chunkers.py`

---

## Q5 — ⚡ Async / Streaming Architecture

**Priority: P2 — Required for high-QPS and streaming LLM responses**

### Why it matters

Every call in `engine.py` and `utils.py` is synchronous. The Flask server in `run()` uses a background `threading.Thread`, which means concurrent search requests block each other on the GIL. Enterprise use cases (high-QPS APIs, streaming LLM responses, large batch ingestion) require async I/O. The current design also makes integration into FastAPI or Django async views impossible.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Introduce `AsyncRagSearchEngine` with `async def search(...)` using `asyncio`; keep sync version as-is | No breaking change; progressive migration | Two code paths to maintain |
| B | Refactor core to be async-first; provide sync wrappers via `asyncio.run()` | Single clean code path | Breaking change; more migration effort |
| C | Use `concurrent.futures.ThreadPoolExecutor` to wrap sync calls | Trivial to implement | No true async; GIL still blocks CPU-bound ops |

**Recommended:** Option A for the current release — follow the `httpx` / `openai-python` dual-mode pattern. Deprecate the sync-only path in a future major version. Replace `threading.Thread` in `run()` as part of Q9 (web decoupling).

**Artifact:** `docs/adr/ADR-0005-async-streaming-architecture.md`

---

## Q6 — ⚙️ Configuration & Secrets Management

**Priority: P0 — API keys stored as plain strings is a security concern**

### Why it matters

`config.py`'s `load_configuration()` reads only JSON. API keys are passed as plain strings to `setup()` and stored as instance attributes on `RagSearchEngine`. There is no environment variable support, no YAML, no schema validation, and no secrets-safe handling. Enterprise deployments use Vault, AWS Secrets Manager, or at minimum environment variables — none of which are currently supported.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Pydantic v2 `BaseSettings` model — reads env vars, `.env` files, JSON, YAML; validated at startup | Full validation, IDE autocomplete, layered overrides | Adds `pydantic-settings` dependency |
| B | Simple `dataclasses` + `python-dotenv` for env vars, manual validation | Lighter dependency | No nested config support; boilerplate validation |
| C | Keep JSON-only; add env-var interpolation (`${VAR}` syntax in JSON values) | Minimal change | Non-standard; fragile parsing |

**Recommended:** Option A — Pydantic v2 `BaseSettings` is the de-facto standard for Python services. API keys must use `SecretStr` type to prevent accidental logging.

**Artifact:** `docs/adr/ADR-0006-configuration-secrets.md` → `ragsearch/settings.py` (replaces `ragsearch/config.py`)

---

## Q7 — 📊 Observability (Structured Logging, Metrics & Distributed Tracing)

**Priority: P2 — Enterprise operations requirement**

### Why it matters

Every module calls `logging.basicConfig(level=logging.INFO, ...)` at module level. This is a **library anti-pattern** — it overrides the host application's log configuration the moment ragsearch is imported. There are no structured log fields, no request IDs, no latency metrics, and no trace context. Enterprise deployments require integration with Datadog, Grafana, and OpenTelemetry stacks.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Replace `basicConfig` with `structlog` for structured JSON logs; emit OpenTelemetry spans for `embed`/`search`/`ingest`; expose Prometheus metrics via optional `ragsearch[observability]` extra | Industry standard; composable | ~3 new optional deps |
| B | Switch to `logging.getLogger(__name__)` (no `basicConfig`), add a `JSONFormatter`; defer metrics/tracing | Zero new deps; fixes anti-pattern immediately | No metrics, no tracing |
| C | Provide lifecycle hooks/callbacks (`on_search_start`, `on_search_end`) and let users instrument | Maximum flexibility | Pushes all observability burden to users |

**Recommended:** Option B immediately (non-breaking, zero deps — removes the `logging.basicConfig` anti-pattern in all four modules); add Option A as `ragsearch[observability]` extra in the next minor release. Add lifecycle hooks from Option C as well — they enable custom APM integrations.

**Artifact:** `docs/adr/ADR-0007-observability.md`

---

## Q8 — 🏢 Multi-Tenancy & Data Isolation

**Priority: P2 — Required before any SaaS deployment**

### Why it matters

The current `VectorDB` (FAISS) is a single shared in-memory index; ChromaDB uses a single `collection_name` per engine instance with no access control. There is no namespace isolation and no per-tenant boundary. A multi-tenant SaaS deployment would mix all customer data in one index, creating both a **data privacy risk** and incorrect cross-tenant retrieval results.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | `tenant_id` as a first-class concept in `RagSearchEngine`; ChromaDB collection name and FAISS index prefixed/isolated per tenant | Clean isolation; maps to existing ChromaDB collection model | Requires collection-per-tenant lifecycle management |
| B | Metadata filtering at query time — store `tenant_id` in embedding metadata, filter results post-search | Simple implementation | No true isolation; FAISS doesn't support metadata filtering natively |
| C | External orchestration only — document that users must instantiate one engine per tenant | Zero code change | Not enterprise-ready; poor DX |

**Recommended:** Option A for vector store isolation + RBAC hooks (documented interface, not yet implemented) for access control.

**Artifact:** `docs/adr/ADR-0008-multi-tenancy.md`

---

## Q9 — 🌐 Web Interface Decoupling (Flask Embedded in `RagSearchEngine.run()`)

**Priority: P0 — Flask is a forced dependency for a pure library use case**

### Why it matters

`RagSearchEngine.run()` in `engine.py` directly imports `flask`, creates a `Flask` app, defines route handlers as closures over `self`, and launches a `threading.Thread`. This means: (1) Flask is a **mandatory transitive dependency** even for users who only call `engine.search()`; (2) there is no way to integrate into an existing web framework; (3) the app cannot be deployed as a standalone WSGI/ASGI service; and (4) threading conflicts arise under production WSGI servers (gunicorn, uWSGI).

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Extract Flask app to `ragsearch/server/` sub-package (`ragsearch[server]` optional extra); `RagSearchEngine` has zero Flask imports; server wires engine via dependency injection | Clean separation; library vs. server are independent | More files; users must explicitly install extra |
| B | Provide a FastAPI app factory `create_app(engine: RagSearchEngine) -> FastAPI` instead of Flask; mark `run()` as deprecated | Modern async-ready API; better production story | Breaking change for existing `run()` users |
| C | Keep current approach but gate Flask import with `try/except ImportError` | Minimal change | Still tight coupling; confusing when Flask not installed |

**Recommended:** Option A first (extract, keep Flask for backward-compat), then Option B in the next major version (FastAPI app factory with streaming support, aligned with Q5).

**Artifact:** `docs/adr/ADR-0009-web-interface-decoupling.md` → `ragsearch/server/flask_app.py`

---

## Q10 — 🔖 Versioning, Backward-Compatibility & Deprecation Strategy

**Priority: P0 — Foundation for a trustworthy library**

### Why it matters

There are currently no deprecation warnings in any module, no `CHANGELOG`, no migration guides, and no declared compatibility policy. With the scope of breaking changes required by Q1–Q9 above (new ABCs, new `settings.py`, new `setup()` signature, async APIs), users will experience **silent breakage on upgrade**. Enterprise users require stability guarantees and advance notice before adopting a library.

### Decision options

| Option | Summary | Pro | Con |
|--------|---------|-----|-----|
| A | Strict SemVer with `CHANGELOG.md` (Keep a Changelog format), `warnings.warn(..., DeprecationWarning)` on deprecated APIs, 1-minor-version deprecation window before removal | Industry standard; builds trust | Requires discipline across all contributors |
| B | Calendar versioning (CalVer `YYYY.MM.PATCH`) | Simple; communicates recency | Doesn't communicate compatibility guarantees |
| C | No formal policy — rely on release notes | Zero overhead now | Unusable in enterprise contexts with dependency pinning |

**Recommended:** Option A — SemVer is non-negotiable for a library. Add a `deprecated` decorator utility to `ragsearch/utils.py`, enforce `CHANGELOG.md` updates in CI, and document the compatibility policy in `CONTRIBUTING.md`.

**Artifact:** `docs/adr/ADR-0010-versioning-deprecation-strategy.md` + `CHANGELOG.md` + CI changelog-lint step

---

## Summary Priority Matrix

| # | Topic | Impact | Effort | Priority |
|---|-------|--------|--------|----------|
| Q1 | Provider abstraction | 🔴 Critical | Medium | P0 |
| Q6 | Config & secrets | 🔴 Critical | Low | P0 |
| Q9 | Web decoupling | 🔴 Critical | Medium | P0 |
| Q10 | Versioning/deprecation | 🟢 Foundation | Low | P0 (process) |
| Q2 | Unstructured ingestion | 🟠 High | High | P1 |
| Q3 | Runtime/LLM selection | 🟠 High | Medium | P1 |
| Q4 | Chunking strategy | 🟠 High | Medium | P1 |
| Q5 | Async/streaming | 🟡 Medium | High | P2 |
| Q7 | Observability | 🟡 Medium | Low | P2 |
| Q8 | Multi-tenancy | 🟡 Medium | High | P2 |

---

## ADR Backlog Created by This Document

The following ADR stubs are to be authored (one PR per ADR) as decisions are made:

- [ ] `docs/adr/ADR-0001-provider-abstraction.md`
- [ ] `docs/adr/ADR-0002-document-ingestion-pipeline.md`
- [ ] `docs/adr/ADR-0003-runtime-llm-selection.md`
- [ ] `docs/adr/ADR-0004-chunking-strategy.md`
- [ ] `docs/adr/ADR-0005-async-streaming-architecture.md`
- [ ] `docs/adr/ADR-0006-configuration-secrets.md`
- [ ] `docs/adr/ADR-0007-observability.md`
- [ ] `docs/adr/ADR-0008-multi-tenancy.md`
- [ ] `docs/adr/ADR-0009-web-interface-decoupling.md`
- [ ] `docs/adr/ADR-0010-versioning-deprecation-strategy.md`

> Linked to enterprise-grade epic #19
