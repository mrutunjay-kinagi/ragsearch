# RAGSearch Documentation

Complete guide to retrieval-augmented generation with ragsearch.

## Quick Start

- **New to ragsearch?** → [Quickstart Guide](./quickstart.md) (5 min read)
- **Try an example?** → [Cookbook: Dataset Analytics](./cookbook-dataset-analytics.md)
- **Need API details?** → [API Reference & Cheat Sheet](./reference-api-cheat-sheet.md)

## Core Documentation

### Using ragsearch

| Document | Purpose | Audience |
|----------|---------|----------|
| [Quickstart Guide](./quickstart.md) | Get up and running in 5 minutes | New users |
| [Cookbook: Dataset Analytics](./cookbook-dataset-analytics.md) | Real-world example with Titanic data | All users |
| [API Reference & Cheat Sheet](./reference-api-cheat-sheet.md) | Complete API contract and schemas | Developers |

### Troubleshooting & Quality

| Document | Purpose | Audience |
|----------|---------|----------|
| [Troubleshooting Guide](./troubleshooting.md) | Common issues and solutions | Users experiencing problems |
| [Benchmark Interpretation Guide](./benchmark-interpretation.md) | Understanding test results and metrics | QA & maintainers |

## Architecture & Design

- **Architecture Decision Records** → See [ADR overview](./adr/README.md)
- **Document parsing pipeline** → [ADR-0002](./adr/ADR-0002-document-parsing-pipeline.md)
- **Embedding model abstraction** → [ADR-0003](./adr/ADR-0003-embedding-model-abstraction.md)
- **LLM provider registry/factory** → [ADR-0004](./adr/ADR-0004-llm-provider-registry.md)
- **Retrieval-to-generation pipeline** → [ADR-0005](./adr/ADR-0005-retrieval-generation-pipeline.md)
- **Incremental indexing manifest** → [ADR-0006](./adr/ADR-0006-incremental-indexing-manifest.md)
- **Retrieval quality hooks** → [ADR-0007](./adr/ADR-0007-retrieval-quality-hooks.md)
- **Observability & evaluation baseline** → [ADR-0008](./adr/ADR-0008-observability-evaluation-baseline.md)

## Support

- **Report issues** → [GitHub Issues](https://github.com/ragsearch/ragsearch/issues)
- **Suggest improvements** → Open a discussion or PR
- **Questions?** → See [Troubleshooting](./troubleshooting.md) first

---

## Documentation Map

```
docs/
├── INDEX.md                              # This file
├── quickstart.md                         # Getting started (5 min)
├── cookbook-dataset-analytics.md         # End-to-end example
├── reference-api-cheat-sheet.md          # API contracts + examples
├── troubleshooting.md                    # Common issues & solutions
├── benchmark-interpretation.md           # Metrics & test results
├── adr/                                  # Architecture decisions
│   ├── README.md
│   ├── ADR-0002-document-parsing-pipeline.md
│   ├── ADR-0003-embedding-model-abstraction.md
│   ├── ADR-0004-llm-provider-registry.md
│   ├── ADR-0005-retrieval-generation-pipeline.md
│   ├── ADR-0006-incremental-indexing-manifest.md
│   ├── ADR-0007-retrieval-quality-hooks.md
│   └── ADR-0008-observability-evaluation-baseline.md
└── _build/                               # Generated Sphinx docs (HTML)
```

---

*Last updated: Slice C (2026-Q1) — Comprehensive API docs, troubleshooting, and benchmarking guides.*
