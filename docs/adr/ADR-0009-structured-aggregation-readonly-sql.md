# ADR-0009: Structured-data aggregation via read-only SQL

- Status: proposed
- Date: 2026-09-24
- Related issue: #81

## Context

Classic top-k retrieval cannot answer aggregate questions over structured data
("average claim amount by region", "how many claims were denied in 2024").
The roadmap adds an `aggregate` query strategy that turns a question into a
query over the user's tabular data.

The query text comes from model output. Model output is untrusted: it can be
wrong, and it can be steered by prompt injection hidden in the documents or the
question. The obvious shortcut, asking the model for pandas/Python code and
running it, gives that untrusted output arbitrary code execution in the user's
process (file system, network, credentials in the environment).

## Decision

Aggregation runs only as **read-only SQL in an embedded DuckDB engine**.
ragsearch never executes model-generated Python, pandas expressions, or shell
commands.

Guardrails, all enforced in library code rather than in the prompt:

1. **Trusted load, untrusted query.** Ingestion (trusted code, no model input)
   loads CSV/Parquet/Excel/JSON into a DuckDB database file in the index
   directory. Generated SQL runs on a separate connection opened with
   `read_only=True`.
2. **No external access.** The query connection sets
   `enable_external_access = false`, disables extension auto-install/auto-load,
   and then sets `lock_configuration = true` so the query cannot re-enable them.
   This blocks `read_csv('/etc/...')`, `COPY ... TO`, `ATTACH`, and HTTP/S3 reads.
3. **Single SELECT only.** Parse with `connection.extract_statements(sql)`;
   reject anything that is not exactly one statement of type `SELECT`
   (rejects `SET`, `ATTACH`, `COPY`, `INSTALL`, `LOAD`, DML/DDL). DuckDB
   reports `PRAGMA` as `SELECT`, so also reject a leading `PRAGMA`/`CALL`
   keyword. This check is defence in depth; guardrails 1–2 are the primary
   controls.
4. **Bounded results.** Wrap the query as `SELECT * FROM (<query>) LIMIT n`
   (default 1,000 rows) and report when the limit was hit.
5. **Bounded cost.** Per-query timeout via `connection.interrupt()` from a timer
   (default 10 s), plus `memory_limit` and `threads` settings on the query
   connection.
6. **Transparent answers.** Every aggregate answer returns the executed SQL,
   the result rows (or a preview), and whether limits were hit, so users can
   verify it. Errors are returned to the planner, which may retry a bounded
   number of times with the error message.

DuckDB becomes an optional dependency (e.g. `ragsearch[structured]`); without
it the `aggregate` strategy is unavailable and the planner does not route to it.

## Alternatives considered

- **Model-generated pandas/Python in a sandbox.** Sandboxing Python in-process
  is not reliable; a subprocess/container sandbox adds heavy platform-specific
  setup and still exposes more surface than SQL. Rejected.
- **SQLite.** No native Parquet/Excel reading, weaker analytical SQL, and no
  equivalent to `enable_external_access` + `lock_configuration`. Rejected.
- **A fixed set of aggregation functions chosen by the model (tool calls).**
  Safest, but too narrow for group-bys, filters, and joins across sheets.
  May still be used later for a "safe mode".

## Consequences

Positive:

- Aggregation cannot run arbitrary code, touch the file system, or reach the network.
- Answers are auditable: the SQL and rows are shown.
- DuckDB reads CSV, Parquet, and Excel directly and handles analytical queries at scale.

Trade-offs:

- Adds an optional native dependency.
- Text-to-SQL quality depends on schema profiling and a data dictionary
  (roadmap Part 5); those are follow-up work.
- Read-only SQL can still return sensitive columns. Document-level permission
  filters must also apply to structured tables (tracked separately).

## Verification

Before `aggregate` ships, tests must show that each of these is rejected or
contained: multiple statements, non-SELECT statements (`COPY`, `ATTACH`,
`INSTALL`, `SET`, `PRAGMA`), file-reading table functions (`read_csv`,
`read_text`), re-enabling configuration, runaway queries (timeout), and
oversized results (row limit).

Prototype check (DuckDB 1.4.5, guardrails 1, 2, 4, 5 applied): an aggregate
`SELECT` succeeded; `read_csv`, `read_text`, `COPY ... TO`, `ATTACH`,
`INSERT`, `INSTALL`, and `SET enable_external_access=true` were all refused;
`interrupt()` stopped a long query at the timeout; the `LIMIT` wrapper capped
results. `extract_statements` classified `PRAGMA version` as `SELECT`, hence
the extra keyword check in guardrail 3.
