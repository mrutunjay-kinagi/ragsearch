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
   loads CSV/Parquet/JSON (and Excel via pandas, see Consequences) into a
   DuckDB database file in the index directory, then closes its read-write
   connection. Generated SQL runs on a separate connection opened with
   `read_only=True`; DuckDB refuses to open the same file read-only while a
   read-write connection to it is still open in the same process.
2. **No external access.** The query connection applies its settings in this
   order: disable extension auto-install/auto-load, set
   `enable_external_access = false`, set `memory_limit` and `threads`
   (guardrail 5), and last set `lock_configuration = true` so the query cannot
   change any of them.
   This blocks `read_csv('/etc/...')`, `COPY ... TO`, `ATTACH`, and HTTP/S3 reads.
3. **Single SELECT only.** Parse with `connection.extract_statements(sql)`;
   reject anything that is not exactly one statement of type `SELECT`
   (rejects `SET`, `ATTACH`, `COPY`, `INSTALL`, `LOAD`, `CALL`, DML/DDL).
   DuckDB reports `PRAGMA` as `SELECT`, so after stripping SQL comments also
   reject a leading `PRAGMA` keyword. `DESCRIBE`, `SHOW`, `SUMMARIZE` and
   `FROM x` also report as `SELECT` and are allowed on purpose (read-only
   schema inspection). This check is defence in depth; guardrails 1–2 are the
   primary controls.
4. **Bounded results.** Apply the limit with the relation API,
   `connection.sql(query).limit(n)` (default 1,000 rows), and report when the
   limit was hit. Unlike wrapping the text as `SELECT * FROM (<query>) LIMIT n`,
   this tolerates a trailing `;` or `-- comment` in model output.
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
- **SQLite.** It can be locked down (`mode=ro`, `set_authorizer`, attach
  limits, extensions off by default), but it has no native Parquet reading
  and weaker analytical SQL. Rejected on capability, not safety.
- **A fixed set of aggregation functions chosen by the model (tool calls).**
  Safest, but too narrow for group-bys, filters, and joins across sheets.
  May still be used later for a "safe mode".

## Consequences

Positive:

- Aggregation cannot run arbitrary code, touch the file system, or reach the network.
- Answers are auditable: the SQL and rows are shown.
- DuckDB reads CSV, Parquet, and JSON natively and handles analytical queries at scale.

Trade-offs:

- Adds an optional native dependency.
- Excel is not built into the DuckDB Python wheel (the `excel` extension is
  downloaded on first use), which conflicts with guardrail 2. Load Excel
  through pandas at ingestion, as `setup()` does today.
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
`interrupt()` stopped a long query at the timeout; `.limit(n)` capped
results, including for queries ending in `;` or a `--` comment. `extract_statements` classified `PRAGMA version` as `SELECT`, hence
the extra keyword check in guardrail 3.
