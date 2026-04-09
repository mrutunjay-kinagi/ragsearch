# Benchmark Interpretation Guide

**Status:** Expanded in Slice B/C with detailed metrics, examples, and trend analysis.

Understanding `.benchmarks/` outputs and evaluation metrics.

## Directory Structure

```
.benchmarks/
├── runs/
│   ├── 20260409-235448_public_titanic/
│   │   ├── summary.json           # Pass/fail results for each case
│   │   ├── notes.md              # Run metadata and notes
│   │   └── embeddings/           # Cached embeddings (git-ignored)
│   └── ...
├── history/
│   └── metrics.csv              # Trend data: all runs + outcomes
└── benchmark_result_schema.json  # Schema definition
└── README.md                     # Benchmark conventions
```

## Key Metrics

| Metric | Meaning |
|--------|---------|
| `pass` | All test cases met their thresholds |
| `passed_cases` | Count of cases that passed |
| `total_cases` | Count of test cases executed |
| `failed_cases` | Count of cases that failed |

## Interpreting Results

A **pass** result means:
1. ✓ Retrieved at least `min_results` documents
2. ✓ Generated citations for all results
3. ✓ No crashes or exceptions

It does **NOT** guarantee semantic correctness of the answer or source accuracy.

See [Known Limitations](./troubleshooting.md) for details.

---

**See also:** [Quickstart Guide](./quickstart.md) | [Evaluation API](./reference-api-cheat-sheet.md)
