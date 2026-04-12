# Benchmark Interpretation Guide

Understanding `.benchmarks/` outputs, metrics, and trend analysis.

## Directory Structure

```
.benchmarks/
├── runs/
│   ├── 20260409-235448_public_titanic/
│   │   ├── summary.json              # Test cases + results
│   │   ├── notes.md                  # Run metadata and notes
│   │   └── embeddings/               # Cached embeddings (git-ignored)
│   ├── 20260409-235447_dataset/
│   └── ...
├── history/
│   └── metrics.csv                  # All runs: trend and delta tracking
└── README.md                         # Benchmark conventions
```

## Key Metrics

### Summary Metrics (per run)

| Metric | Type | Meaning |
|--------|------|---------|
| `pass` | bool | All test cases met thresholds |
| `total_cases` | int | Number of test cases executed |
| `passed_cases` | int | Count that passed |
| `failed_cases` | int | Count that failed |
| `dataset` | str | Source or dataset name |
| `dataset_url` | str | Reference URL (if public) |

### Per-Case Metrics

```json
{
  "query": "female passengers first class",
  "top_k": 5,
  "observed_results": 5,           // Records actually retrieved
  "observed_citations": 5,          // Citations generated
  "expected_min_results": 1,        // Threshold: minimum required
  "expected_min_citations": 1,      // Threshold: minimum required
  "passed": true                   // met? Both observed >= expected
}
```

## Interpreting Pass/Fail

### What "Pass" Means ✓

1. Retrieved ≥ `min_results` documents
2. Generated citations meeting `min_citations` threshold
3. No crashes or exceptions
4. Deterministic thresholds met

### What "Pass" Does NOT Mean ✗

- Answer is semantically correct
- Retrieved sources are relevant
- Citations point to right sources
- Production-ready readiness

**Important:** Answer quality and source accuracy must be validated manually.

See [Citation Mismatch](./troubleshooting.md) for details.

## Reading metrics.csv

```text
run_id,timestamp,dataset,total_cases,passed_cases,failed_cases,pass
20260409-235448_public_titanic,2026-04-09T23:54:48,Titanic (datasciencedojo),3,3,0,True
20260409-235447_dataset,2026-04-09T23:54:47,Local CSV,3,3,0,True
```

**Trend analysis:**
```python
import pandas as pd
metrics = pd.read_csv(".benchmarks/history/metrics.csv")
print(metrics.tail(10))  # Last 10 runs
print(metrics["pass"].value_counts())  # Pass rate
```

## Benchmark Examples

### Example 1: Passed Run

```json
{
  "pass": true,
  "total_cases": 3,
  "passed_cases": 3,
  "failed_cases": 0
}
```

✓ All test cases met thresholds. Proceed with normal review and test gates.

### Example 2: Failed Run

```json
{
  "pass": false,
  "total_cases": 3,
  "passed_cases": 1,
  "failed_cases": 2,
  "results": [
    {
      "query": "rare query",
      "observed_results": 0,  // ← Failed: no results
      "expected_min_results": 1,
      "passed": false
    },
    // ...
  ]
}
```

✗ Some cases failed. Debug why expected results weren't retrieved.

## Debugging Failed Cases

```python
import json
from pathlib import Path

run_dir = Path(".benchmarks/runs/20260409-235448_public_titanic")
summary = json.loads((run_dir / "summary.json").read_text())

for result in summary["results"]:
  if not result["passed"]:
        print(f"Failed: {result['query']}")
        print(f"  Expected: {result['expected_min_results']} results")
        print(f"  Got: {result['observed_results']}")
```

**Remediation:**
1. Try the query manually: `engine.answer(query)`
2. Check embedding model coverage
3. Verify data was indexed
4. Lower threshold if test case is unrealistic

## Saving Benchmark Results

Benchmark runner scripts in this repository can persist outputs under `.benchmarks/runs/<timestamp>_*/` and aggregate history in `.benchmarks/history/metrics.csv`.

```bash
python -m ragsearch.evaluation \
  --engine-factory mymodule.build_engine \
  --cases eval_cases.json
```

The evaluation CLI itself prints the summary JSON to stdout; persistence is handled by the runner that invokes it.

## Next Steps

- Compare runs: `diff .benchmarks/runs/run1/summary.json .benchmarks/runs/run2/summary.json`
- Track trends: Graph `metrics.csv` over time
- Investigate failures: Read `notes.md` in each run
- Iterate: Update code, re-run, verify improvement

---

**See also:** [API Reference](./reference-api-cheat-sheet.md) | [Quickstart Guide](./quickstart.md) | [Troubleshooting](./troubleshooting.md)
