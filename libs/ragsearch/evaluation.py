"""Evaluation harness baseline for deterministic regression gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationThresholds:
    min_results: int = 1
    min_citations: int = 1


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation cases file must contain a list")
    return payload


def run_regression_gates(engine, cases: list[dict[str, Any]], thresholds: EvaluationThresholds) -> dict[str, Any]:
    results = []
    for case in cases:
        query = str(case.get("query", "")).strip()
        top_k = int(case.get("top_k", 5))
        if not query:
            raise ValueError("each evaluation case must include a non-empty query")

        expected_min_results = int(case.get("min_results", thresholds.min_results))
        expected_min_citations = int(case.get("min_citations", thresholds.min_citations))

        payload = engine.answer(query, top_k=top_k)
        observed_results = len(payload.get("results", []))
        observed_citations = len(payload.get("citations", []))
        passed = observed_results >= expected_min_results and observed_citations >= expected_min_citations

        results.append(
            {
                "query": query,
                "top_k": top_k,
                "observed_results": observed_results,
                "observed_citations": observed_citations,
                "expected_min_results": expected_min_results,
                "expected_min_citations": expected_min_citations,
                "passed": passed,
            }
        )

    passed_cases = sum(1 for item in results if item["passed"])
    return {
        "total_cases": len(results),
        "passed_cases": passed_cases,
        "failed_cases": len(results) - passed_cases,
        "pass": passed_cases == len(results),
        "results": results,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Run ragsearch evaluation regression gates")
    parser.add_argument("--cases", required=True, help="Path to JSON file with evaluation cases")
    parser.add_argument("--min-results", type=int, default=1, help="Default minimum expected retrieval results")
    parser.add_argument("--min-citations", type=int, default=1, help="Default minimum expected citations")
    parser.add_argument("--summary-only", action="store_true", help="Print only pass/fail summary")
    args = parser.parse_args()

    raise RuntimeError(
        "CLI execution requires an application-specific engine bootstrap. "
        "Import run_regression_gates from libs.ragsearch.evaluation in project scripts/tests."
    )


if __name__ == "__main__":
    raise SystemExit(_cli())
