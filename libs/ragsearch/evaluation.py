"""Evaluation harness baseline for deterministic regression gates."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationThresholds:
    min_results: int = 1
    min_citations: int = 1


def _validate_thresholds(thresholds: EvaluationThresholds) -> None:
    if thresholds.min_results < 0:
        raise ValueError("min_results must be >= 0")
    if thresholds.min_citations < 0:
        raise ValueError("min_citations must be >= 0")


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation cases file must contain a list")
    if len(payload) == 0:
        raise ValueError("evaluation cases file must contain at least one case")
    return payload


def run_regression_gates(engine, cases: list[dict[str, Any]], thresholds: EvaluationThresholds) -> dict[str, Any]:
    _validate_thresholds(thresholds)
    if len(cases) == 0:
        raise ValueError("evaluation cases must contain at least one case")

    results = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each evaluation case must be a JSON object")

        query = str(case.get("query", "")).strip()
        top_k = int(case.get("top_k", 5))
        if not query:
            raise ValueError("each evaluation case must include a non-empty query")
        if top_k <= 0:
            raise ValueError("each evaluation case must include top_k > 0")

        expected_min_results = int(case.get("min_results", thresholds.min_results))
        expected_min_citations = int(case.get("min_citations", thresholds.min_citations))
        if expected_min_results < 0:
            raise ValueError("min_results must be >= 0")
        if expected_min_citations < 0:
            raise ValueError("min_citations must be >= 0")

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


def _load_engine_factory(path: str):
    if "." not in path:
        raise ValueError("engine factory must be a dotted path, e.g. module.factory")
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise ValueError("engine factory path must point to a callable")
    return factory


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ragsearch evaluation regression gates")
    parser.add_argument(
        "--engine-factory",
        required=True,
        help="Dotted path to callable that returns an engine (e.g. mymodule.build_engine)",
    )
    parser.add_argument("--cases", required=True, help="Path to JSON file with evaluation cases")
    parser.add_argument("--min-results", type=int, default=1, help="Default minimum expected retrieval results")
    parser.add_argument("--min-citations", type=int, default=1, help="Default minimum expected citations")
    parser.add_argument("--summary-only", action="store_true", help="Print only pass/fail summary")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases))
    thresholds = EvaluationThresholds(
        min_results=int(args.min_results),
        min_citations=int(args.min_citations),
    )

    engine_factory = _load_engine_factory(args.engine_factory)
    engine = engine_factory()
    summary = run_regression_gates(engine, cases, thresholds)

    if args.summary_only:
        print(
            f"pass={summary['pass']} total_cases={summary['total_cases']} "
            f"passed_cases={summary['passed_cases']} failed_cases={summary['failed_cases']}"
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
