from libs.ragsearch.evaluation import EvaluationThresholds, run_regression_gates


class FakeEngine:
    def __init__(self, responses):
        self.responses = responses

    def answer(self, query: str, top_k: int = 5):
        return self.responses[query]


def test_run_regression_gates_passes_when_thresholds_are_met():
    engine = FakeEngine(
        {
            "alpha": {
                "results": [{"id": 1}],
                "citations": [{"id": 1}],
            }
        }
    )

    summary = run_regression_gates(
        engine,
        [{"query": "alpha", "top_k": 1}],
        EvaluationThresholds(min_results=1, min_citations=1),
    )

    assert summary["pass"] is True
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 0


def test_run_regression_gates_fails_when_thresholds_not_met():
    engine = FakeEngine(
        {
            "alpha": {
                "results": [],
                "citations": [],
            }
        }
    )

    summary = run_regression_gates(
        engine,
        [{"query": "alpha", "top_k": 1}],
        EvaluationThresholds(min_results=1, min_citations=1),
    )

    assert summary["pass"] is False
    assert summary["passed_cases"] == 0
    assert summary["failed_cases"] == 1
