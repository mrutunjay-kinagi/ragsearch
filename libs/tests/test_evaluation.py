import pytest

from libs.ragsearch.evaluation import EvaluationThresholds, _run_cli, load_cases, run_regression_gates


class FakeEngine:
    def __init__(self, responses):
        self.responses = responses

    def answer(self, query: str, top_k: int = 5):
        return self.responses[query]


def build_pass_engine():
    return FakeEngine({"alpha": {"results": [{"id": 1}], "citations": [{"id": 1}]}})


def build_fail_engine():
    return FakeEngine({"alpha": {"results": [], "citations": []}})


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


def test_run_regression_gates_rejects_empty_cases():
    with pytest.raises(ValueError, match="at least one case"):
        run_regression_gates(build_pass_engine(), [], EvaluationThresholds())


def test_run_regression_gates_rejects_non_dict_case():
    with pytest.raises(ValueError, match="JSON object"):
        run_regression_gates(build_pass_engine(), ["alpha"], EvaluationThresholds())


def test_run_regression_gates_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k > 0"):
        run_regression_gates(
            build_pass_engine(),
            [{"query": "alpha", "top_k": 0}],
            EvaluationThresholds(),
        )


def test_run_regression_gates_rejects_negative_thresholds():
    with pytest.raises(ValueError, match="min_results must be >= 0"):
        run_regression_gates(
            build_pass_engine(),
            [{"query": "alpha", "top_k": 1}],
            EvaluationThresholds(min_results=-1, min_citations=0),
        )


def test_load_cases_rejects_non_list_payload(tmp_path):
    payload_path = tmp_path / "invalid_cases.json"
    payload_path.write_text('{"query":"alpha"}', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a list"):
        load_cases(payload_path)


def test_load_cases_rejects_empty_list(tmp_path):
    payload_path = tmp_path / "empty_cases.json"
    payload_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one case"):
        load_cases(payload_path)


def test_cli_returns_zero_for_passing_summary(tmp_path, capsys):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text('[{"query":"alpha","top_k":1}]', encoding="utf-8")

    rc = _run_cli(
        [
            "--engine-factory",
            "libs.tests.test_evaluation.build_pass_engine",
            "--cases",
            str(cases_path),
            "--summary-only",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "pass=True" in output


def test_cli_returns_nonzero_for_failing_summary(tmp_path, capsys):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text('[{"query":"alpha","top_k":1}]', encoding="utf-8")

    rc = _run_cli(
        [
            "--engine-factory",
            "libs.tests.test_evaluation.build_fail_engine",
            "--cases",
            str(cases_path),
            "--summary-only",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 1
    assert "pass=False" in output
