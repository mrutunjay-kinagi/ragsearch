"""
Tests for empty-data handling in ragsearch.setup.
"""

from pathlib import Path

import pytest

from libs.ragsearch.errors import NoDataFoundError, RagSearchError
from libs.ragsearch.setup import setup


def test_no_data_found_is_ragsearch_error():
    assert issubclass(NoDataFoundError, RagSearchError)


def test_setup_raises_no_data_found_for_empty_csv(tmp_path, monkeypatch):
    data_path = tmp_path / "empty.csv"
    data_path.write_text("name,description\n", encoding="utf-8")

    class FailIfCalledCohereClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CohereClient should not be initialized for empty data")

    class FailIfCalledVectorDB:
        def __init__(self, *args, **kwargs):
            raise AssertionError("VectorDB should not be initialized for empty data")

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", FailIfCalledCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", FailIfCalledVectorDB)

    with pytest.raises(NoDataFoundError, match="No data found"):
        setup(Path(data_path), llm_api_key="test-key")