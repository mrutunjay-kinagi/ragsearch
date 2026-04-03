"""
Tests for empty-data handling in ragsearch.setup.
"""

from pathlib import Path

import pandas as pd
import pytest

from libs.ragsearch.errors import NoDataFoundError, RagSearchError
from libs.ragsearch.engine import RagSearchEngine
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


def test_engine_raises_no_data_found_for_empty_dataframe(tmp_path, monkeypatch):
    data = pd.DataFrame()

    class DummyEmbeddingModel:
        def embed(self, *args, **kwargs):
            raise AssertionError("embed should not be called for empty data")

    class DummyLLMClient:
        pass

    def fail_if_mkdir_called(*args, **kwargs):
        raise AssertionError("mkdir should not be called for empty data")

    monkeypatch.setattr("libs.ragsearch.engine.Path.mkdir", fail_if_mkdir_called)

    with pytest.raises(NoDataFoundError, match="No data found"):
        RagSearchEngine(
            data=data,
            embedding_model=DummyEmbeddingModel(),
            llm_client=DummyLLMClient(),
            vector_db=None,
            save_dir=str(tmp_path / "embeddings"),
        )