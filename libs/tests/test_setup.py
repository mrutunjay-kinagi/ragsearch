"""
Tests for empty-data handling in ragsearch.setup.
"""

from pathlib import Path

import pandas as pd
import pytest

from libs.ragsearch.errors import NoDataFoundError, ParseTimeoutError, RagSearchError
from libs.ragsearch.parsers import ParsedDocument
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


def test_setup_structured_path_skips_parser(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    def fail_if_parser_called(*args, **kwargs):
        raise AssertionError("get_parser should not be called for structured inputs")

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", DummyVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", fail_if_parser_called)

    engine = setup(Path(data_path), llm_api_key="test-key")
    assert isinstance(engine, DummyEngine)


def test_setup_unstructured_path_uses_parser(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("hello", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            pass

    captured = {}

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["data"] = kwargs["data"]

    class FakeParser:
        def parse(self, path):
            return iter(
                [
                    ParsedDocument(
                        text="parsed text",
                        metadata={"source": "fixture"},
                        source_path=str(path),
                        parser_name="fallback/plain_text",
                    )
                ]
            )

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", DummyVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda *args, **kwargs: FakeParser())

    setup(Path(data_path), llm_api_key="test-key")

    assert not captured["data"].empty
    assert captured["data"].iloc[0]["text"] == "parsed text"


def test_setup_unstructured_raises_no_data_when_parser_returns_empty(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("hello", encoding="utf-8")

    class EmptyParser:
        def parse(self, path):
            return iter([])

    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda *args, **kwargs: EmptyParser())

    with pytest.raises(NoDataFoundError, match="No data found"):
        setup(Path(data_path), llm_api_key="test-key")


def test_setup_unstructured_filters_whitespace_documents(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("hello", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            pass

    captured = {}

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["data"] = kwargs["data"]

    class FakeParser:
        def parse(self, path):
            return iter(
                [
                    ParsedDocument(text="   ", metadata={}, source_path=str(path), parser_name="fake"),
                    ParsedDocument(text="kept", metadata={}, source_path=str(path), parser_name="fake"),
                ]
            )

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", DummyVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda *args, **kwargs: FakeParser())

    setup(Path(data_path), llm_api_key="test-key")

    assert captured["data"].shape[0] == 1
    assert captured["data"].iloc[0]["text"] == "kept"


def test_setup_unstructured_parser_timeout_propagates(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("hello", encoding="utf-8")

    class TimeoutParser:
        def parse(self, path):
            raise ParseTimeoutError("timed out")

    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda *args, **kwargs: TimeoutParser())

    with pytest.raises(ParseTimeoutError, match="timed out"):
        setup(Path(data_path), llm_api_key="test-key")


def test_setup_unstructured_all_whitespace_documents_raise_no_data(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("hello", encoding="utf-8")

    class WhitespaceParser:
        def parse(self, path):
            return iter(
                [
                    ParsedDocument(text="  ", metadata={}, source_path=str(path), parser_name="fake"),
                    ParsedDocument(text="\n\t", metadata={}, source_path=str(path), parser_name="fake"),
                ]
            )

    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda *args, **kwargs: WhitespaceParser())

    with pytest.raises(NoDataFoundError, match="No data found"):
        setup(Path(data_path), llm_api_key="test-key")


def test_setup_uses_embedding_dimension_from_model(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    captured = {}

    class CapturingVectorDB:
        def __init__(self, embedding_dim):
            captured["embedding_dim"] = embedding_dim

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", CapturingVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(Path(data_path), llm_api_key="test-key")

    assert captured["embedding_dim"] == 3


def test_setup_falls_back_to_legacy_dimension_when_probe_shape_is_invalid(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            return object()

    captured = {}

    class CapturingVectorDB:
        def __init__(self, embedding_dim):
            captured["embedding_dim"] = embedding_dim

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", CapturingVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(Path(data_path), llm_api_key="test-key")

    assert captured["embedding_dim"] == 4096


def test_setup_falls_back_to_legacy_dimension_when_probe_runtime_fails(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            raise RuntimeError("provider temporarily unavailable")

    captured = {}

    class CapturingVectorDB:
        def __init__(self, embedding_dim):
            captured["embedding_dim"] = embedding_dim

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", CapturingVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(Path(data_path), llm_api_key="test-key")

    assert captured["embedding_dim"] == 4096


def test_setup_uses_backend_factory_for_vector_backend(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    sentinel_backend = object()
    captured = {}

    def fake_build_vector_backend(*, embedding_dim):
        captured["embedding_dim"] = embedding_dim
        return sentinel_backend

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["vector_db"] = kwargs["vector_db"]

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.build_vector_backend", fake_build_vector_backend)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(Path(data_path), llm_api_key="test-key")

    assert captured["embedding_dim"] == 3
    assert captured["vector_db"] is sentinel_backend


def test_setup_wraps_llm_client_with_protocol_adapter(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    captured = {}

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["llm_client"] = kwargs["llm_client"]

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.build_vector_backend", lambda **kwargs: object())
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(Path(data_path), llm_api_key="test-key")

    assert hasattr(captured["llm_client"], "generate")