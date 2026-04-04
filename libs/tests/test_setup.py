"""
Tests for empty-data handling in ragsearch.setup.
"""

from pathlib import Path

import pandas as pd
import pytest

from libs.ragsearch.errors import NoDataFoundError, ParseCorruptError, ParseTimeoutError, RagSearchError
from libs.ragsearch.parsers import ParsedDocument
from libs.ragsearch.parsers._fallback import FallbackParser
from libs.ragsearch.parsers._liteparse import LiteParseAdapter
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


def test_setup_uses_configured_embedding_provider_factory(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    class DummyEmbeddingModel:
        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    captured = {}

    def fake_create_embedding_model(provider, **kwargs):
        captured["provider"] = provider
        captured["model"] = kwargs.get("model")
        captured["api_key"] = kwargs.get("api_key")
        captured["base_url"] = kwargs.get("base_url")
        return DummyEmbeddingModel()

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.create_embedding_model", fake_create_embedding_model)
    monkeypatch.setattr("libs.ragsearch.setup.build_vector_backend", lambda **kwargs: object())
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(
        Path(data_path),
        llm_api_key="llm-key",
        embedding_provider="openai",
        embedding_model_name="text-embedding-3-small",
        embedding_api_key="embedding-key",
        embedding_base_url="https://api.openai.com/v1",
    )

    assert captured == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "embedding-key",
        "base_url": "https://api.openai.com/v1",
    }


def test_setup_raises_runtime_error_for_invalid_embedding_provider(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr(
        "libs.ragsearch.setup.create_embedding_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Unsupported embedding provider")),
    )

    with pytest.raises(RuntimeError, match="Failed to initialize embedding model"):
        setup(Path(data_path), llm_api_key="test-key", embedding_provider="invalid")


def test_setup_uses_configured_llm_provider_factory(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CohereClient should not be initialized for non-Cohere LLM providers")

    class DummyLLMClient:
        def generate(self, prompt, **kwargs):
            return "ok"

    class DummyEmbeddingModel:
        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    captured = {}

    def fake_create_llm_client(provider, **kwargs):
        captured["provider"] = provider
        captured["model"] = kwargs.get("model")
        captured["api_key"] = kwargs.get("api_key")
        captured["base_url"] = kwargs.get("base_url")
        return DummyLLMClient()

    def fake_create_embedding_model(provider, **kwargs):
        return DummyEmbeddingModel()

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["llm_client"] = kwargs["llm_client"]

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.create_llm_client", fake_create_llm_client)
    monkeypatch.setattr("libs.ragsearch.setup.create_embedding_model", fake_create_embedding_model)
    monkeypatch.setattr("libs.ragsearch.setup.build_vector_backend", lambda **kwargs: object())
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    setup(
        Path(data_path),
        llm_api_key="llm-key",
        llm_provider="openai",
        llm_model_name="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        embedding_provider="openai",
    )

    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "llm-key"
    assert captured["base_url"] == "https://api.openai.com/v1"
    assert isinstance(captured["llm_client"], DummyLLMClient)


def test_setup_raises_runtime_error_for_invalid_llm_provider(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CohereClient should not be initialized for invalid LLM provider config")

    class DummyEmbeddingModel:
        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.create_embedding_model", lambda *args, **kwargs: DummyEmbeddingModel())

    with pytest.raises(RuntimeError, match="Failed to initialize LLM client"):
        setup(Path(data_path), llm_api_key="test-key", llm_provider="invalid")


def test_setup_exposes_structured_ingestion_diagnostics(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    data_path.write_text("name,description\na,b\n", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.build_vector_backend", lambda **kwargs: object())
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    engine = setup(Path(data_path), llm_api_key="test-key")

    assert engine.ingestion_diagnostics == {
        "source_path": str(data_path),
        "selected_parser": "structured/pandas",
        "status": "success",
        "failure_reason": "",
        "indexing": {},
    }


def test_setup_unstructured_uses_fallback_when_liteparse_runtime_fails(tmp_path, monkeypatch, caplog):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("fallback parser content", encoding="utf-8")
    caplog.set_level("WARNING")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, texts):
            class Resp:
                embeddings = [[0.1, 0.2, 0.3]]

            return Resp()

    class DummyVectorDB:
        def __init__(self, *args, **kwargs):
            pass

    captured = {}

    class DummyEngine:
        def __init__(self, *args, **kwargs):
            captured["data"] = kwargs["data"]

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr("libs.ragsearch.setup.VectorDB", DummyVectorDB)
    monkeypatch.setattr("libs.ragsearch.setup.RagSearchEngine", DummyEngine)

    # Force LiteParse selection and runtime failure.
    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))

    def raise_liteparse_error(self, path):
        raise ParseTimeoutError("LiteParse timed out")

    monkeypatch.setattr(LiteParseAdapter, "parse", raise_liteparse_error)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda path: LiteParseAdapter())

    engine = setup(Path(data_path), llm_api_key="test-key")

    assert captured["data"].iloc[0]["text"] == "fallback parser content"
    assert captured["data"].iloc[0]["parser_name"] == "fallback/plain_text"
    assert "LiteParse parsing failed; using fallback parser" in caplog.text
    assert engine.ingestion_diagnostics == {
        "source_path": str(data_path),
        "selected_parser": "fallback",
        "status": "recovered_with_fallback",
        "failure_reason": "LiteParse timed out",
        "indexing": {},
    }


def test_setup_unstructured_reraises_primary_error_when_fallback_also_fails(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("input text", encoding="utf-8")

    class DummyCohereClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("libs.ragsearch.setup.CohereClient", DummyCohereClient)
    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))

    def raise_liteparse_error(self, path):
        raise ParseTimeoutError("LiteParse timed out")

    def raise_fallback_error(self, path):
        raise ParseCorruptError("Fallback failed")

    monkeypatch.setattr(LiteParseAdapter, "parse", raise_liteparse_error)
    monkeypatch.setattr(FallbackParser, "parse", raise_fallback_error)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda path: LiteParseAdapter())

    with pytest.raises(ParseTimeoutError, match="LiteParse timed out"):
        setup(Path(data_path), llm_api_key="test-key")


def test_setup_unstructured_fallback_primary_error_propagates(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.txt"
    data_path.write_text("input text", encoding="utf-8")

    class FailingFallbackParser(FallbackParser):
        def parse(self, path):
            raise ParseCorruptError("Fallback primary failed")

    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda path: FailingFallbackParser())

    with pytest.raises(ParseCorruptError, match="Fallback primary failed"):
        setup(Path(data_path), llm_api_key="test-key")


def test_setup_unstructured_reraises_liteparse_error_when_fallback_unsupported(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.png"
    data_path.write_text("binary-like placeholder", encoding="utf-8")

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))

    def raise_liteparse_error(self, path):
        raise ParseTimeoutError("LiteParse timed out")

    monkeypatch.setattr(LiteParseAdapter, "parse", raise_liteparse_error)
    monkeypatch.setattr("libs.ragsearch.setup.get_parser", lambda path: LiteParseAdapter())

    with pytest.raises(ParseTimeoutError, match="LiteParse timed out"):
        setup(Path(data_path), llm_api_key="test-key")


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