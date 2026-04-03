"""
Tests for parser boundary contracts and fallback behavior.
"""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from libs.ragsearch.errors import (
    ParseCorruptError,
    ParseTimeoutError,
    ParsingError,
    ParserUnavailableError,
    RagSearchError,
    UnsupportedFileTypeError,
)
from libs.ragsearch.parsers import DocumentParser, FallbackParser, LiteParseAdapter, ParsedDocument, get_parser


class DuckParser:
    def supports(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def parse(self, path: Path):
        yield ParsedDocument(text=path.read_text(encoding="utf-8"), metadata={"parser": "duck"})


def test_parsed_document_can_be_serialized():
    parsed = ParsedDocument(text="hello", metadata={"source": "fixture"})
    data = asdict(parsed)

    assert data["text"] == "hello"
    assert data["metadata"] == {"source": "fixture"}
    assert data["source_path"] == ""
    assert data["parser_name"] == ""


def test_duck_typed_parser_satisfies_protocol():
    parser = DuckParser()

    assert isinstance(parser, DocumentParser)
    assert parser.supports(Path("note.txt")) is True


def test_duck_typed_parser_uses_text_fixture(tmp_path):
    parser = DuckParser()
    path = tmp_path / "note.txt"
    path.write_text("duck parser fixture", encoding="utf-8")

    assert next(parser.parse(path)).metadata == {"parser": "duck"}


def test_parsing_error_is_subclass_of_ragsearch_error():
    assert issubclass(ParsingError, RagSearchError)


def test_fallback_parser_raises_for_none_path():
    with pytest.raises(ParseCorruptError, match="cannot be None"):
        list(FallbackParser().parse(None))


def test_fallback_parser_raises_for_unsupported_extension(tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text("unknown", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        list(FallbackParser().parse(path))


def test_fallback_parser_reads_text_fixture(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello fallback", encoding="utf-8")

    documents = list(FallbackParser().parse(path))

    assert len(documents) == 1
    assert documents[0].text == "hello fallback"
    assert documents[0].parser_name == "fallback/plain_text"


def test_liteparse_adapter_returns_documents_for_mocked_subprocess(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored by mock", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps(
            {
                "text": "liteparse text",
                "metadata": {"pages": 1},
                "source_path": str(path),
                "parser_name": "liteparse",
            }
        )
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    documents = list(LiteParseAdapter().parse(path))

    assert len(documents) == 1
    assert documents[0].text == "liteparse text"
    assert documents[0].metadata == {"pages": 1}
    assert documents[0].parser_name == "liteparse"


def test_liteparse_adapter_raises_unavailable_when_node_missing(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: False))

    with pytest.raises(ParserUnavailableError, match="LiteParse CLI not found"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_timeout(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))

    def raise_timeout(*args, **kwargs):
        import subprocess

        raise subprocess.TimeoutExpired(cmd="liteparse", timeout=60)

    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", raise_timeout)

    with pytest.raises(ParseTimeoutError, match="timed out"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_invalid_document_shapes(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps(
            {
                "documents": [
                    {"text": 123, "metadata": {}},
                ]
            }
        )
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="invalid text"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_invalid_metadata_shape(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps(
            {
                "documents": [
                    {"text": "ok", "metadata": ["not", "a", "dict"]},
                ]
            }
        )
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="invalid metadata"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_non_zero_exit(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="LiteParse failed"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_invalid_json(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = "not-json"
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="not valid JSON"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_missing_documents_list(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"summary": "no documents"})
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="did not contain documents"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_raises_for_invalid_document_entry(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"documents": ["bad-entry"]})
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="invalid document entry"):
        list(LiteParseAdapter().parse(path))


def test_get_parser_raises_for_none_path():
    with pytest.raises(UnsupportedFileTypeError, match="got None"):
        get_parser(None)


def test_get_parser_prefers_liteparse_when_available(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))

    parser = get_parser(path)

    assert isinstance(parser, LiteParseAdapter)


def test_get_parser_falls_back_when_liteparse_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: False))

    parser = get_parser(path)

    assert isinstance(parser, FallbackParser)


def test_get_parser_raises_for_unknown_extension(tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        get_parser(path)
