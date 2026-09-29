"""
Tests for parser boundary contracts and fallback behavior.
"""

from dataclasses import asdict
import builtins
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")

    def raise_timeout(*args, **kwargs):
        import subprocess

        raise subprocess.TimeoutExpired(cmd="liteparse", timeout=60)

    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", raise_timeout)

    with pytest.raises(ParseTimeoutError, match="timed out"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_adapter_timeout_configurable(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"text": "ok"})
        stderr = ""

    captured = {}

    def fake_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return FakeCompletedProcess()

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", fake_run)

    documents = list(LiteParseAdapter(timeout_s=7).parse(path))

    assert captured["timeout"] == 7
    assert len(documents) == 1


def test_liteparse_adapter_raises_for_empty_documents_list(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("ignored", encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0
        stdout = json.dumps({"documents": []})
        stderr = ""

    monkeypatch.setattr(LiteParseAdapter, "available", classmethod(lambda cls: True))
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.subprocess.run", lambda *args, **kwargs: FakeCompletedProcess())

    with pytest.raises(ParseCorruptError, match="did not contain documents"):
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")
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
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")

    parser = get_parser(path)

    assert isinstance(parser, LiteParseAdapter)


def test_liteparse_available_honors_env_cli_path(monkeypatch, tmp_path):
    cli_path = tmp_path / "liteparse-cli"
    cli_path.write_text("#!/bin/sh\necho ok", encoding="utf-8")

    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, str(cli_path))
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.shutil.which", lambda _: None)

    assert LiteParseAdapter.available() is True


def test_fallback_parser_empty_txt_file_yields_no_documents(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    documents = list(FallbackParser().parse(path))

    assert documents == []


def test_fallback_parser_docx_missing_raises_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "sample.docx"
    path.write_text("fake", encoding="utf-8")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("missing docx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ParserUnavailableError, match="pip install python-docx"):
        list(FallbackParser().parse(path))


def test_fallback_parser_html_missing_bs4_raises_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "sample.html"
    path.write_text("<p>hello</p>", encoding="utf-8")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bs4":
            raise ImportError("missing bs4")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ParserUnavailableError, match="pip install beautifulsoup4"):
        list(FallbackParser().parse(path))


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


def test_liteparse_is_not_selected_by_default_even_with_node_and_npx(monkeypatch, tmp_path):
    # The npm package the adapter used to call does not exist, and the real LiteParse CLI
    # has a different interface (#100), so LiteParse must be opted into explicitly.
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.delenv(LiteParseAdapter.ENV_CLI_PATH, raising=False)
    monkeypatch.setattr("libs.ragsearch.parsers._liteparse.shutil.which", lambda name: f"/usr/bin/{name}")

    assert LiteParseAdapter.available() is False
    assert isinstance(get_parser(path), FallbackParser)


def test_liteparse_without_configured_cli_raises_clear_unavailable_error(monkeypatch, tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")
    monkeypatch.delenv(LiteParseAdapter.ENV_CLI_PATH, raising=False)

    with pytest.raises(ParserUnavailableError, match="RAGSEARCH_LITEPARSE_CLI"):
        list(LiteParseAdapter().parse(path))


def test_liteparse_never_invokes_an_unpublished_npm_package(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    monkeypatch.setenv(LiteParseAdapter.ENV_CLI_PATH, "compatible-liteparse-cli")

    command = LiteParseAdapter()._build_command(path)

    assert command == ["compatible-liteparse-cli", "--json", str(path)]
    assert not any("npx" in part or "@run-llama" in part for part in command)


def test_fallback_parser_pdf_missing_pypdf_names_the_package_to_install(monkeypatch, tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("missing pypdf")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ParserUnavailableError, match="pip install pypdf"):
        list(FallbackParser().parse(path))


@pytest.mark.parametrize("module", ["pypdf", "docx", "bs4"])
def test_fallback_parser_dependencies_are_installed_with_ragsearch(module):
    # pypdf, python-docx and beautifulsoup4 are regular dependencies (#100).
    __import__(module)
