"""
Default chunking for unstructured files (#77) and the answer context budget (#88),
exercised through the public setup() / search() / answer() path with offline fakes.
"""

import importlib
import json
import logging
import math
from pathlib import Path

import pytest

from libs.ragsearch.chunking import (
    FixedWordChunkingStrategy,
    RowChunkingStrategy,
    default_unstructured_chunking_strategy,
    describe_chunking_strategy,
)
from libs.ragsearch.engine import DEFAULT_MAX_CONTEXT_TOKENS, estimate_tokens
from libs.ragsearch.setup import setup
from libs.tests.fakes import CannedLLMClient, KeywordEmbeddingModel, write_text_pdf

# Patch the module object: ``libs.ragsearch.setup`` as an attribute path is the setup() function (#91).
ragsearch_setup_module = importlib.import_module("libs.ragsearch.setup")

FILLER = "The committee reviewed access control policy and authentication records for the quarter. " * 25
LATE_FACT = "The harbour inspection found a cracked mooring bollard on pier seven."


@pytest.fixture
def offline_providers(monkeypatch):
    """Replace the provider factories used by setup() with deterministic fakes."""
    llm = CannedLLMClient("Canned answer [1].")
    monkeypatch.setattr(ragsearch_setup_module, "CohereClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        ragsearch_setup_module,
        "create_embedding_model",
        lambda **kwargs: KeywordEmbeddingModel(["harbour", "bollard", "mooring", "committee", "authentication"]),
    )
    monkeypatch.setattr(ragsearch_setup_module, "create_llm_client", lambda **kwargs: llm)
    return llm


@pytest.fixture
def long_pdf(tmp_path):
    pages = [f"Page {number}. {FILLER}" for number in range(1, 13)] + [LATE_FACT]
    return write_text_pdf(tmp_path / "report.pdf", pages)


def _setup(path, tmp_path, **kwargs):
    return setup(Path(path), llm_api_key="offline-key", embeddings_dir=str(tmp_path / "cache"), **kwargs)


# --- #77: default chunking ---------------------------------------------------------------


def test_setup_chunks_unstructured_files_by_default(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path)

    assert describe_chunking_strategy(engine.chunking_strategy) == describe_chunking_strategy(
        default_unstructured_chunking_strategy()
    )
    assert len(engine.index_data) > 5, "a 13-page document must produce several chunks"
    assert list(engine.index_data["chunk_index"]) == list(range(len(engine.index_data)))


def test_setup_keeps_row_chunking_for_structured_files(tmp_path, offline_providers):
    data_path = tmp_path / "claims.csv"
    data_path.write_text("claim_id,description\nC1,harbour damage\nC2,committee review\n", encoding="utf-8")

    engine = _setup(data_path, tmp_path)

    assert isinstance(engine.chunking_strategy, RowChunkingStrategy)
    assert len(engine.index_data) == 2


def test_setup_honours_an_explicit_chunking_strategy(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path, chunking_strategy=RowChunkingStrategy())

    assert len(engine.index_data) == 1


def test_question_about_late_document_content_retrieves_the_right_chunk(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path)

    top = engine.search("Where was the cracked mooring bollard found?", top_k=1)[0]

    assert "mooring bollard on pier seven" in engine._result_text(top["metadata"])
    assert top["metadata"]["chunk_index"] == len(engine.index_data) - 1


# --- cache identity includes chunking --------------------------------------------------------


def test_changing_chunking_settings_triggers_reindex_with_warning(long_pdf, tmp_path, offline_providers, caplog):
    _setup(long_pdf, tmp_path, chunking_strategy=FixedWordChunkingStrategy(words_per_chunk=300, overlap_words=30))

    with caplog.at_level(logging.WARNING):
        engine = _setup(long_pdf, tmp_path, chunking_strategy=FixedWordChunkingStrategy(words_per_chunk=200, overlap_words=40))

    reason = engine.indexing_diagnostics["cache_invalidated_reason"]
    assert "chunking changed" in reason and "words_per_chunk=300" in reason and "words_per_chunk=200" in reason
    assert engine.indexing_diagnostics["reused_records"] == 0
    assert any("One-time re-embedding" in record.getMessage() for record in caplog.records)


def test_unchanged_chunking_reuses_the_cache(long_pdf, tmp_path, offline_providers, caplog):
    _setup(long_pdf, tmp_path)

    with caplog.at_level(logging.WARNING):
        engine = _setup(long_pdf, tmp_path)

    assert engine.indexing_diagnostics["embedded_records"] == 0
    assert engine.indexing_diagnostics["reused_records"] == len(engine.index_data)
    assert not [record for record in caplog.records if "re-embedding" in record.getMessage()]


def test_manifest_records_the_chunking_identity(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path)

    manifest = json.loads(Path(engine.indexing_diagnostics["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["chunking"] == describe_chunking_strategy(default_unstructured_chunking_strategy())


def test_cache_without_chunking_metadata_is_treated_as_row_chunking(tmp_path, offline_providers):
    # Manifests written before chunking was recorded (ragsearch <= 0.2.0.dev) always used row chunking.
    data_path = tmp_path / "claims.csv"
    data_path.write_text("claim_id,description\nC1,harbour damage\nC2,committee review\n", encoding="utf-8")
    engine = _setup(data_path, tmp_path)
    manifest_path = Path(engine.indexing_diagnostics["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("chunking")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    engine = _setup(data_path, tmp_path)

    assert engine.indexing_diagnostics["cache_invalidated_reason"] == ""
    assert engine.indexing_diagnostics["reused_records"] == 2


# --- #88: context budget ---------------------------------------------------------------------


def test_answer_trims_context_to_the_budget_in_score_order(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path)

    response = engine.answer("committee authentication policy", top_k=10, max_context_tokens=900)

    assert response["context_truncated"] is True
    assert 0 < response["context_sources"] < len(response["results"])
    assert response["context_tokens_estimate"] <= 900
    assert estimate_tokens(response["context"]) <= 900
    # Only the sources actually sent to the LLM are cited, in score order.
    assert response["citations"] == [r["citation"] for r in response["results"][: response["context_sources"]]]
    assert f"[{response['context_sources']}]" in response["context"]
    assert f"[{response['context_sources'] + 1}]" not in response["context"]
    assert response["context"] in offline_providers.prompts[-1]


def test_answer_within_budget_is_not_truncated(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path)

    response = engine.answer("Where was the cracked mooring bollard found?", top_k=3)

    assert response["context_truncated"] is False
    assert response["context_sources"] == len(response["results"]) == 3
    assert response["citations"] == [r["citation"] for r in response["results"]]
    assert response["context_tokens_estimate"] == estimate_tokens(response["context"])


def test_single_oversized_source_is_cut_to_the_budget(long_pdf, tmp_path, offline_providers):
    engine = _setup(long_pdf, tmp_path, chunking_strategy=RowChunkingStrategy())

    response = engine.answer("committee", top_k=1, max_context_tokens=300)

    assert response["context_truncated"] is True
    assert response["context_sources"] == 1
    assert estimate_tokens(response["context"]) <= 300


def test_setup_passes_the_default_budget_and_accepts_an_override(long_pdf, tmp_path, offline_providers):
    assert _setup(long_pdf, tmp_path).max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS
    assert _setup(long_pdf, tmp_path, max_context_tokens=1234).max_context_tokens == 1234
    assert _setup(long_pdf, tmp_path, max_context_tokens=None).max_context_tokens is None


def test_estimate_tokens_errs_on_the_high_side():
    # ~3.5 characters per token; measured 3.55-4.87 on real documents with Cohere's tokenizer.
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 35) == 10
    assert estimate_tokens("x" * 36) == math.ceil(36 / 3.5)


# --- DOCX regression (#83) ---------------------------------------------------------------------


def test_docx_table_content_is_not_indexed_yet(tmp_path, offline_providers):
    """Records current behaviour: DOCX tables are skipped by the built-in parser (#83).

    Update this test when #83 lands; the claim document 'Neuroscience Center of Buffalo.docx'
    keeps its content in tables and currently yields only its paragraph text.
    """
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Neuroscience Center intake form.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Diagnosis"
    table.rows[0].cells[1].text = "Lumbar radiculopathy"
    path = tmp_path / "intake.docx"
    document.save(str(path))

    engine = _setup(path, tmp_path)

    indexed = " ".join(engine.index_data["combined_text"])
    assert "Neuroscience Center intake form." in indexed
    assert "Lumbar radiculopathy" not in indexed
