"""
Tests for RagSearchEngine.run(): localhost by default, host/port parameters, and the HTTP contracts (#104).

The server thread is replaced by a recorder, so no socket is opened; the Flask app that run()
builds is exercised through Flask's test client.
"""

import importlib
import logging
from pathlib import Path

import pytest

from libs.ragsearch.setup import setup
from libs.tests.fakes import CannedLLMClient, KeywordEmbeddingModel

# Patch the module objects: ``libs.ragsearch.setup`` as an attribute path is the setup() function (#91).
ragsearch_setup_module = importlib.import_module("libs.ragsearch.setup")
ragsearch_engine_module = importlib.import_module("libs.ragsearch.engine")


class RecordingThread:
    """Stands in for threading.Thread: records how run() starts the server, without starting it."""

    started = []

    def __init__(self, target=None, kwargs=None, **_):
        self.target = target
        self.kwargs = kwargs or {}

    def start(self):
        RecordingThread.started.append(self)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(ragsearch_setup_module, "CohereClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(ragsearch_setup_module, "create_embedding_model", lambda **kwargs: KeywordEmbeddingModel(["harbour", "claim"]))
    monkeypatch.setattr(ragsearch_setup_module, "create_llm_client", lambda **kwargs: CannedLLMClient("Canned answer [1]."))
    monkeypatch.setattr(ragsearch_engine_module.threading, "Thread", RecordingThread)
    RecordingThread.started = []
    data_path = tmp_path / "claims.csv"
    data_path.write_text("claim_id,description\nC1,harbour claim\nC2,roof claim\n", encoding="utf-8")
    return setup(Path(data_path), llm_api_key="offline-key", embeddings_dir=str(tmp_path / "cache"))


def _started_server():
    assert len(RecordingThread.started) == 1
    return RecordingThread.started[0]


def test_run_listens_on_localhost_port_8080_by_default(engine):
    engine.run()

    server = _started_server()
    assert server.kwargs["host"] == "127.0.0.1"
    assert server.kwargs["port"] == 8080
    assert server.kwargs["use_reloader"] is False


def test_run_accepts_host_and_port(engine):
    engine.run(host="0.0.0.0", port=9001)

    server = _started_server()
    assert (server.kwargs["host"], server.kwargs["port"]) == ("0.0.0.0", 9001)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_run_warns_when_binding_beyond_localhost(engine, caplog, host):
    with caplog.at_level(logging.WARNING):
        engine.run(host=host)

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("no authentication" in message and host in message for message in warnings), warnings


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_run_does_not_warn_for_loopback_hosts(engine, caplog, host):
    with caplog.at_level(logging.WARNING):
        engine.run(host=host)

    assert not [record for record in caplog.records if "no authentication" in record.getMessage()]


@pytest.mark.parametrize("port", [0, 65536, -1, "8080"])
def test_run_rejects_invalid_ports(engine, port):
    with pytest.raises(ValueError, match="port"):
        engine.run(port=port)

    assert RecordingThread.started == []


def test_http_query_and_answer_contracts_are_unchanged(engine):
    engine.run()
    app = _started_server().target.__self__
    client = app.test_client()

    query = client.post("/query", json={"query": "harbour claim", "top_k": 1})
    assert query.status_code == 200
    assert set(query.get_json()) == {"results"}
    assert query.get_json()["results"][0]["claim_id"] == "C1"

    detailed = client.post("/query", json={"query": "harbour claim", "top_k": 1, "include_details": True})
    assert {"metadata", "citation", "similarity"} <= set(detailed.get_json()["results"][0])

    answer = client.post("/answer", json={"query": "harbour claim", "top_k": 1})
    assert answer.status_code == 200
    assert {"question", "answer", "results", "citations", "context"} <= set(answer.get_json())
    assert answer.get_json()["answer"] == "Canned answer [1]."

    assert client.post("/query", json={}).status_code == 400
    assert client.post("/answer", json={}).status_code == 400
