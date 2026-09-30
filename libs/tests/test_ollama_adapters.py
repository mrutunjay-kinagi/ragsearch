"""
Ollama adapters against the response types the ollama client actually returns (#129).

The ollama client returns typed pydantic responses (e.g. ``ollama.EmbedResponse``) that are not
dicts; older clients and raw HTTP use plain dicts. These tests feed the adapters both forms, and
run setup() through the real ``ollama.Client`` against a local stub server, so response-type drift
is caught in CI without a real Ollama server.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ollama = pytest.importorskip("ollama")

from libs.ragsearch.embedding_models import OllamaEmbeddingAdapter, extract_embeddings  # noqa: E402
from libs.ragsearch.llm_clients import OllamaLLMClientAdapter  # noqa: E402
from libs.ragsearch.setup import setup  # noqa: E402


class _EmbedClient:
    """Fake client whose embed() returns a fixed payload."""

    def __init__(self, payload):
        self.payload = payload

    def embed(self, model, input):
        return self.payload


class _LegacyEmbeddingsClient:
    """Fake client with only the older one-text-at-a-time embeddings() API."""

    def __init__(self, make_payload):
        self.make_payload = make_payload

    def embeddings(self, model, prompt):
        return self.make_payload([float(len(prompt)), 1.0])


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(ollama.EmbedResponse(model="nomic-embed-text", embeddings=[[0.1, 0.2], [0.3, 0.4]]), id="typed-EmbedResponse"),
        pytest.param({"model": "nomic-embed-text", "embeddings": [[0.1, 0.2], [0.3, 0.4]]}, id="dict"),
    ],
)
def test_embed_reads_typed_and_dict_responses(payload):
    adapter = OllamaEmbeddingAdapter(client=_EmbedClient(payload))

    assert extract_embeddings(adapter.embed(["a", "b"])) == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.parametrize(
    "make_payload",
    [
        pytest.param(lambda vector: ollama.EmbeddingsResponse(embedding=vector), id="typed-EmbeddingsResponse"),
        pytest.param(lambda vector: {"embedding": vector}, id="dict"),
    ],
)
def test_legacy_embeddings_api_reads_typed_and_dict_responses(make_payload):
    adapter = OllamaEmbeddingAdapter(client=_LegacyEmbeddingsClient(make_payload))

    assert extract_embeddings(adapter.embed(["ab", "abcd"])) == [[2.0, 1.0], [4.0, 1.0]]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(ollama.ChatResponse(model="llama3.1", message=ollama.Message(role="assistant", content="hi [1]")), id="typed-ChatResponse"),
        pytest.param({"message": {"role": "assistant", "content": "hi [1]"}}, id="dict"),
    ],
)
def test_chat_reads_typed_and_dict_responses(response):
    class _ChatClient:
        def chat(self, model, messages, **kwargs):
            return response

    assert OllamaLLMClientAdapter(client=_ChatClient()).generate("hello") == "hi [1]"


class _StubOllamaHandler(BaseHTTPRequestHandler):
    """Minimal Ollama HTTP API: /api/embed and /api/chat, as documented by Ollama."""

    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        _StubOllamaHandler.requests.append((self.path, body.get("model")))
        if self.path == "/api/embed":
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            reply = {"model": body["model"], "embeddings": [[float(len(text) % 7 + 1), 1.0, 0.5] for text in inputs]}
        elif self.path == "/api/chat":
            reply = {
                "model": body["model"],
                "created_at": "2026-09-30T00:00:00Z",
                "done": True,
                "message": {"role": "assistant", "content": "Stub answer [1]."},
            }
        else:
            self.send_error(404)
            return
        data = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_ollama_server():
    _StubOllamaHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_fully_local_ollama_setup_through_the_real_client(stub_ollama_server, tmp_path):
    data_path = tmp_path / "claims.csv"
    data_path.write_text("claim_id,description\nC1,windshield claim denied\nC2,roof hail damage\n", encoding="utf-8")

    engine = setup(
        Path(data_path),
        "not-used-by-ollama",
        llm_provider="ollama",
        llm_model_name="llama3.1",
        llm_base_url=stub_ollama_server,
        embedding_provider="ollama",
        embedding_model_name="nomic-embed-text",
        embedding_base_url=stub_ollama_server,
        embeddings_dir=str(tmp_path / "cache"),
    )
    response = engine.answer("Why was the windshield claim denied?", top_k=1)

    assert response["answer"] == "Stub answer [1]."
    assert ("/api/embed", "nomic-embed-text") in _StubOllamaHandler.requests
    assert ("/api/chat", "llama3.1") in _StubOllamaHandler.requests
