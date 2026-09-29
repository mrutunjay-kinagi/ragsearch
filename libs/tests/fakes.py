"""
Deterministic offline stand-ins for embedding and LLM providers.

These used to live in docs/quickstart.md. The quickstart now uses the real
setup() path, and these fakes let tests run the documented examples
without network access or API keys.
"""


class FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class KeywordEmbeddingModel:
    """Keyword-count embedding: one dimension per keyword, plus a constant bias dimension."""

    def __init__(self, keywords):
        self.keywords = [keyword.lower() for keyword in keywords]
        self.model_name = "keyword-" + "-".join(self.keywords)

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            vectors.append([0.1] + [float(lowered.count(keyword)) for keyword in self.keywords])
        return FakeEmbeddingResponse(vectors)


class CannedLLMClient:
    """Returns a fixed answer and records every prompt it receives."""

    def __init__(self, answer="Canned answer grounded in the retrieved sources [1]."):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answer
