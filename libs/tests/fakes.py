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


def write_text_pdf(path, pages):
    """Write a minimal multi-page PDF (Helvetica text, one string per page) without extra dependencies."""

    def escape(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    page_ids = []
    for text in pages:
        lines = [text[i:i + 90] for i in range(0, len(text), 90)] or [""]
        ops = "BT /F1 10 Tf 12 TL 40 800 Td " + " ".join(f"({escape(line)}) Tj T*" for line in lines) + " ET"
        stream = ops.encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{ops}\nendstream")
        content_id = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(len(objects))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(page_ids)} >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("latin-1")
    path.write_bytes(bytes(out))
    return path
