"""
Offline checks for the documented examples (docs/quickstart.md).

The quickstart script is extracted from the Markdown and run exactly as
written, with the provider factories replaced by deterministic fakes, so
the documented setup() -> answer() flow is exercised in CI without keys.
"""

import importlib
import re
from pathlib import Path

from libs.tests.fakes import CannedLLMClient, KeywordEmbeddingModel

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
SAMPLE_CSV = REPO_ROOT / "samples" / "quickstart" / "insurance_claims.csv"

_PYTHON_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _quickstart_script() -> str:
    blocks = _PYTHON_BLOCK.findall(QUICKSTART.read_text(encoding="utf-8"))
    assert len(blocks) == 1, "docs/quickstart.md should contain exactly one python block"
    return blocks[0]


def test_quickstart_inline_sample_matches_shipped_sample_file():
    namespace = {}
    match = re.search(r'SAMPLE_CSV = """\\\n(.*?)"""', _quickstart_script(), re.DOTALL)
    assert match, "quickstart script must define SAMPLE_CSV"
    exec(f'SAMPLE_CSV = """{match.group(1)}"""', namespace)

    assert namespace["SAMPLE_CSV"] == SAMPLE_CSV.read_text(encoding="utf-8")


def test_quickstart_script_runs_as_written_offline(tmp_path, monkeypatch, capsys):
    # The script imports the public package name, exactly as a user would.
    setup_module = importlib.import_module("ragsearch.setup")
    llm = CannedLLMClient("Claim CLM-1003 was denied because the policy has no glass coverage [1].")
    monkeypatch.setattr(setup_module, "CohereClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        setup_module,
        "create_embedding_model",
        lambda **kwargs: KeywordEmbeddingModel(["windshield", "glass", "denied", "water"]),
    )
    monkeypatch.setattr(setup_module, "create_llm_client", lambda **kwargs: llm)
    monkeypatch.setenv("COHERE_API_KEY", "offline-test-key")
    monkeypatch.chdir(tmp_path)

    exec(compile(_quickstart_script(), str(QUICKSTART), "exec"), {"__name__": "__main__"})

    output = capsys.readouterr().out
    assert "Indexed 8 claims." in output
    assert "Answer: Claim CLM-1003 was denied" in output
    assert "[1] record 2: " in output and "CLM-1003" in output
    assert "CLM-1003" in llm.prompts[0]
    assert (tmp_path / "insurance_claims.csv").read_text(encoding="utf-8") == SAMPLE_CSV.read_text(encoding="utf-8")
