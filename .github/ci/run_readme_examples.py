"""
Release rehearsal helper (#109): run every python example in README.md, in order, against the
*installed* ragsearch package (e.g. 0.2.0rc1 from TestPyPI), with the real provider APIs.

Run it from a clone of the release tag (so samples/quickstart/ exists), in a virtual environment
where ragsearch was installed from the package index, never from the repository:

    COHERE_API_KEY=... python .github/ci/run_readme_examples.py

Examples that need something beyond a Cohere key are skipped unless it is available:
- OpenAI examples: need OPENAI_API_KEY (and `pip install openai`)
- LM Studio example: set RUN_LMSTUDIO=1 with an LM Studio server on localhost:1234 (and `pip install openai`)
- Ollama example: set RUN_OLLAMA=1 with an Ollama server running (and `pip install ollama`)
- ChromaDB example: currently broken (#76); it only runs with RUN_CHROMADB=1 and the chromadb extra
- `rag_engine.run()`: starts a server and blocks, so it is skipped (check it by hand)
"""

import os
import re
import sys
import traceback
from pathlib import Path


def main():
    import ragsearch

    if "site-packages" not in ragsearch.__file__:
        raise SystemExit(f"ragsearch must be the installed package, not the repository: {ragsearch.__file__}")
    import importlib.metadata

    print(f"ragsearch {importlib.metadata.version('ragsearch')} from {ragsearch.__file__}")
    if not os.environ.get("COHERE_API_KEY"):
        raise SystemExit("set COHERE_API_KEY")

    readme = Path("README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"^```python[^\n]*\n(.*?)^```", readme, re.DOTALL | re.MULTILINE)
    rules = [
        ("rag_engine.run()", lambda: False, "starts a blocking server; check by hand"),
        ("OPENAI_API_KEY", lambda: bool(os.environ.get("OPENAI_API_KEY")), "needs OPENAI_API_KEY"),
        ("localhost:1234", lambda: os.environ.get("RUN_LMSTUDIO") == "1", "needs RUN_LMSTUDIO=1 and LM Studio"),
        ('llm_provider="ollama"', lambda: os.environ.get("RUN_OLLAMA") == "1", "needs RUN_OLLAMA=1 and Ollama"),
        ("use_chromadb=True", lambda: os.environ.get("RUN_CHROMADB") == "1", "ChromaDB mode is broken (#76)"),
    ]
    namespace = {"__name__": "__main__"}
    failures = 0
    for number, block in enumerate(blocks, start=1):
        first = next(line for line in block.splitlines() if line.strip())
        skip = next((reason for marker, enabled, reason in rules if marker in block and not enabled()), None)
        if skip:
            print(f"[{number:2d}] SKIP  {first[:60]}  ({skip})")
            continue
        try:
            exec(compile(block, f"README.md example {number}", "exec"), namespace)
            print(f"[{number:2d}] OK    {first[:60]}")
        except Exception:
            failures += 1
            print(f"[{number:2d}] FAIL  {first[:60]}\n{traceback.format_exc(limit=3)}")
    print(f"{len(blocks)} examples, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
