"""
ragsearch: Retrieval-Augmented Generation over CSV/JSON/Parquet files and documents (PDF, DOCX,
HTML, Markdown, text).

``setup()`` loads and indexes a file (FAISS by default); the returned engine's ``search()`` returns
cited results and ``answer()`` asks an LLM for an answer grounded in, and citing, the retrieved
sources. Ingestion and indexing diagnostics and an evaluation harness (``ragsearch.evaluation``) are
included.

    from ragsearch import setup
"""

import sys
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "setup",
    "RagSearchEngine",
]


class _RagSearchPackage(ModuleType):
    """Package module that keeps ``ragsearch.setup`` bound to the ``setup()`` function.

    The ``setup`` submodule and the ``setup`` function share a name. Importing a
    submodule rebinds the package attribute to the module, which made
    ``from ragsearch import setup`` return the module (#91). Intercepting that
    rebinding keeps the public name pointing at the function while the heavy
    imports stay lazy. ``from ragsearch.setup import setup`` is unaffected.
    """

    def __setattr__(self, name, value):
        if name == "setup" and isinstance(value, ModuleType):
            value = value.setup
        super().__setattr__(name, value)


# Supported since Python 3.5: https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
sys.modules[__name__].__class__ = _RagSearchPackage


def __getattr__(name):
    if name == "setup":
        from .setup import setup

        return setup
    if name == "RagSearchEngine":
        from .engine import RagSearchEngine

        return RagSearchEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .engine import RagSearchEngine
    from .setup import setup
