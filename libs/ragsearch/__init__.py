"""
ragsearch is a Python library designed for building a
Retrieval-Augmented Generation (RAG) application
that enables natural language querying over structured data.
"""

from typing import TYPE_CHECKING

__all__ = [
    "setup",
    "RagSearchEngine",
]


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
