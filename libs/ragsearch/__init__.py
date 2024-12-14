"""
ragsearch is a Python library designed for building a
Retrieval-Augmented Generation (RAG) application
that enables natural language querying over structured data.
"""
from .setup import setup
from .engine import RagSearchEngine

__all__ = [
    "setup",
    "RagSearchEngine",
]
