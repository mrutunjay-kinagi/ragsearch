"""
Document parser boundary for ragsearch.
"""

from ._dispatch import get_parser
from ._fallback import FallbackParser
from ._liteparse import LiteParseAdapter
from ._models import ParsedDocument
from ._protocol import DocumentParser

__all__ = [
    "DocumentParser",
    "FallbackParser",
    "LiteParseAdapter",
    "ParsedDocument",
    "get_parser",
]
