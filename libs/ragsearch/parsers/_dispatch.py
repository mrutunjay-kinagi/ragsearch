"""
Parser selection helpers.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import UnsupportedFileTypeError
from ._fallback import FallbackParser
from ._liteparse import LiteParseAdapter
from ._protocol import DocumentParser


def get_parser(path: Path | str) -> DocumentParser:
    """Return the best parser for a file path."""

    if not isinstance(path, Path):
        path = Path(path)

    liteparse = LiteParseAdapter()
    fallback = FallbackParser()

    if liteparse.available() and liteparse.supports(path):
        return liteparse
    if fallback.supports(path):
        return fallback
    raise UnsupportedFileTypeError(f"Unsupported file type: {path.suffix} for {path}")
