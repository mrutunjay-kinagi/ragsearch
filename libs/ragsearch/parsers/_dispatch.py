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

    if path is None:
        raise UnsupportedFileTypeError("Invalid parser path: expected a file path, got None")

    if not isinstance(path, Path):
        try:
            path = Path(path)
        except TypeError as exc:
            raise UnsupportedFileTypeError(
                f"Invalid parser path: expected str or Path, got {type(path).__name__}"
            ) from exc

    liteparse = LiteParseAdapter()
    fallback = FallbackParser()

    if liteparse.available() and liteparse.supports(path):
        return liteparse
    if fallback.supports(path):
        return fallback
    raise UnsupportedFileTypeError(f"Unsupported file type: {path.suffix} for {path}")
