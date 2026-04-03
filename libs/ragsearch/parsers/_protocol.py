"""
Parser protocol definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from ._models import ParsedDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Structural contract for document parsers."""

    def supports(self, path: Path | str) -> bool:
        """Return True when the parser can handle the given path."""

    def parse(self, path: Path | str) -> Iterator[ParsedDocument]:
        """Parse a path into normalized document objects."""
