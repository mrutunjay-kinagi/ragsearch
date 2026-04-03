"""
Parser data models.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized document content returned by parsers."""

    text: str
    metadata: dict = field(default_factory=dict)
    source_path: str = ""
    parser_name: str = ""
