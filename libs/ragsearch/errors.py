"""
Shared ragsearch error hierarchy.
"""

from __future__ import annotations


class RagSearchError(Exception):
    """Base error for ragsearch failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause


class ParsingError(RagSearchError):
    """Base error for parser failures."""


class UnsupportedFileTypeError(ParsingError):
    """Raised when no parser supports a file type."""


class ParserUnavailableError(ParsingError):
    """Raised when an optional parser backend is unavailable."""


class ParseTimeoutError(ParsingError):
    """Raised when parsing exceeds the configured timeout."""


class ParseCorruptError(ParsingError):
    """Raised when parser output or input data is invalid."""
