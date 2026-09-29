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


class NoDataFoundError(RagSearchError):
    """Raised when an operation requires data but none is available."""


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


class EmbeddingProbeError(RagSearchError, RuntimeError):
    """Raised when setup() cannot determine the embedding dimension from a probe embedding.

    Subclasses RuntimeError so existing ``except RuntimeError`` handlers around setup() still catch it.
    """


class MissingOptionalDependencyError(RagSearchError, ImportError):
    """Raised when a feature needs an optional extra that is not installed (e.g. ragsearch[chromadb])."""


class ModelNotFoundError(RagSearchError):
    """Raised when a provider rejects the configured model name as unknown or retired."""


def is_model_not_found_error(exc: Exception) -> bool:
    """Return True when a provider SDK error reports an unknown or removed model.

    Cohere reports both cases as HTTP 404 with a body message such as
    "model 'large' not found" or "model 'command-r' was removed".
    """
    if getattr(exc, "status_code", None) != 404:
        return False
    body = getattr(exc, "body", None)
    message = body.get("message", "") if isinstance(body, dict) else str(exc)
    return "model" in str(message).lower()


def provider_error_message(exc: Exception) -> str:
    """Extract the human-readable message from a provider SDK error."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])
    return str(exc)
