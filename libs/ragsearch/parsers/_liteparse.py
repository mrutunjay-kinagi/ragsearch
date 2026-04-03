"""
LiteParse subprocess adapter.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from ..errors import ParseCorruptError, ParseTimeoutError, ParserUnavailableError, UnsupportedFileTypeError
from ._models import ParsedDocument


class LiteParseAdapter:
    """Adapter around the LiteParse CLI."""

    ENV_CLI_PATH = "RAGSEARCH_LITEPARSE_CLI"

    SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".txt", ".png", ".jpg", ".jpeg"}

    def __init__(self, timeout_s: int = 60):
        self.timeout_s = timeout_s

    @classmethod
    def available(cls) -> bool:
        """Return True when the required Node executables are available."""

        cli_path = os.environ.get(cls.ENV_CLI_PATH)
        if cli_path:
            return bool(shutil.which(cli_path) or Path(cli_path).exists())
        return bool(shutil.which("node") and shutil.which("npx"))

    def _build_command(self, path: Path) -> list[str]:
        cli_path = os.environ.get(self.ENV_CLI_PATH)
        if cli_path:
            return [cli_path, "--json", str(path)]
        return ["npx", "--no-install", "@run-llama/liteparse", "--json", str(path)]

    def supports(self, path: Path | str) -> bool:
        path = Path(path) if not isinstance(path, Path) else path
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def parse(self, path: Path | str) -> Iterator[ParsedDocument]:
        if path is None:
            raise ParseCorruptError("Input path cannot be None")
        if not isinstance(path, Path):
            path = Path(path)
        if not self.available():
            raise ParserUnavailableError("LiteParse CLI not found; install Node.js 18+ and npx")
        if not path.exists():
            raise ParseCorruptError(f"Input path is missing or unreadable: {path}")
        if not self.supports(path):
            raise UnsupportedFileTypeError(f"LiteParse does not support file type: {path.suffix}")

        command = self._build_command(path)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParseTimeoutError(f"LiteParse timed out after {self.timeout_s}s for {path}", cause=exc) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise ParseCorruptError(f"LiteParse failed for {path}: {stderr}")

        try:
            payload = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise ParseCorruptError(f"LiteParse output for {path} was not valid JSON", cause=exc) from exc

        documents = payload.get("documents") if isinstance(payload, dict) else None
        if documents is None:
            documents = [payload]
        if not isinstance(documents, list) or not documents:
            raise ParseCorruptError(f"LiteParse output for {path} did not contain documents")

        for document in documents:
            if not isinstance(document, dict):
                raise ParseCorruptError(f"LiteParse output for {path} contained an invalid document entry")
            text = document.get("text", "")
            if not isinstance(text, str):
                raise ParseCorruptError(f"LiteParse output for {path} contained a document with invalid text")

            metadata = document.get("metadata", {})
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                raise ParseCorruptError(f"LiteParse output for {path} contained a document with invalid metadata")

            source_path = document.get("source_path", str(path))
            parser_name = document.get("parser_name", "liteparse")
            yield ParsedDocument(
                text=text,
                metadata=metadata,
                source_path=source_path,
                parser_name=parser_name,
            )
