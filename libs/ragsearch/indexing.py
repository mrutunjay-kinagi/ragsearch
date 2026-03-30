"""
indexing.py – SHA-256 based incremental file tracker.

Scans one or more root paths (files or directories), computes a SHA-256
digest for every file found, and compares the result against a persisted
state snapshot so that callers can act only on files that are new, changed,
deleted, or renamed since the last run.

Example usage::

    from libs.ragsearch.indexing import IncrementalIndexer

    indexer = IncrementalIndexer(state_path=".ragsearch_index_state.json")
    changes = indexer.classify_changes(["/data/docs"])
    # changes = {"new": [...], "changed": [...], "deleted": [...], "unchanged": [...]}
    indexer.update_state(["/data/docs"])
    indexer.save_state()
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class IncrementalIndexer:
    """Tracks file changes across one or more root paths using SHA-256 digests.

    State is persisted as a JSON file mapping ``str(path)`` → hex-digest so
    that incremental changes can be detected between runs.

    Args:
        state_path: Path to the JSON file used for persisting state.
    """

    def __init__(self, state_path: Union[str, Path] = ".ragsearch_index_state.json") -> None:
        self.state_path = Path(state_path)
        self._state: Dict[str, str] = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, str]:
        """Load persisted state from *state_path*, or return an empty dict."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load index state from %s: %s", self.state_path, exc)
        return {}

    def save_state(self) -> None:
        """Persist the current in-memory state to *state_path*."""
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2)
        logger.debug("Index state saved to %s", self.state_path)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    # Buffer size used for streaming reads; 64 KB is a good trade-off between
    # memory usage and the number of system-call round-trips for most files.
    _READ_CHUNK = 65_536

    @staticmethod
    def compute_sha256(path: Union[str, Path]) -> str:
        """Return the hex-encoded SHA-256 digest for *path*."""
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(IncrementalIndexer._READ_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------------
    # Change classification
    # ------------------------------------------------------------------

    def _collect_files(self, root_paths: Sequence[Union[str, Path]]) -> Dict[str, str]:
        """Return ``{str(path): sha256}`` for every file reachable from *root_paths*."""
        files: Dict[str, str] = {}
        for root in root_paths:
            root = Path(root)
            if root.is_file():
                files[str(root)] = self.compute_sha256(root)
            elif root.is_dir():
                for child in sorted(root.rglob("*")):
                    if child.is_file():
                        files[str(child)] = self.compute_sha256(child)
        return files

    def classify_changes(
        self, root_paths: Sequence[Union[str, Path]]
    ) -> Dict[str, List[str]]:
        """Compare current files under *root_paths* against persisted state.

        Returns a dict with four keys:

        * ``"new"``       – paths present now but not in previous state
        * ``"changed"``   – paths in both states but with a different digest
        * ``"deleted"``   – paths in previous state but no longer present
        * ``"unchanged"`` – paths present in both states with the same digest

        Args:
            root_paths: One or more file/directory paths to scan.

        Returns:
            Classification dict as described above.
        """
        current = self._collect_files(root_paths)
        previous = self._state

        new = [k for k in current if k not in previous]
        changed = [k for k in current if k in previous and current[k] != previous[k]]
        deleted = [k for k in previous if k not in current]
        unchanged = [k for k in current if k in previous and current[k] == previous[k]]

        return {
            "new": sorted(new),
            "changed": sorted(changed),
            "deleted": sorted(deleted),
            "unchanged": sorted(unchanged),
        }

    def update_state(self, root_paths: Sequence[Union[str, Path]]) -> None:
        """Scan *root_paths* and merge current digests into the in-memory state.

        Does **not** remove paths that have been deleted; call
        :meth:`remove_from_state` for that.
        """
        current = self._collect_files(root_paths)
        self._state.update(current)

    def remove_from_state(self, paths: Sequence[Union[str, Path]]) -> None:
        """Remove *paths* from the in-memory state (e.g. after deletion)."""
        for p in paths:
            self._state.pop(str(p), None)

    def reset_state(self) -> None:
        """Clear the entire in-memory state."""
        self._state.clear()

    @property
    def state(self) -> Dict[str, str]:
        """Read-only view of the current in-memory state."""
        return dict(self._state)
