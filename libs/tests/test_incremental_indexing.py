"""
Incremental indexing tests.

Covers:
- New files detected on first scan.
- Changed files detected (SHA-256 digest differs after modification).
- Deleted files detected (present in state but no longer on disk).
- Renamed files detected (original path deleted; new path appears as new).
- Multi-root scanning (multiple directories / mixed files + dirs).
- SHA-256 correctness (known content → known digest).
- State persistence (save → reload → same hashes).
- update_state / remove_from_state lifecycle.
"""

import hashlib
import json
from pathlib import Path

import pytest

from libs.ragsearch.indexing import IncrementalIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str = "hello") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SHA-256 correctness
# ---------------------------------------------------------------------------

class TestSHA256:
    def test_known_content_matches_expected_digest(self, tmp_path):
        content = "ragsearch incremental indexing"
        f = _write(tmp_path / "known.txt", content)
        indexer = IncrementalIndexer(tmp_path / "state.json")
        assert indexer.compute_sha256(f) == _sha256(content)

    def test_same_content_same_digest(self, tmp_path):
        f1 = _write(tmp_path / "a.txt", "same")
        f2 = _write(tmp_path / "b.txt", "same")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        assert indexer.compute_sha256(f1) == indexer.compute_sha256(f2)

    def test_different_content_different_digest(self, tmp_path):
        f1 = _write(tmp_path / "a.txt", "hello")
        f2 = _write(tmp_path / "b.txt", "world")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        assert indexer.compute_sha256(f1) != indexer.compute_sha256(f2)

    def test_single_byte_change_changes_digest(self, tmp_path):
        f = _write(tmp_path / "file.txt", "abcdef")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        digest_before = indexer.compute_sha256(f)
        f.write_text("abcdeF", encoding="utf-8")
        digest_after = indexer.compute_sha256(f)
        assert digest_before != digest_after


# ---------------------------------------------------------------------------
# New files
# ---------------------------------------------------------------------------

class TestNewFiles:
    def test_all_files_new_on_empty_state(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        changes = indexer.classify_changes([tmp_path])
        assert len(changes["new"]) == 2
        assert changes["changed"] == []
        assert changes["deleted"] == []
        assert changes["unchanged"] == []

    def test_single_new_file_added_after_initial_scan(self, tmp_path):
        _write(tmp_path / "existing.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])

        _write(tmp_path / "newfile.txt")
        changes = indexer.classify_changes([tmp_path])
        assert str(tmp_path / "newfile.txt") in changes["new"]
        assert str(tmp_path / "existing.txt") in changes["unchanged"]


# ---------------------------------------------------------------------------
# Changed files
# ---------------------------------------------------------------------------

class TestChangedFiles:
    def test_modified_file_appears_in_changed(self, tmp_path):
        f = _write(tmp_path / "doc.txt", "original")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])

        f.write_text("modified", encoding="utf-8")
        changes = indexer.classify_changes([tmp_path])
        assert str(f) in changes["changed"]
        assert str(f) not in changes["unchanged"]
        assert str(f) not in changes["new"]

    def test_unmodified_file_stays_unchanged(self, tmp_path):
        f = _write(tmp_path / "stable.txt", "constant")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])
        changes = indexer.classify_changes([tmp_path])
        assert str(f) in changes["unchanged"]
        assert changes["changed"] == []


# ---------------------------------------------------------------------------
# Deleted files
# ---------------------------------------------------------------------------

class TestDeletedFiles:
    def test_removed_file_appears_in_deleted(self, tmp_path):
        f = _write(tmp_path / "todelete.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])

        f.unlink()
        changes = indexer.classify_changes([tmp_path])
        assert str(f) in changes["deleted"]
        assert str(f) not in changes["new"]
        assert str(f) not in changes["unchanged"]

    def test_remaining_file_not_deleted_after_sibling_removed(self, tmp_path):
        f_keep = _write(tmp_path / "keep.txt")
        f_del = _write(tmp_path / "delete.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])

        f_del.unlink()
        changes = indexer.classify_changes([tmp_path])
        assert str(f_del) in changes["deleted"]
        assert str(f_keep) in changes["unchanged"]


# ---------------------------------------------------------------------------
# Renamed files
# ---------------------------------------------------------------------------

class TestRenamedFiles:
    def test_rename_shows_as_deleted_plus_new(self, tmp_path):
        old = _write(tmp_path / "old_name.txt", "data")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])

        new = tmp_path / "new_name.txt"
        old.rename(new)
        changes = indexer.classify_changes([tmp_path])
        assert str(old) in changes["deleted"]
        assert str(new) in changes["new"]
        assert changes["changed"] == []


# ---------------------------------------------------------------------------
# Multi-root scanning
# ---------------------------------------------------------------------------

class TestMultiRoot:
    def test_two_directories_combined(self, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        _write(dir_a / "fa.txt", "alpha")
        _write(dir_b / "fb.txt", "beta")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        changes = indexer.classify_changes([dir_a, dir_b])
        paths = changes["new"]
        assert str(dir_a / "fa.txt") in paths
        assert str(dir_b / "fb.txt") in paths

    def test_file_and_directory_mix(self, tmp_path):
        loose = _write(tmp_path / "loose.txt", "lone file")
        sub = tmp_path / "sub"
        nested = _write(sub / "nested.txt", "nested")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        changes = indexer.classify_changes([loose, sub])
        assert str(loose) in changes["new"]
        assert str(nested) in changes["new"]

    def test_overlapping_roots_no_duplicates(self, tmp_path):
        f = _write(tmp_path / "file.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        # Passing the same root twice should not duplicate entries
        changes = indexer.classify_changes([tmp_path, tmp_path])
        all_paths = changes["new"] + changes["changed"] + changes["deleted"] + changes["unchanged"]
        assert len(all_paths) == len(set(all_paths)), "Duplicate paths detected"

    def test_independent_roots_both_tracked(self, tmp_path):
        r1 = tmp_path / "r1"
        r2 = tmp_path / "r2"
        f1 = _write(r1 / "a.txt")
        f2 = _write(r2 / "b.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([r1, r2])
        changes = indexer.classify_changes([r1, r2])
        assert str(f1) in changes["unchanged"]
        assert str(f2) in changes["unchanged"]

    def test_change_in_one_root_does_not_affect_other(self, tmp_path):
        r1 = tmp_path / "r1"
        r2 = tmp_path / "r2"
        f1 = _write(r1 / "stable.txt", "unchanged")
        f2 = _write(r2 / "changing.txt", "v1")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([r1, r2])

        f2.write_text("v2", encoding="utf-8")
        changes = indexer.classify_changes([r1, r2])
        assert str(f1) in changes["unchanged"]
        assert str(f2) in changes["changed"]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_reload_produces_same_state(self, tmp_path):
        f = _write(tmp_path / "file.txt", "content")
        state_file = tmp_path / "state.json"
        indexer = IncrementalIndexer(state_file)
        indexer.update_state([tmp_path])
        indexer.save_state()

        reloaded = IncrementalIndexer(state_file)
        assert reloaded.state == indexer.state

    def test_reload_detects_unchanged_from_saved_state(self, tmp_path):
        f = _write(tmp_path / "file.txt", "hello")
        state_file = tmp_path / "state.json"
        indexer = IncrementalIndexer(state_file)
        indexer.update_state([tmp_path])
        indexer.save_state()

        reloaded = IncrementalIndexer(state_file)
        changes = reloaded.classify_changes([tmp_path])
        assert str(f) in changes["unchanged"]

    def test_missing_state_file_starts_fresh(self, tmp_path):
        state_file = tmp_path / "nonexistent_state.json"
        indexer = IncrementalIndexer(state_file)
        assert indexer.state == {}

    def test_corrupted_state_file_starts_fresh(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_bytes(b"\xff\xfe not valid json")
        indexer = IncrementalIndexer(state_file)
        assert isinstance(indexer.state, dict)


# ---------------------------------------------------------------------------
# remove_from_state / reset_state lifecycle
# ---------------------------------------------------------------------------

class TestStateManagement:
    def test_remove_from_state_clears_entry(self, tmp_path):
        f = _write(tmp_path / "file.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])
        assert str(f) in indexer.state

        indexer.remove_from_state([f])
        assert str(f) not in indexer.state

    def test_remove_nonexistent_path_does_not_raise(self, tmp_path):
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.remove_from_state([tmp_path / "ghost.txt"])  # must not raise

    def test_reset_clears_all_state(self, tmp_path):
        _write(tmp_path / "a.txt")
        _write(tmp_path / "b.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])
        indexer.reset_state()
        assert indexer.state == {}

    def test_classify_after_reset_treats_all_as_new(self, tmp_path):
        f = _write(tmp_path / "file.txt")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        indexer.update_state([tmp_path])
        indexer.reset_state()
        changes = indexer.classify_changes([tmp_path])
        assert str(f) in changes["new"]
        assert changes["unchanged"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_directory_returns_no_changes(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        indexer = IncrementalIndexer(tmp_path / "state.json")
        changes = indexer.classify_changes([empty_dir])
        assert all(len(v) == 0 for v in changes.values())

    def test_nested_directory_structure_all_detected(self, tmp_path):
        root = tmp_path / "root"
        _write(root / "a" / "b" / "deep.txt", "deep")
        _write(root / "shallow.txt", "shallow")
        indexer = IncrementalIndexer(tmp_path / "state.json")
        changes = indexer.classify_changes([root])
        assert len(changes["new"]) == 2

    def test_binary_file_hashed_correctly(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(bytes(range(256)))
        expected = hashlib.sha256(bytes(range(256))).hexdigest()
        indexer = IncrementalIndexer(tmp_path / "state.json")
        assert indexer.compute_sha256(f) == expected

    def test_large_file_hashed_without_error(self, tmp_path):
        f = tmp_path / "large.bin"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        indexer = IncrementalIndexer(tmp_path / "state.json")
        digest = indexer.compute_sha256(f)
        assert len(digest) == 64  # SHA-256 hex length
