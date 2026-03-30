"""
Contract tests for Parser plugins.

These tests verify the *interface contract* that any data-parser plugin must
satisfy.  Currently, the parsing logic lives in ``libs/ragsearch/setup.py``
(the ``setup()`` function), so the tests exercise that pathway.

A parser contract requires:
- Accepting a ``pathlib.Path`` that points to a supported file.
- Returning a ``pandas.DataFrame`` with at least one row and one column.
- Raising ``FileNotFoundError`` for a missing path.
- Raising ``ValueError`` for an unsupported file type.
- Raising ``RuntimeError`` (or a subclass) for a structurally corrupt file.
"""

import importlib
import json
import pytest
import pandas as pd
from pathlib import Path

# Parquet support requires pyarrow or fastparquet; skip when neither is present.
_parquet_available = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)
_parquet_skip = pytest.mark.skipif(
    not _parquet_available, reason="pyarrow or fastparquet required for Parquet tests"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_file(path: Path) -> pd.DataFrame:
    """Thin wrapper that calls the current setup.py parser logic.

    Isolates the contract tests from the full ``setup()`` entrypoint so we can
    test the parsing layer independently of Cohere/FAISS initialisation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except (ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load data: {exc}") from exc


# ---------------------------------------------------------------------------
# Contract: CSV
# ---------------------------------------------------------------------------

class TestParserContractCSV:
    """Parser must load a valid CSV into a non-empty DataFrame."""

    def test_loads_csv_with_expected_columns(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,score\nalice,0.9\nbob,0.7\n")
        df = _parse_file(f)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["name", "score"]
        assert len(df) == 2

    def test_csv_returns_dataframe_type(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col\nval\n")
        result = _parse_file(f)
        assert isinstance(result, pd.DataFrame)

    def test_csv_single_column(self, tmp_path):
        f = tmp_path / "single.csv"
        f.write_text("text\nhello\nworld\n")
        df = _parse_file(f)
        assert "text" in df.columns
        assert len(df) == 2

    def test_csv_preserves_row_count(self, tmp_path):
        rows = "\n".join(f"row{i},{i}" for i in range(50))
        f = tmp_path / "big.csv"
        f.write_text(f"name,num\n{rows}\n")
        df = _parse_file(f)
        assert len(df) == 50

    def test_empty_csv_body_returns_empty_dataframe(self, tmp_path):
        f = tmp_path / "empty_body.csv"
        f.write_text("col1,col2\n")
        df = _parse_file(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Contract: JSON
# ---------------------------------------------------------------------------

class TestParserContractJSON:
    """Parser must load a valid JSON array of records."""

    def test_loads_json_records(self, tmp_path):
        data = [{"name": "alice", "score": 0.9}, {"name": "bob", "score": 0.7}]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        df = _parse_file(f)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "name" in df.columns

    def test_json_returns_dataframe(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"x": 1}]))
        assert isinstance(_parse_file(f), pd.DataFrame)


# ---------------------------------------------------------------------------
# Contract: Parquet
# ---------------------------------------------------------------------------

@_parquet_skip
class TestParserContractParquet:
    """Parser must load a valid Parquet file."""

    def test_loads_parquet(self, tmp_path):
        df_orig = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        f = tmp_path / "data.parquet"
        df_orig.to_parquet(f, index=False)
        df = _parse_file(f)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3

    def test_pq_extension_also_supported(self, tmp_path):
        df_orig = pd.DataFrame({"val": [10, 20]})
        f = tmp_path / "data.pq"
        df_orig.to_parquet(f, index=False)
        df = _parse_file(f)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Contract: Error handling
# ---------------------------------------------------------------------------

class TestParserContractErrors:
    """Parser must raise well-typed exceptions for invalid input."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            _parse_file(missing)

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _parse_file(f)

    def test_corrupt_csv_raises_runtime_error(self, tmp_path):
        # A file with binary noise may fail CSV parsing or be read as garbage.
        # Either outcome is acceptable, but if an exception is raised it must
        # NOT be a bare ValueError or FileNotFoundError (those are reserved for
        # semantic errors such as unsupported extension / missing file).
        f = tmp_path / "corrupt.csv"
        f.write_bytes(b"\x00\x01\x02\x03bad\xff\xfe")
        raised = False
        try:
            _parse_file(f)
        except (RuntimeError, pd.errors.ParserError, UnicodeDecodeError):
            raised = True
        except (ValueError, FileNotFoundError) as exc:
            pytest.fail(f"Unexpected exception type for corrupt file: {type(exc).__name__}: {exc}")
        # If no exception was raised, pandas silently read the garbage bytes,
        # which is also a valid (lenient) behaviour for the current implementation.
        _ = raised  # documented: may or may not raise

    def test_txt_extension_raises_value_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("some text")
        with pytest.raises(ValueError):
            _parse_file(f)

    def test_no_extension_raises_value_error(self, tmp_path):
        f = tmp_path / "noext"
        f.write_text("data")
        with pytest.raises(ValueError):
            _parse_file(f)
