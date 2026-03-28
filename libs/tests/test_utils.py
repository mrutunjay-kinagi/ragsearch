"""
Unit tests for ragsearch utility functions (libs/ragsearch/utils.py).
Tests cover pure functions that require no external services.
"""
import pytest
import pandas as pd
from libs.ragsearch.utils import (
    extract_textual_columns,
    preprocess_search_text,
    preprocess_text,
    log_data_summary,
)


class TestExtractTextualColumns:
    def test_returns_object_dtype_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25], "city": ["NY", "LA"]})
        cols = extract_textual_columns(df)
        assert "name" in cols
        assert "city" in cols
        assert "age" not in cols

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame()
        cols = extract_textual_columns(df)
        assert cols == []

    def test_no_text_columns_returns_empty_list(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        cols = extract_textual_columns(df)
        assert cols == []

    def test_all_text_columns(self):
        df = pd.DataFrame({"a": ["foo"], "b": ["bar"]})
        cols = extract_textual_columns(df)
        assert set(cols) == {"a", "b"}


class TestPreprocessSearchText:
    def test_strips_whitespace(self):
        assert preprocess_search_text("  hello  ") == "hello"

    def test_converts_to_lowercase(self):
        assert preprocess_search_text("Hello World") == "hello world"

    def test_strips_and_lowercases(self):
        assert preprocess_search_text("  QUERY  ") == "query"

    def test_empty_string(self):
        assert preprocess_search_text("") == ""

    def test_already_lowercase_no_whitespace(self):
        assert preprocess_search_text("test query") == "test query"


class TestPreprocessText:
    def test_joins_columns_with_pipe(self):
        row = pd.Series({"col1": "foo", "col2": "bar"})
        result = preprocess_text(row, ["col1", "col2"])
        assert result == "foo | bar"

    def test_handles_nan_as_empty_string(self):
        row = pd.Series({"col1": "foo", "col2": float("nan")})
        result = preprocess_text(row, ["col1", "col2"])
        assert result == "foo | "

    def test_single_column(self):
        row = pd.Series({"col1": "only"})
        result = preprocess_text(row, ["col1"])
        assert result == "only"

    def test_numeric_values_converted_to_string(self):
        # Each value must be in its own column so pandas preserves per-column dtype
        df = pd.DataFrame({"a": [42], "b": [3.14]})
        result = preprocess_text(df.iloc[0], ["a", "b"])
        assert "42" in result and "3.14" in result


class TestLogDataSummary:
    def test_does_not_raise(self):
        df = pd.DataFrame({"name": ["Alice"], "age": [30]})
        # Should log without raising any exceptions
        log_data_summary(df)

    def test_works_with_empty_dataframe(self):
        df = pd.DataFrame()
        log_data_summary(df)
