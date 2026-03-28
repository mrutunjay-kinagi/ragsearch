"""
Unit tests for libs/ragsearch/config.py (load_configuration).
"""
import json
import os
import pytest
import tempfile

from libs.ragsearch.config import load_configuration


class TestLoadConfiguration:
    def test_loads_valid_json(self, tmp_path):
        config = {"key": "value", "number": 42}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        result = load_configuration(str(config_file))
        assert result == config

    def test_loads_nested_config(self, tmp_path):
        config = {"section": {"key": "val"}, "list": [1, 2, 3]}
        config_file = tmp_path / "nested.json"
        config_file.write_text(json.dumps(config))
        result = load_configuration(str(config_file))
        assert result["section"]["key"] == "val"
        assert result["list"] == [1, 2, 3]

    def test_raises_on_missing_file(self, tmp_path):
        missing = str(tmp_path / "nonexistent.json")
        with pytest.raises(Exception):
            load_configuration(missing)

    def test_raises_on_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ not valid json }")
        with pytest.raises(Exception):
            load_configuration(str(bad_file))

    def test_empty_json_object(self, tmp_path):
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")
        result = load_configuration(str(config_file))
        assert result == {}
