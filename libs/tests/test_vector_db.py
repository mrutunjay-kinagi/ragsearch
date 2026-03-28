"""
Unit tests for libs/ragsearch/vector_db.py (VectorDB class).
Tests cover insert, search, and _normalize_embedding without external services.
"""
import pytest
import numpy as np

from libs.ragsearch.vector_db import VectorDB


class TestVectorDBInit:
    def test_default_dimension(self):
        db = VectorDB()
        assert db.index.d == 1024
        assert db.current_id == 0
        assert db.metadata_store == {}

    def test_custom_dimension(self):
        db = VectorDB(embedding_dim=128)
        assert db.index.d == 128

    def test_invalid_dimension_zero(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim=0)

    def test_invalid_dimension_negative(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim=-1)

    def test_invalid_dimension_not_int(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim="128")


class TestNormalizeEmbedding:
    def test_unit_vector_unchanged(self):
        v = [1.0, 0.0, 0.0]
        result = VectorDB._normalize_embedding(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_normalized_has_unit_norm(self):
        v = [3.0, 4.0]
        result = VectorDB._normalize_embedding(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            VectorDB._normalize_embedding([0.0, 0.0, 0.0])

    def test_returns_float32_array(self):
        result = VectorDB._normalize_embedding([1.0, 2.0, 3.0])
        assert result.dtype == np.float32


class TestVectorDBInsert:
    def test_insert_increments_id(self):
        db = VectorDB(embedding_dim=4)
        db.insert([1.0, 0.0, 0.0, 0.0], {"label": "a"})
        assert db.current_id == 1
        assert db.index.ntotal == 1

    def test_insert_stores_metadata(self):
        db = VectorDB(embedding_dim=4)
        meta = {"name": "test", "score": 0.9}
        db.insert([1.0, 0.0, 0.0, 0.0], meta)
        assert db.metadata_store[0] == meta

    def test_multiple_inserts(self):
        db = VectorDB(embedding_dim=4)
        db.insert([1.0, 0.0, 0.0, 0.0], {"id": 0})
        db.insert([0.0, 1.0, 0.0, 0.0], {"id": 1})
        assert db.current_id == 2
        assert db.index.ntotal == 2


class TestVectorDBSearch:
    def _make_db(self, dim=4):
        db = VectorDB(embedding_dim=dim)
        db.insert([1.0, 0.0, 0.0, 0.0], {"label": "a"})
        db.insert([0.0, 1.0, 0.0, 0.0], {"label": "b"})
        db.insert([0.0, 0.0, 1.0, 0.0], {"label": "c"})
        return db

    def test_search_returns_top_k_results(self):
        db = self._make_db()
        results = db.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_search_most_similar_first(self):
        db = self._make_db()
        results = db.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0]["metadata"]["label"] == "a"

    def test_search_result_has_expected_keys(self):
        db = self._make_db()
        results = db.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert "index" in results[0]
        assert "similarity" in results[0]
        assert "metadata" in results[0]

    def test_search_empty_index_raises(self):
        db = VectorDB(embedding_dim=4)
        with pytest.raises(ValueError, match="empty"):
            db.search([1.0, 0.0, 0.0, 0.0])

    def test_search_top_k_capped_by_index_size(self):
        db = self._make_db()
        results = db.search([1.0, 0.0, 0.0, 0.0], top_k=10)
        assert len(results) <= 3
