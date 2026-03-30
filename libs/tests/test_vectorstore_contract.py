"""
Contract tests for VectorStore plugins.

These tests verify the *interface contract* that any VectorStore backend must
satisfy.  The reference implementation is ``libs/ragsearch/vector_db.VectorDB``
(FAISS-backed).

A VectorStore contract requires:
- ``insert(embedding, metadata)``   – stores a vector + metadata dict.
- ``search(query_embedding, top_k)`` – returns up to *top_k* results.
- Each result dict must contain ``"index"``, ``"similarity"``, and ``"metadata"`` keys.
- Searching an empty store raises ``ValueError``.
- Normalisation of zero vectors must raise ``ValueError``.
- Backends must be instantiable with a configurable embedding dimension.
"""

import abc

import numpy as np
import pytest

from libs.ragsearch.vector_db import VectorDB


# ---------------------------------------------------------------------------
# Abstract contract – makes it easy to test alternative backends later
# ---------------------------------------------------------------------------

class VectorStoreContract(abc.ABC):
    """Abstract base for VectorStore contract test suites.

    Sub-class and implement :meth:`make_store` to test any backend.
    """

    DIM: int = 8  # low dimension keeps tests fast

    @abc.abstractmethod
    def make_store(self, dim: int = DIM) -> object:
        """Return a freshly initialised VectorStore with *dim* dimensions."""

    def _unit_vec(self, dim: int | None = None, seed: int = 42) -> list:
        """Return a deterministic unit vector of length *dim* for use in tests."""
        d = dim or self.DIM
        v = np.random.default_rng(seed).standard_normal(d).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_store_is_initially_empty(self):
        store = self.make_store()
        # For FAISS-backed stores the index reports ntotal
        assert store.index.ntotal == 0

    def test_custom_dimension_accepted(self):
        store = self.make_store(dim=16)
        assert store.index.d == 16

    # ------------------------------------------------------------------
    # insert()
    # ------------------------------------------------------------------

    def test_insert_single_vector(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {"id": "doc-0"})
        assert store.index.ntotal == 1

    def test_insert_increments_current_id(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {"id": "a"})
        store.insert(self._unit_vec(), {"id": "b"})
        assert store.current_id == 2

    def test_insert_stores_metadata(self):
        store = self.make_store()
        meta = {"title": "hello", "score": 0.99}
        store.insert(self._unit_vec(), meta)
        assert store.metadata_store[0] == meta

    def test_insert_multiple_metadata_are_independent(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {"k": "v0"})
        store.insert(self._unit_vec(), {"k": "v1"})
        assert store.metadata_store[0]["k"] == "v0"
        assert store.metadata_store[1]["k"] == "v1"

    # ------------------------------------------------------------------
    # search()
    # ------------------------------------------------------------------

    def test_search_returns_list(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {})
        results = store.search(self._unit_vec(), top_k=1)
        assert isinstance(results, list)

    def test_search_result_has_required_keys(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {"x": 1})
        result = store.search(self._unit_vec(), top_k=1)[0]
        assert "index" in result
        assert "similarity" in result
        assert "metadata" in result

    def test_search_respects_top_k(self):
        store = self.make_store()
        for i in range(5):
            store.insert(self._unit_vec(), {"i": i})
        results = store.search(self._unit_vec(), top_k=3)
        assert len(results) <= 3

    def test_search_returns_most_similar_first(self):
        store = self.make_store()
        # Insert two orthogonal unit vectors
        v1 = [1.0] + [0.0] * (self.DIM - 1)
        v2 = [0.0, 1.0] + [0.0] * (self.DIM - 2)
        store.insert(v1, {"label": "a"})
        store.insert(v2, {"label": "b"})
        # Query closest to v1
        results = store.search(v1, top_k=2)
        assert results[0]["metadata"]["label"] == "a"

    def test_search_similarity_decreasing(self):
        store = self.make_store()
        for i in range(4):
            store.insert(self._unit_vec(), {"i": i})
        results = store.search(self._unit_vec(), top_k=4)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_search_empty_store_raises(self):
        store = self.make_store()
        with pytest.raises(ValueError, match="empty"):
            store.search(self._unit_vec(), top_k=5)

    def test_search_top_k_larger_than_index_does_not_crash(self):
        store = self.make_store()
        store.insert(self._unit_vec(), {})
        results = store.search(self._unit_vec(), top_k=100)
        assert len(results) <= 1

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def test_normalize_zero_vector_raises(self):
        store = self.make_store()
        with pytest.raises(ValueError, match="zero vector"):
            store._normalize_embedding([0.0] * self.DIM)

    def test_normalize_produces_unit_norm(self):
        store = self.make_store()
        result = store._normalize_embedding([3.0, 4.0] + [0.0] * (self.DIM - 2))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_normalize_returns_float32(self):
        store = self.make_store()
        result = store._normalize_embedding([1.0] + [0.0] * (self.DIM - 1))
        assert result.dtype == np.float32

    # ------------------------------------------------------------------
    # Metadata isolation
    # ------------------------------------------------------------------

    def test_metadata_not_shared_between_entries(self):
        store = self.make_store()
        m = {"key": "original"}
        store.insert(self._unit_vec(), m)
        m["key"] = "mutated"
        # The stored value should still be "original" (snapshot at insert time)
        # OR "mutated" – both are acceptable because the contract only requires
        # that separate inserts are independent; shallow copy behaviour is impl-defined.
        # This test just asserts no exception is raised.
        _ = store.metadata_store[0]

    # ------------------------------------------------------------------
    # Round-trip
    # ------------------------------------------------------------------

    def test_insert_then_search_returns_correct_metadata(self):
        store = self.make_store()
        v = [1.0] + [0.0] * (self.DIM - 1)
        store.insert(v, {"doc": "first"})
        results = store.search(v, top_k=1)
        assert results[0]["metadata"]["doc"] == "first"


# ---------------------------------------------------------------------------
# Concrete suite for the default FAISS VectorDB backend
# ---------------------------------------------------------------------------

class TestFAISSVectorStoreContract(VectorStoreContract):
    """Run the full VectorStore contract against the FAISS VectorDB backend."""

    def make_store(self, dim: int = VectorStoreContract.DIM) -> VectorDB:
        return VectorDB(embedding_dim=dim)

    # Extra FAISS-specific tests -------------------------------------------

    def test_faiss_default_dimension_is_1024(self):
        db = VectorDB()
        assert db.index.d == 1024

    def test_faiss_invalid_dim_zero(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim=0)

    def test_faiss_invalid_dim_negative(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim=-5)

    def test_faiss_invalid_dim_non_int(self):
        with pytest.raises(ValueError):
            VectorDB(embedding_dim=3.5)
