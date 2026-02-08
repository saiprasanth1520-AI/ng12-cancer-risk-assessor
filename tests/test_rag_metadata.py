"""Tests for RAG metadata filtering and reranker optimizations."""

import pytest
from unittest.mock import patch, MagicMock

from app.rag import _extract_cancer_filter, CANCER_KEYWORDS


# ── _extract_cancer_filter ────────────────────────────────────────────


class TestExtractCancerFilter:
    def test_lung_query(self):
        result = _extract_cancer_filter("patient with hemoptysis and persistent cough")
        assert result is not None
        assert "lung" in result["cancer_type"]["$in"]
        assert "general" in result["cancer_type"]["$in"]
        assert "multi" in result["cancer_type"]["$in"]

    def test_colorectal_query(self):
        result = _extract_cancer_filter("change in bowel habit rectal bleeding")
        assert result is not None
        assert "colorectal" in result["cancer_type"]["$in"]

    def test_prostate_query(self):
        result = _extract_cancer_filter("elevated PSA level in male patient")
        assert result is not None
        assert "prostate" in result["cancer_type"]["$in"]

    def test_multiple_cancer_types(self):
        result = _extract_cancer_filter("lung cancer and bowel symptoms")
        assert result is not None
        types = result["cancer_type"]["$in"]
        assert "lung" in types
        assert "colorectal" in types
        assert "general" in types
        assert "multi" in types

    def test_no_cancer_type_returns_none(self):
        result = _extract_cancer_filter("general health advice")
        assert result is None

    def test_empty_query_returns_none(self):
        result = _extract_cancer_filter("")
        assert result is None

    def test_skin_melanoma_query(self):
        result = _extract_cancer_filter("suspicious mole melanoma check")
        assert result is not None
        assert "skin" in result["cancer_type"]["$in"]

    def test_haematological_query(self):
        result = _extract_cancer_filter("lymphoma symptoms lymph nodes")
        assert result is not None
        assert "haematological" in result["cancer_type"]["$in"]

    def test_gynaecological_query(self):
        result = _extract_cancer_filter("ovarian cancer symptoms")
        assert result is not None
        assert "gynaecological" in result["cancer_type"]["$in"]

    def test_brain_query(self):
        result = _extract_cancer_filter("brain tumour neurological symptoms")
        assert result is not None
        assert "brain_cns" in result["cancer_type"]["$in"]

    @patch("app.rag.METADATA_FILTER_ENABLED", False)
    def test_disabled_returns_none(self):
        result = _extract_cancer_filter("lung cancer symptoms")
        assert result is None

    def test_always_includes_multi_and_general(self):
        result = _extract_cancer_filter("bladder haematuria")
        assert result is not None
        types = result["cancer_type"]["$in"]
        assert "multi" in types
        assert "general" in types


# ── Metadata filter integration with search ──────────────────────────


class TestMetadataFilterFallback:
    @patch("app.rag.get_collection")
    @patch("app.rag.METADATA_FILTER_ENABLED", True)
    @patch("app.rag.BM25_ENABLED", False)
    @patch("app.rag.RERANK_ENABLED", False)
    def test_fallback_when_too_few_results(self, mock_get_collection):
        """If filtered search returns < 3 results, retry without filter."""
        from app.rag import search_guidelines

        mock_collection = MagicMock()

        # First call (filtered): returns only 1 result
        filtered_result = {
            "ids": [["chunk1"]],
            "documents": [["some text"]],
            "metadatas": [[{"page": 1}]],
            "distances": [[0.1]],
        }
        # Second call (unfiltered): returns 5 results
        unfiltered_result = {
            "ids": [["c1", "c2", "c3", "c4", "c5"]],
            "documents": [["t1", "t2", "t3", "t4", "t5"]],
            "metadatas": [[{"page": 1}, {"page": 2}, {"page": 3}, {"page": 4}, {"page": 5}]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }

        mock_collection.query.side_effect = [filtered_result, unfiltered_result]
        mock_get_collection.return_value = mock_collection

        results = search_guidelines("lung cancer symptoms", top_k=5)

        # Should have been called twice (filter, then fallback)
        assert mock_collection.query.call_count == 2
        assert len(results) == 5

    @patch("app.rag.get_collection")
    @patch("app.rag.METADATA_FILTER_ENABLED", True)
    @patch("app.rag.BM25_ENABLED", False)
    @patch("app.rag.RERANK_ENABLED", False)
    def test_no_fallback_when_enough_results(self, mock_get_collection):
        """If filtered search returns >= 3 results, don't retry."""
        from app.rag import search_guidelines

        mock_collection = MagicMock()
        filtered_result = {
            "ids": [["c1", "c2", "c3", "c4", "c5"]],
            "documents": [["t1", "t2", "t3", "t4", "t5"]],
            "metadatas": [[{"page": 1}, {"page": 2}, {"page": 3}, {"page": 4}, {"page": 5}]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }
        mock_collection.query.return_value = filtered_result
        mock_get_collection.return_value = mock_collection

        results = search_guidelines("lung cancer symptoms", top_k=5)

        # Should only be called once (enough results from filtered search)
        assert mock_collection.query.call_count == 1
        assert len(results) == 5


# ── Reranker score cache ─────────────────────────────────────────────


class TestRerankerCache:
    @patch("app.rag._get_reranker")
    def test_reranker_caches_scores(self, mock_get_reranker):
        """Calling _rerank twice with same query should reuse cached scores."""
        from app.rag import _rerank, _rerank_cache

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.5]
        mock_get_reranker.return_value = mock_reranker

        chunks = [
            {"chunk_id": "test_c1", "text": "lung cancer referral"},
            {"chunk_id": "test_c2", "text": "breast cancer referral"},
        ]

        # First call — should invoke predict
        result1 = _rerank("test query for cache", chunks.copy())
        assert mock_reranker.predict.call_count == 1

        # Second call with same query — should use cache
        chunks2 = [
            {"chunk_id": "test_c1", "text": "lung cancer referral"},
            {"chunk_id": "test_c2", "text": "breast cancer referral"},
        ]
        result2 = _rerank("test query for cache", chunks2)
        # predict should NOT be called again
        assert mock_reranker.predict.call_count == 1

        # Clean up cache entries we added
        import hashlib
        qh = hashlib.md5(b"test query for cache").hexdigest()[:8]
        _rerank_cache.pop(f"{qh}:test_c1", None)
        _rerank_cache.pop(f"{qh}:test_c2", None)
