"""Tests for corrective RAG pipeline functions."""

import pytest
from app.rag import _compute_confidence, _reformulate_query


# ── Confidence scoring ────────────────────────────────────────────────

class TestComputeConfidence:
    def test_empty_list_returns_zero(self):
        assert _compute_confidence([]) == 0.0

    def test_rerank_scores_high(self):
        chunks = [
            {"chunk_id": "a", "rerank_score": 5.0},
            {"chunk_id": "b", "rerank_score": 3.0},
        ]
        confidence = _compute_confidence(chunks)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Both scores are positive

    def test_rerank_scores_low(self):
        chunks = [
            {"chunk_id": "a", "rerank_score": -8.0},
            {"chunk_id": "b", "rerank_score": -7.0},
        ]
        confidence = _compute_confidence(chunks)
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.3  # Very negative scores

    def test_distance_based_confidence(self):
        chunks = [
            {"chunk_id": "a", "distance": 0.2},
            {"chunk_id": "b", "distance": 0.5},
        ]
        confidence = _compute_confidence(chunks)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Low distances = high confidence

    def test_high_distance_low_confidence(self):
        chunks = [
            {"chunk_id": "a", "distance": 1.8},
            {"chunk_id": "b", "distance": 2.0},
        ]
        confidence = _compute_confidence(chunks)
        assert confidence < 0.2  # High distances = low confidence

    def test_mixed_rerank_and_distance(self):
        chunks = [
            {"chunk_id": "a", "rerank_score": 3.0},
            {"chunk_id": "b", "distance": 0.5},
        ]
        confidence = _compute_confidence(chunks)
        assert 0.0 <= confidence <= 1.0

    def test_single_chunk(self):
        chunks = [{"chunk_id": "a", "rerank_score": 5.0}]
        confidence = _compute_confidence(chunks)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_is_clamped(self):
        """Even extreme values should be clamped to [0, 1]."""
        chunks = [{"chunk_id": "a", "rerank_score": 100.0}]
        confidence = _compute_confidence(chunks)
        assert confidence <= 1.0

        chunks = [{"chunk_id": "a", "rerank_score": -100.0}]
        confidence = _compute_confidence(chunks)
        assert confidence >= 0.0


# ── Query reformulation ──────────────────────────────────────────────

class TestReformulateQuery:
    def test_attempt_1_includes_nice_ng12(self):
        result = _reformulate_query("hemoptysis referral", 1)
        assert "NICE NG12" in result
        assert "hemoptysis referral" in result

    def test_attempt_2_broader_search(self):
        result = _reformulate_query("hemoptysis referral criteria", 2)
        assert "cancer recognition" in result
        assert "hemoptysis" in result

    def test_different_queries_per_attempt(self):
        q1 = _reformulate_query("lung cancer symptoms", 1)
        q2 = _reformulate_query("lung cancer symptoms", 2)
        assert q1 != q2

    def test_empty_query_attempt_2(self):
        """Attempt 2 with empty query should still return something."""
        result = _reformulate_query("", 2)
        assert "cancer recognition" in result
