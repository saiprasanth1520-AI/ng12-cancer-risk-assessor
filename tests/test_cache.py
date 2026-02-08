"""Tests for RAG cache including warm-up functionality."""

import pytest
from unittest.mock import patch, MagicMock

from app.cache import (
    _cache_key, cache_get, cache_set, cache_stats,
    warm_cache, WARMUP_QUERIES,
)


# ── Cache key generation ─────────────────────────────────────────────


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("lung cancer", 5)
        k2 = _cache_key("lung cancer", 5)
        assert k1 == k2

    def test_different_queries(self):
        k1 = _cache_key("lung cancer", 5)
        k2 = _cache_key("breast cancer", 5)
        assert k1 != k2

    def test_different_top_k(self):
        k1 = _cache_key("lung cancer", 5)
        k2 = _cache_key("lung cancer", 10)
        assert k1 != k2

    def test_case_insensitive(self):
        k1 = _cache_key("Lung Cancer", 5)
        k2 = _cache_key("lung cancer", 5)
        assert k1 == k2

    def test_strips_whitespace(self):
        k1 = _cache_key("  lung cancer  ", 5)
        k2 = _cache_key("lung cancer", 5)
        assert k1 == k2


# ── Cache get/set (in-memory) ────────────────────────────────────────


class TestCacheGetSet:
    @patch("app.cache._get_redis", return_value=None)
    def test_set_and_get(self, _):
        results = [{"chunk_id": "c1", "text": "test", "page": 1}]
        cache_set("test query cache", 5, results)
        got = cache_get("test query cache", 5)
        assert got is not None
        assert got[0]["chunk_id"] == "c1"

    @patch("app.cache._get_redis", return_value=None)
    def test_miss_returns_none(self, _):
        got = cache_get("nonexistent query xyzzy", 5)
        assert got is None


# ── Cache stats ──────────────────────────────────────────────────────


class TestCacheStats:
    @patch("app.cache._get_redis", return_value=None)
    def test_stats_memory_backend(self, _):
        stats = cache_stats()
        assert stats["backend"] == "memory"
        assert isinstance(stats["entries"], int)
        assert stats["ttl_seconds"] > 0


# ── Cache warm-up ────────────────────────────────────────────────────


class TestWarmCache:
    @patch("app.cache.CACHE_WARMUP_ENABLED", True)
    @patch("app.rag.search_guidelines_with_timeout")
    def test_warm_cache_runs_all_queries(self, mock_search):
        mock_search.return_value = [{"chunk_id": "c1", "text": "t", "page": 1}]
        warmed = warm_cache()
        assert warmed == len(WARMUP_QUERIES)
        assert mock_search.call_count == len(WARMUP_QUERIES)

    @patch("app.cache.CACHE_WARMUP_ENABLED", False)
    def test_warm_cache_disabled(self):
        warmed = warm_cache()
        assert warmed == 0

    @patch("app.cache.CACHE_WARMUP_ENABLED", True)
    @patch("app.rag.search_guidelines_with_timeout")
    def test_warm_cache_handles_failures(self, mock_search):
        mock_search.side_effect = Exception("model not loaded")
        warmed = warm_cache()
        assert warmed == 0

    @patch("app.cache.CACHE_WARMUP_ENABLED", True)
    @patch("app.rag.search_guidelines_with_timeout")
    def test_warm_cache_counts_nonempty_only(self, mock_search):
        # First 5 return results, last 5 return empty
        side_effects = (
            [[{"chunk_id": "c1", "text": "t", "page": 1}]] * 5
            + [[]] * 5
        )
        mock_search.side_effect = side_effects
        warmed = warm_cache()
        assert warmed == 5

    def test_warmup_queries_nonempty(self):
        assert len(WARMUP_QUERIES) > 0
        for q in WARMUP_QUERIES:
            assert isinstance(q, str)
            assert len(q) > 10
