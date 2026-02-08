"""RAG query cache — avoids redundant vector store + LLM calls for repeated queries.

Uses Redis when available, falls back to in-memory LRU dict.
Cache keys are SHA-256 hashes of (query, top_k).
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

from app.config import REDIS_HOST, REDIS_PORT, CACHE_WARMUP_ENABLED

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "rag_cache:"
_CACHE_TTL = 300  # 5 minutes
_MAX_MEMORY_ENTRIES = 200

# ── Redis client (shared with chat_agent, lazy init) ─────────────────

_redis_client = None
_redis_checked = False


def _get_redis():
    """Lazy-init Redis client for caching (separate from session store)."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=1,
            decode_responses=True, socket_connect_timeout=2,
        )
        client.ping()
        _redis_client = client
        logger.info("RAG cache: Redis connected at %s:%s (db=1)", REDIS_HOST, REDIS_PORT)
    except Exception:
        _redis_client = None
        logger.info("RAG cache: using in-memory fallback")
    return _redis_client


# ── In-memory LRU fallback ───────────────────────────────────────────

_memory_cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()


def _cache_key(query: str, top_k: int) -> str:
    """Generate a deterministic cache key from query + top_k."""
    raw = f"{query.strip().lower()}|{top_k}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_get(query: str, top_k: int) -> Optional[list[dict]]:
    """Look up cached RAG results. Returns None on miss."""
    key = _cache_key(query, top_k)

    # Try Redis first
    redis_client = _get_redis()
    if redis_client:
        try:
            raw = redis_client.get(f"{_CACHE_PREFIX}{key}")
            if raw:
                logger.debug("RAG cache HIT (Redis): %s", key)
                return json.loads(raw)
        except Exception:
            pass

    # Fall back to in-memory
    if key in _memory_cache:
        ts, data = _memory_cache[key]
        if time.monotonic() - ts < _CACHE_TTL:
            logger.debug("RAG cache HIT (memory): %s", key)
            _memory_cache.move_to_end(key)
            return data
        else:
            del _memory_cache[key]

    return None


def cache_set(query: str, top_k: int, results: list[dict]) -> None:
    """Store RAG results in cache."""
    key = _cache_key(query, top_k)

    # Try Redis
    redis_client = _get_redis()
    if redis_client:
        try:
            redis_client.set(
                f"{_CACHE_PREFIX}{key}",
                json.dumps(results, default=str),
                ex=_CACHE_TTL,
            )
            return
        except Exception:
            pass

    # Fall back to in-memory
    _memory_cache[key] = (time.monotonic(), results)
    if len(_memory_cache) > _MAX_MEMORY_ENTRIES:
        _memory_cache.popitem(last=False)


def cache_stats() -> dict:
    """Return cache stats for health checks."""
    redis_client = _get_redis()
    stats = {"backend": "redis" if redis_client else "memory"}
    if redis_client:
        try:
            keys = redis_client.keys(f"{_CACHE_PREFIX}*")
            stats["entries"] = len(keys)
        except Exception:
            stats["entries"] = "unknown"
    else:
        stats["entries"] = len(_memory_cache)
    stats["ttl_seconds"] = _CACHE_TTL
    return stats


# ── Cache warm-up ────────────────────────────────────────────────────

WARMUP_QUERIES = [
    "lung cancer symptoms urgent referral criteria",
    "breast cancer lump referral two week wait",
    "colorectal cancer bowel habit change referral",
    "prostate cancer PSA referral criteria",
    "skin cancer melanoma suspicious mole referral",
    "bladder cancer haematuria referral criteria",
    "oesophageal cancer dysphagia referral",
    "head and neck cancer symptoms referral",
    "haematological cancer lymphoma leukaemia referral",
    "unexplained weight loss cancer referral criteria",
]


def warm_cache() -> int:
    """Pre-run common queries to populate the cache.

    Runs sequentially to avoid overloading the CPU with reranker calls.
    Returns the number of queries warmed.
    """
    if not CACHE_WARMUP_ENABLED:
        logger.info("Cache warm-up disabled")
        return 0

    from app.rag import search_guidelines_with_timeout

    warmed = 0
    for query in WARMUP_QUERIES:
        try:
            results = search_guidelines_with_timeout(query)
            if results:
                warmed += 1
                logger.info("Cache warmed: %s (%d results)", query[:50], len(results))
            else:
                logger.warning("Cache warm-up returned no results: %s", query[:50])
        except Exception:
            logger.exception("Cache warm-up failed for: %s", query[:50])

    logger.info("Cache warm-up complete: %d/%d queries warmed", warmed, len(WARMUP_QUERIES))
    return warmed
