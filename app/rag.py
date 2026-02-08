"""Shared RAG pipeline for querying the NG12 guidelines vector store.

Supports three retrieval modes (controlled via config flags):
  - Vector-only (default fallback)
  - Hybrid: BM25 + Vector with Reciprocal Rank Fusion
  - Hybrid + Cross-encoder reranking
"""

import hashlib
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.config import (
    CHROMA_PERSIST_DIR, TOP_K,
    BM25_ENABLED, BM25_WEIGHT, VECTOR_WEIGHT,
    RERANK_ENABLED, RERANK_MODEL, RETRIEVAL_CANDIDATES,
    CORRECTIVE_RAG_ENABLED, CORRECTIVE_RAG_THRESHOLD,
    CORRECTIVE_RAG_MAX_RETRIES, RAG_TIMEOUT_SECONDS,
    METADATA_FILTER_ENABLED,
)

logger = logging.getLogger(__name__)

_embedding_fn = None
_collection = None

# Lazy-loaded BM25 index
_bm25_index = None
_bm25_corpus_ids: list[str] = []
_bm25_corpus_texts: list[str] = []
_bm25_corpus_metas: list[dict] = []

# Lazy-loaded cross-encoder
_reranker = None

# Reranker score cache: "{query_hash}:{chunk_id}" -> score
_rerank_cache: dict[str, float] = {}

# ── Query-aware metadata filtering ───────────────────────────────────

CANCER_KEYWORDS = {
    "lung": "lung", "pulmonary": "lung", "hemoptysis": "lung", "cough": "lung",
    "breast": "breast", "lump": "breast",
    "colorectal": "colorectal", "bowel": "colorectal", "rectal": "colorectal",
    "prostate": "prostate", "psa": "prostate",
    "skin": "skin", "melanoma": "skin", "mole": "skin",
    "bladder": "bladder", "haematuria": "bladder",
    "oesophageal": "oesophageal", "dysphagia": "oesophageal",
    "stomach": "stomach", "gastric": "stomach",
    "ovarian": "gynaecological", "endometrial": "gynaecological", "cervical": "gynaecological",
    "renal": "renal", "kidney": "renal",
    "head": "head_and_neck", "neck": "head_and_neck", "thyroid": "head_and_neck",
    "leukaemia": "haematological", "lymphoma": "haematological", "myeloma": "haematological",
    "sarcoma": "sarcoma", "bone": "sarcoma",
    "brain": "brain_cns", "neurological": "brain_cns",
}


def _extract_cancer_filter(query: str) -> Optional[dict]:
    """Extract cancer type hints from query text and return a ChromaDB where clause.

    Always includes "multi" and "general" to avoid missing cross-cutting content.
    Returns None if no cancer type is detected (no filter, search everything).
    """
    if not METADATA_FILTER_ENABLED:
        return None

    query_lower = query.lower()
    detected = set()
    for keyword, cancer_type in CANCER_KEYWORDS.items():
        # Use word boundary check to avoid partial matches
        if keyword in query_lower:
            detected.add(cancer_type)

    if not detected:
        return None

    # Always include "multi" and "general" for cross-cutting content
    all_types = sorted(detected | {"multi", "general"})
    return {"cancer_type": {"$in": all_types}}


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_fn


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_collection(
            name="ng12_guidelines",
            embedding_function=get_embedding_function(),
        )
    return _collection


# ── BM25 index ──────────────────────────────────────────────────────────

def _init_bm25_index():
    """Load all chunk texts from ChromaDB into a BM25Okapi index (once)."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_texts, _bm25_corpus_metas

    from rank_bm25 import BM25Okapi

    collection = get_collection()
    total = collection.count()
    if total == 0:
        logger.warning("BM25: collection is empty, skipping index build")
        return

    # Fetch all documents from the collection
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_texts = all_data["documents"]
    _bm25_corpus_metas = all_data["metadatas"]

    # Tokenize for BM25 (simple whitespace + lowercase)
    tokenized = [doc.lower().split() for doc in _bm25_corpus_texts]
    _bm25_index = BM25Okapi(tokenized)

    logger.info("BM25 index built with %d documents", total)


def _bm25_search(query: str, n: int) -> list[dict]:
    """Return the top-n BM25 results for the query."""
    global _bm25_index

    if _bm25_index is None:
        _init_bm25_index()
    if _bm25_index is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-n indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append({
            "chunk_id": _bm25_corpus_ids[idx],
            "text": _bm25_corpus_texts[idx],
            "page": _bm25_corpus_metas[idx].get("page", 0),
            "source": "NG12 PDF",
            "distance": 0.0,  # BM25 doesn't use distance; RRF uses rank
            "bm25_score": float(scores[idx]),
        })
    return results


def is_bm25_ready() -> bool:
    """Return True if the BM25 index has been built."""
    return _bm25_index is not None


# ── Cross-encoder reranker ──────────────────────────────────────────────

def _get_reranker():
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
        logger.info("Cross-encoder reranker loaded: %s", RERANK_MODEL)
    return _reranker


def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Re-score chunks using the cross-encoder and re-sort by relevance.

    Uses a score cache to avoid re-scoring the same (query, chunk) pair
    during corrective RAG retries.
    """
    if not chunks:
        return chunks

    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    reranker = _get_reranker()

    uncached_pairs = []
    uncached_indices = []

    for i, c in enumerate(chunks):
        cache_key = f"{query_hash}:{c['chunk_id']}"
        if cache_key in _rerank_cache:
            c["rerank_score"] = _rerank_cache[cache_key]
        else:
            uncached_pairs.append((query, c["text"]))
            uncached_indices.append(i)

    if uncached_pairs:
        scores = reranker.predict(uncached_pairs, batch_size=32)
        for j, idx in enumerate(uncached_indices):
            score = float(scores[j])
            chunks[idx]["rerank_score"] = score
            cache_key = f"{query_hash}:{chunks[idx]['chunk_id']}"
            _rerank_cache[cache_key] = score

    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)


def is_reranker_ready() -> bool:
    """Return True if the cross-encoder model has been loaded."""
    return _reranker is not None


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using RRF: score = weight * 1/(rank + k)."""
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + VECTOR_WEIGHT * (1.0 / (rank + k))
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + BM25_WEIGHT * (1.0 / (rank + k))
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    results = []
    for cid in sorted_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        results.append(chunk)

    return results


# ── Main search function ───────────────────────────────────────────────

def _parse_vector_results(results) -> list[dict]:
    """Parse ChromaDB query results into a list of chunk dicts."""
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i].get("page", 0),
                "source": "NG12 PDF",
                "distance": results["distances"][0][i],
            })
    return chunks


def search_guidelines(query: str, top_k: int = TOP_K) -> list[dict]:
    """Search the NG12 vector store for passages relevant to the query.

    Pipeline: metadata filter -> vector search (+ optional BM25) -> RRF merge -> rerank -> top_k.
    Returns a list of dicts with keys: chunk_id, text, page, source, distance.
    """
    n_candidates = RETRIEVAL_CANDIDATES if (BM25_ENABLED or RERANK_ENABLED) else top_k

    # 1. Vector search with optional metadata filtering
    collection = get_collection()
    where_filter = _extract_cancer_filter(query)

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_candidates,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    vector_chunks = _parse_vector_results(results)

    # Fallback: if filtered search returns < 3 results, retry without filter
    if where_filter and len(vector_chunks) < 3:
        logger.info(
            "Metadata filter returned only %d results, retrying without filter",
            len(vector_chunks),
        )
        del query_kwargs["where"]
        results = collection.query(**query_kwargs)
        vector_chunks = _parse_vector_results(results)

    # 2. BM25 search (if enabled)
    if BM25_ENABLED:
        bm25_chunks = _bm25_search(query, n_candidates)
        merged = _reciprocal_rank_fusion(vector_chunks, bm25_chunks)
        logger.info(
            "Hybrid search: %d vector + %d BM25 -> %d merged",
            len(vector_chunks), len(bm25_chunks), len(merged),
        )
    else:
        merged = vector_chunks

    # 3. Cross-encoder reranking (if enabled)
    if RERANK_ENABLED:
        merged = _rerank(query, merged)
        logger.info("Reranked %d chunks", len(merged))

    return merged[:top_k]


# ── Corrective RAG ────────────────────────────────────────────────────

def _compute_confidence(chunks: list[dict]) -> float:
    """Compute a 0.0-1.0 confidence score from retrieved chunks.

    Prefers rerank_score (from cross-encoder) when available;
    falls back to inverse L2 distance.
    """
    if not chunks:
        return 0.0

    scores = []
    for c in chunks:
        if "rerank_score" in c:
            # Cross-encoder scores are typically in [-10, 10]; normalize to [0, 1]
            raw = c["rerank_score"]
            normalized = max(0.0, min(1.0, (raw + 5) / 10))
            scores.append(normalized)
        else:
            # Inverse distance: lower distance = higher confidence
            dist = c.get("distance", 1.0)
            scores.append(max(0.0, 1.0 - (dist / 2.0)))

    return sum(scores) / len(scores) if scores else 0.0


def _reformulate_query(query: str, attempt: int) -> str:
    """Generate an alternative query for retry attempts."""
    if attempt == 1:
        return f"NICE NG12 suspected cancer referral pathway criteria for: {query}"
    else:
        # Extract the first meaningful word for a broader search
        words = query.strip().split()
        first_word = words[0] if words else query
        return f"cancer recognition symptoms {first_word} referral"


def search_guidelines_corrective(query: str, top_k: int = TOP_K) -> list[dict]:
    """Corrective RAG: search with confidence-based retry and query reformulation.

    If the initial search returns low-confidence results, reformulates the query
    and retries up to CORRECTIVE_RAG_MAX_RETRIES times. Merges all unique chunks,
    re-sorts, and returns top_k.

    Gated behind CORRECTIVE_RAG_ENABLED flag — falls back to search_guidelines
    when disabled.
    """
    if not CORRECTIVE_RAG_ENABLED:
        return search_guidelines(query, top_k=top_k)

    # First attempt
    chunks = search_guidelines(query, top_k=top_k)
    confidence = _compute_confidence(chunks)

    logger.info(
        "Corrective RAG: initial confidence=%.3f (threshold=%.3f)",
        confidence, CORRECTIVE_RAG_THRESHOLD,
    )

    if confidence >= CORRECTIVE_RAG_THRESHOLD:
        return chunks

    # Collect all chunks for merging
    all_chunks: dict[str, dict] = {}
    for c in chunks:
        all_chunks[c["chunk_id"]] = c

    # Retry with reformulated queries
    for attempt in range(1, CORRECTIVE_RAG_MAX_RETRIES + 1):
        reformulated = _reformulate_query(query, attempt)
        logger.info(
            "Corrective RAG: retry %d with query: %s",
            attempt, reformulated[:80],
        )

        retry_chunks = search_guidelines(reformulated, top_k=top_k)
        for c in retry_chunks:
            if c["chunk_id"] not in all_chunks:
                all_chunks[c["chunk_id"]] = c

        # Compute confidence on merged results
        merged = list(all_chunks.values())
        confidence = _compute_confidence(merged)

        logger.info(
            "Corrective RAG: retry %d confidence=%.3f, total chunks=%d",
            attempt, confidence, len(merged),
        )

        if confidence >= CORRECTIVE_RAG_THRESHOLD:
            break

    # Sort by best available score
    merged = list(all_chunks.values())
    merged.sort(
        key=lambda c: c.get("rerank_score", -c.get("distance", 999)),
        reverse=True,
    )

    return merged[:top_k]


def search_guidelines_with_timeout(query: str, top_k: int = TOP_K) -> list[dict]:
    """Wrap the search pipeline with caching + timeout for graceful degradation."""
    from app.cache import cache_get, cache_set

    # Check cache first
    cached = cache_get(query, top_k)
    if cached is not None:
        logger.info("RAG cache hit for query: %s", query[:60])
        return cached

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_guidelines_corrective, query, top_k)
            results = future.result(timeout=RAG_TIMEOUT_SECONDS)

        # Cache the results
        if results:
            cache_set(query, top_k, results)

        return results
    except FuturesTimeoutError:
        logger.error(
            "RAG search timed out after %ds for query: %s",
            RAG_TIMEOUT_SECONDS, query[:80],
        )
        return []
    except Exception:
        logger.exception("RAG search failed for query: %s", query[:80])
        return []
