"""
server/retriever.py
───────────────────
Retrieval layer supporting three modes:
  - dense:  FAISS cosine similarity (BGE-M3 vectors)
  - sparse: BM25 (rank_bm25)
  - hybrid: Reciprocal Rank Fusion (RRF) of dense + sparse

Design: plain Python, no framework. Swap retrieval_mode per experiment.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

import faiss

from server.config import settings
from server.indexer import (
    embed_texts,
    get_chunk_ids,
    get_chunk_store,
    get_index,
)
from server.models import Chunk

logger = logging.getLogger(__name__)


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _build_bm25() -> tuple[BM25Okapi, list[str]]:
    """Build a BM25 index from the current chunk store. O(N) rebuild."""
    chunk_ids = get_chunk_ids()
    chunk_store = get_chunk_store()
    tokenized = [
        chunk_store[cid].text.lower().split()
        for cid in chunk_ids
    ]
    return BM25Okapi(tokenized), chunk_ids


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int = 60,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    alpha: weight given to dense ranking (1-alpha for sparse)
    Returns list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + alpha / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Core retrieval functions ───────────────────────────────────────────────────

def dense_retrieve(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    FAISS inner-product search (cosine similarity after L2 normalisation).

    filters: optional dict like {"doc_id": "report.pdf"} to restrict results.
    """
    index = get_index()
    chunk_store = get_chunk_store()
    chunk_ids = get_chunk_ids()

    if index.ntotal == 0:
        return []

    query_vec = embed_texts([query])  # shape (1, dim), normalised
    scores, idxs = index.search(query_vec, min(top_k * 3, index.ntotal))

    results: list[Chunk] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        cid = chunk_ids[idx]
        chunk = chunk_store[cid].model_copy(update={"score": float(score)})
        if _passes_filter(chunk, filters):
            results.append(chunk)
        if len(results) >= top_k:
            break

    return results


def sparse_retrieve(
    query: str,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """BM25 retrieval over the current chunk store."""
    chunk_store = get_chunk_store()
    if not chunk_store:
        return []

    bm25, chunk_ids = _build_bm25()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    ranked_idxs = np.argsort(bm25_scores)[::-1]
    results: list[Chunk] = []
    for idx in ranked_idxs:
        cid = chunk_ids[idx]
        score = float(bm25_scores[idx])
        chunk = chunk_store[cid].model_copy(update={"score": score})
        if _passes_filter(chunk, filters):
            results.append(chunk)
        if len(results) >= top_k:
            break

    return results


def hybrid_retrieve(
    query: str,
    top_k: int,
    alpha: float | None = None,
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Hybrid retrieval: RRF over dense + sparse.

    alpha: weight for dense (0=pure sparse, 1=pure dense).
    """
    alpha = alpha if alpha is not None else settings.hybrid_alpha
    chunk_store = get_chunk_store()

    # Retrieve extended candidate pools
    pool = min(top_k * 5, len(chunk_store)) if chunk_store else top_k
    dense_chunks  = dense_retrieve(query, top_k=pool, filters=filters)
    sparse_chunks = sparse_retrieve(query, top_k=pool, filters=filters)

    dense_ids  = [c.chunk_id for c in dense_chunks]
    sparse_ids = [c.chunk_id for c in sparse_chunks]

    fused = _rrf(dense_ids, sparse_ids, alpha=alpha)

    results: list[Chunk] = []
    for cid, rrf_score in fused[:top_k]:
        chunk = chunk_store[cid].model_copy(update={"score": rrf_score})
        results.append(chunk)

    return results


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    mode: str | None = None,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Chunk], float]:
    """
    Unified retrieval entry point.

    Returns (chunks, retrieval_time_seconds).
    """
    top_k = top_k or settings.top_k
    mode  = mode  or settings.retrieval_mode

    t0 = time.perf_counter()

    if mode == "dense":
        chunks = dense_retrieve(query, top_k, filters)
    elif mode == "sparse":
        chunks = sparse_retrieve(query, top_k, filters)
    elif mode == "hybrid":
        chunks = hybrid_retrieve(query, top_k, filters=filters)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    elapsed = time.perf_counter() - t0
    logger.debug("Retrieved %d chunks in %.3fs (mode=%s)", len(chunks), elapsed, mode)
    return chunks, elapsed


# ── Filter helper ─────────────────────────────────────────────────────────────

def _passes_filter(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """Return True if the chunk satisfies all filter conditions."""
    if not filters:
        return True
    meta = chunk.metadata
    for key, value in filters.items():
        chunk_val = getattr(meta, key, meta.extra.get(key))
        if chunk_val != value:
            return False
    return True
