"""
server/pipeline.py
───────────────────
Modular RAG Pipeline Orchestrator.

Accepts a PipelineConfig and runs a query through the full pipeline:
  Query → Transform → Retrieve (×N) → Merge → Rerank → Process → Generate

Each step is pluggable — swap strategies without touching other steps.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from server.config import settings
from server.models import Chunk, PipelineConfig
from server.techniques.query_transformers import transform_query
from server.techniques.context_processors import process_context

logger = logging.getLogger(__name__)


# ── RRF for multi-query merging ────────────────────────────────────────────────

def _rrf_merge(
    ranked_lists: list[list[Chunk]],
    top_k: int,
    k: int = 60,
) -> list[Chunk]:
    """Merge multiple ranked chunk lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in best_chunk or chunk.score > best_chunk[cid].score:
                best_chunk[cid] = chunk

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        best_chunk[cid].model_copy(update={"score": rrf_score})
        for cid, rrf_score in merged
        if cid in best_chunk
    ]


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    config: PipelineConfig,
    generate: bool = True,
) -> dict[str, Any]:
    """
    Execute the full pluggable RAG pipeline.

    Args:
        query:    User question.
        config:   PipelineConfig specifying which strategy to use at each step.
        generate: If False, skip generation and return retrieval results only.

    Returns dict with: answer, chunks, retrieval_time, generation_time,
                       total_time, model, pipeline_config, query_variants
    """
    t_total = time.perf_counter()

    top_k = config.top_k or settings.top_k

    # ── Step 1: Query Transformation ──────────────────────────────────────────
    logger.info("[Pipeline] query_transformer=%s query=%s…", config.query_transformer, query[:60])
    query_variants = transform_query(
        query,
        strategy=config.query_transformer,
        n=config.multi_query_n,
    )

    # ── Step 2: Retrieval (one call per query variant) ────────────────────────
    from server.retriever import retrieve as _retrieve

    t_ret = time.perf_counter()
    all_ranked: list[list[Chunk]] = []
    for q in query_variants:
        chunks, _ = _retrieve(
            query=q,
            top_k=top_k * 2,          # fetch extra for RRF pool
            mode=config.retriever,
            filters=config.filters or {},
        )
        all_ranked.append(chunks)

    # Merge if multiple queries, otherwise just take the single list
    if len(all_ranked) == 1:
        retrieved = all_ranked[0][:top_k]
    else:
        retrieved = _rrf_merge(all_ranked, top_k=top_k)

    retrieval_time = time.perf_counter() - t_ret
    logger.info("[Pipeline] retrieved %d chunks in %.3fs", len(retrieved), retrieval_time)

    # ── Step 3: Reranking ─────────────────────────────────────────────────────
    if config.reranker != "none" and retrieved:
        from server.reranker import rerank as _rerank
        logger.info("[Pipeline] reranking with %s", config.reranker)
        retrieved = _rerank(query, retrieved, top_k=top_k)

    # ── Step 4: Context Processing ────────────────────────────────────────────
    logger.info("[Pipeline] context_processor=%s", config.context_processor)
    processed = process_context(
        chunks=retrieved,
        query=query,
        strategy=config.context_processor,
        window_size=config.window_size,
    )

    if not generate:
        return {
            "query":          query,
            "query_variants": query_variants,
            "chunks":         [c.model_dump() for c in processed],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "total_time":     time.perf_counter() - t_total,
            "pipeline_config": config.model_dump(),
        }

    # ── Step 5: Generation ────────────────────────────────────────────────────
    from server.generator import generate as _generate

    t_gen = time.perf_counter()
    gen_result = _generate(
        query=query,
        chunks=processed,
        system_prompt=config.system_prompt,
    )
    generation_time = time.perf_counter() - t_gen

    return {
        "query":           query,
        "query_variants":  query_variants,
        "answer":          gen_result["answer"],
        "chunks":          [c.model_dump() for c in processed],
        "retrieval_time":  retrieval_time,
        "generation_time": generation_time,
        "total_time":      time.perf_counter() - t_total,
        "model":           gen_result["model"],
        "pipeline_config": config.model_dump(),
    }
