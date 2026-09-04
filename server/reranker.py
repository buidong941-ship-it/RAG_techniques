"""
server/reranker.py
──────────────────
Cross-encoder reranking using BAAI/bge-reranker-v2-m3.

This module is optional — it is only loaded when reranker_enabled=True
in config (or per-request via the `rerank` flag).

BGE-reranker-v2-m3 is a multilingual cross-encoder that scores
(query, passage) pairs directly. It supports Vietnamese.
"""

from __future__ import annotations

import logging
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from server.config import settings
from server.models import Chunk

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

_reranker_model = None
_reranker_tokenizer = None


def _load_reranker():
    """Load the cross-encoder model (lazy, cached)."""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        logger.info("Loading reranker model: %s", settings.reranker_model)
        _reranker_tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            settings.reranker_model,
        )
        device = settings.reranker_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU for reranker")
            device = "cpu"
        _reranker_model = _reranker_model.to(device)
        _reranker_model.eval()
        logger.info("Reranker loaded on %s", device)
    return _reranker_model, _reranker_tokenizer


def rerank(query: str, chunks: list[Chunk], top_k: int | None = None) -> tuple[list[Chunk], float]:
    """
    Rerank `chunks` using the cross-encoder.

    Returns (reranked_chunks, reranking_time_seconds).
    The returned list is sorted by descending cross-encoder score.
    """
    if not chunks:
        return chunks, 0.0

    model, tokenizer = _load_reranker()
    device = next(model.parameters()).device

    pairs = [[query, c.text] for c in chunks]

    t0 = time.perf_counter()

    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits.squeeze(-1)  # shape: (N,)
        scores = logits.float().cpu().tolist()

    elapsed = time.perf_counter() - t0

    # Attach reranker scores and sort
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    reranked = [c.model_copy(update={"score": s}) for c, s in scored]
    if top_k:
        reranked = reranked[:top_k]

    logger.debug("Reranked %d chunks in %.3fs", len(reranked), elapsed)
    return reranked, elapsed
