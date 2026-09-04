"""
src/evaluation/retrieval_metrics.py
────────────────────────────────────
Retrieval quality metrics — computed WITHOUT needing an LLM.

Metrics implemented:
  - Recall@K      How many relevant chunks were retrieved?
  - Precision@K   How many retrieved chunks are relevant?
  - MRR           Mean Reciprocal Rank (first relevant at what rank?)
  - NDCG@K        Normalized Discounted Cumulative Gain

All functions accept:
  retrieved_ids:   List of chunk IDs returned by the retriever (ranked order)
  relevant_ids:    List of chunk IDs that are ground-truth relevant
  k:               Cut-off (same as top_k used during retrieval)

Each function returns a float in [0, 1].
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Answers: "Of all the relevant chunks, what fraction did we retrieve?"
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set  = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / k

    Answers: "Of what we retrieved, how much of it is actually relevant?"
    """
    if not retrieved_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    relevant_set   = set(relevant_ids)
    hits = sum(1 for cid in retrieved_at_k if cid in relevant_set)
    return hits / len(retrieved_at_k)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank (for a single query).

    MRR = 1 / rank_of_first_relevant_result

    Answers: "How high up is the first relevant chunk?"
    """
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    NDCG@K — Normalized Discounted Cumulative Gain.

    Assumes binary relevance (1 if in relevant_ids, 0 otherwise).
    Penalises relevant results that appear lower in the ranking.

    Answers: "How well-ordered is the ranking of relevant results?"
    """
    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for rank, cid in enumerate(retrieved_at_k, start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG (all relevant results at top)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Returns a dict ready for JSON serialisation.
    """
    return {
        f"recall@{k}":    recall_at_k(retrieved_ids, relevant_ids, k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr":            mrr(retrieved_ids, relevant_ids),
        f"ndcg@{k}":      ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def aggregate_retrieval_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average per-query metrics across the full evaluation set.

    Input: list of dicts from compute_retrieval_metrics()
    Output: dict of mean values.
    """
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {
        key: sum(m[key] for m in per_query_metrics) / len(per_query_metrics)
        for key in keys
    }
