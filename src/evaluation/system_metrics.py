"""
src/evaluation/system_metrics.py
──────────────────────────────────
System-level performance metrics:
  - latency breakdown (embedding, retrieval, generation, total)
  - memory usage (RSS)
  - token counts
  - chunk count statistics

These are measured on the CLIENT side by wrapping API responses.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

import psutil


# ── Per-query system record ────────────────────────────────────────────────────

def build_system_record(
    response: dict[str, Any],
    memory_before_mb: float,
    memory_after_mb: float,
) -> dict[str, Any]:
    """
    Build a system metrics record from a QueryResponse dict + memory snapshots.

    Expected keys in `response`:
        retrieval_time, generation_time, total_time, num_retrieved
    """
    return {
        "retrieval_time_s":    response.get("retrieval_time", 0.0),
        "generation_time_s":   response.get("generation_time", 0.0),
        "total_latency_s":     response.get("total_time", 0.0),
        "num_retrieved_chunks": response.get("num_retrieved", 0),
        "memory_delta_mb":     round(memory_after_mb - memory_before_mb, 2),
        "memory_after_mb":     round(memory_after_mb, 2),
    }


# ── Memory helper ─────────────────────────────────────────────────────────────

def get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_system_metrics(
    per_query_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute mean, median, p95 for latency metrics across the eval set.
    """
    if not per_query_records:
        return {}

    def _stats(key: str) -> dict[str, float]:
        values = [r[key] for r in per_query_records if key in r]
        if not values:
            return {}
        sorted_v = sorted(values)
        p95_idx  = int(len(sorted_v) * 0.95)
        return {
            "mean":   round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "p95":    round(sorted_v[min(p95_idx, len(sorted_v) - 1)], 4),
            "max":    round(max(values), 4),
        }

    return {
        "retrieval_time_s":     _stats("retrieval_time_s"),
        "generation_time_s":    _stats("generation_time_s"),
        "total_latency_s":      _stats("total_latency_s"),
        "num_retrieved_chunks": _stats("num_retrieved_chunks"),
        "memory_delta_mb":      _stats("memory_delta_mb"),
        "num_samples":          len(per_query_records),
    }
