"""
src/evaluation/evaluator.py
────────────────────────────
Main evaluation orchestrator.

Usage:
    from src.evaluation.evaluator import Evaluator
    from src.client import RAGClient

    client   = RAGClient("http://server:8000")
    eval_run = Evaluator(client, dataset_path="data/evaluation/eval_dataset.json")
    results  = eval_run.run(experiment_name="baseline")

Results are saved to: experiments/results/<experiment_name>/metrics.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.client import RAGClient
from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    aggregate_retrieval_metrics,
)
from src.evaluation.generation_metrics import (
    compute_generation_metrics,
    aggregate_generation_metrics,
    diagnose_failure,
)
from src.evaluation.system_metrics import (
    build_system_record,
    aggregate_system_metrics,
    get_memory_mb,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")


class Evaluator:
    """
    Runs a full evaluation pass over the fixed evaluation dataset.

    Args:
        client:           RAGClient instance pointing at the server.
        dataset_path:     Path to eval_dataset.json.
        top_k:            Override retrieval top_k.
        mode:             Override retrieval mode ("dense"/"sparse"/"hybrid").
        rerank:           Enable reranking.
        run_generation:   If False, only run retrieval metrics (faster).
        run_judge:        If False, skip LLM-judge metrics (fastest).
    """

    def __init__(
        self,
        client: RAGClient,
        dataset_path: str | Path = "data/evaluation/eval_dataset.json",
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool = False,
        run_generation: bool = True,
        run_judge: bool = True,
    ):
        self.client     = client
        self.dataset    = self._load_dataset(Path(dataset_path))
        self.top_k      = top_k
        self.mode       = mode
        self.rerank     = rerank
        self.run_generation = run_generation
        self.run_judge  = run_judge

    @staticmethod
    def _load_dataset(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Eval dataset must be a JSON array")
        if len(data) == 0:
            logger.warning("Eval dataset is empty — add QA pairs to %s", path)
        else:
            logger.info("Loaded %d eval samples from %s", len(data), path)
        return data

    def run(
        self,
        experiment_name: str,
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run evaluation and return aggregated results.
        Also saves per-sample and aggregate results to disk.
        """
        logger.info("Starting evaluation: experiment=%s, n=%d", experiment_name, len(self.dataset))

        per_sample_results: list[dict[str, Any]] = []
        retrieval_metrics_list: list[dict[str, float]] = []
        generation_metrics_list: list[dict[str, float]] = []
        system_metrics_list: list[dict[str, Any]] = []

        for i, sample in enumerate(self.dataset):
            question        = sample["question"]
            ground_truth    = sample.get("ground_truth", "")
            relevant_chunks = sample.get("relevant_chunks", None)  # None = not provided
            has_retrieval_gt = relevant_chunks is not None and len(relevant_chunks) > 0

            logger.info("[%d/%d] Evaluating: %s", i + 1, len(self.dataset), question[:60])

            # ── Query ──────────────────────────────────────────────────────
            mem_before = get_memory_mb()
            try:
                if self.run_generation:
                    response = self.client.query(
                        query=question,
                        top_k=self.top_k,
                        mode=self.mode,
                        rerank=self.rerank,
                    )
                else:
                    response = self.client.retrieve(
                        query=question,
                        top_k=self.top_k,
                        mode=self.mode,
                        rerank=self.rerank,
                    )
            except Exception as e:
                logger.error("Query failed for sample %d: %s", i, e)
                continue
            mem_after = get_memory_mb()

            # ── Extract info ───────────────────────────────────────────────
            retrieved_ids = [c["chunk_id"] for c in response.get("chunks", [])]
            answer        = response.get("answer", "")
            context       = "\n\n".join(c["text"] for c in response.get("chunks", []))

            # ── Retrieval metrics ──────────────────────────────────────────────
            top_k_used = self.top_k or len(retrieved_ids)
            ret_metrics: dict[str, float] = {}
            if has_retrieval_gt:
                ret_metrics = compute_retrieval_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_chunks,
                    k=top_k_used,
                )
                retrieval_metrics_list.append(ret_metrics)

            # ── Generation metrics (LLM-judge) ─────────────────────────────
            gen_metrics: dict[str, float] = {}
            if self.run_generation and self.run_judge and answer:
                gen_metrics = compute_generation_metrics(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                    context=context,
                    judge_fn=self.client.judge,
                )
                generation_metrics_list.append(gen_metrics)

                # Failure mode — only meaningful if we have retrieval ground truth
                if has_retrieval_gt:
                    recall_k = ret_metrics.get(f"recall@{top_k_used}", 0.0)
                    gen_metrics["failure_mode"] = diagnose_failure(
                        retrieval_recall=recall_k,
                        generation_correctness=gen_metrics.get("answer_correctness", 0.0),
                    )
                else:
                    # Without retrieval GT, classify based on generation quality only
                    correctness = gen_metrics.get("answer_correctness", 0.0)
                    gen_metrics["failure_mode"] = "ok" if correctness >= 0.5 else "generation_failure"

            # ── System metrics ─────────────────────────────────────────────
            sys_record = build_system_record(response, mem_before, mem_after)
            system_metrics_list.append(sys_record)

            # ── Per-sample result ──────────────────────────────────────────
            per_sample_results.append({
                "sample_idx":      i,
                "question":        question,
                "ground_truth":    ground_truth,
                "answer":          answer,
                "retrieved_ids":   retrieved_ids,
                "has_retrieval_gt": has_retrieval_gt,
                **ret_metrics,
                **gen_metrics,
                **sys_record,
            })

        # ── Aggregate ──────────────────────────────────────────────────────
        agg_retrieval   = aggregate_retrieval_metrics(retrieval_metrics_list)
        agg_generation  = aggregate_generation_metrics(generation_metrics_list)
        agg_system      = aggregate_system_metrics(system_metrics_list)

        # Failure mode summary
        failure_counts: dict[str, int] = {}
        for s in per_sample_results:
            mode = s.get("failure_mode", "unknown")
            failure_counts[mode] = failure_counts.get(mode, 0) + 1

        results = {
            "experiment":         experiment_name,
            "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "top_k":   self.top_k,
                "mode":    self.mode,
                "rerank":  self.rerank,
                **(extra_config or {}),
            },
            "num_samples":        len(per_sample_results),
            "retrieval_metrics":  agg_retrieval,
            "generation_metrics": agg_generation,
            "system_metrics":     agg_system,
            "failure_analysis":   failure_counts,
            "per_sample":         per_sample_results,
        }

        # ── Save ───────────────────────────────────────────────────────────
        self._save_results(experiment_name, results)
        self._print_summary(results)
        return results

    @staticmethod
    def _save_results(experiment_name: str, results: dict[str, Any]) -> None:
        out_dir = RESULTS_DIR / experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved → %s", metrics_path)

    @staticmethod
    def _print_summary(results: dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print(f"  Experiment: {results['experiment']}")
        print(f"  Samples:    {results['num_samples']}")
        print(f"\n  📊 Retrieval Metrics")
        for k, v in results["retrieval_metrics"].items():
            print(f"    {k:<20} {v:.4f}")
        if results["generation_metrics"]:
            print(f"\n  🤖 Generation Metrics")
            for k, v in results["generation_metrics"].items():
                if not k.endswith("_coverage"):
                    print(f"    {k:<25} {v:.4f}")
        print(f"\n  ⏱  System Metrics")
        sys_m = results["system_metrics"]
        for key in ["retrieval_time_s", "generation_time_s", "total_latency_s"]:
            if key in sys_m:
                print(f"    {key:<25} mean={sys_m[key]['mean']:.3f}s  p95={sys_m[key]['p95']:.3f}s")
        print(f"\n  ⚠️  Failure Analysis")
        for mode, count in results["failure_analysis"].items():
            print(f"    {mode:<25} {count}")
        print(f"{'='*60}\n")
