"""
src/experiments/run_baseline.py
────────────────────────────────
Run the BASELINE RAG experiment and save evaluation results.

Baseline config:
  - chunk_size    = 500
  - chunk_overlap = 50
  - embedding     = BAAI/bge-m3
  - retrieval     = dense (FAISS cosine)
  - top_k         = 5
  - rerank        = False
  - LLM           = qwen2.5:7b

Usage:
    python -m src.experiments.run_baseline --server http://<server_ip>:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.client import RAGClient
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "baseline"
BASELINE_CONFIG = {
    "chunk_size":    500,
    "chunk_overlap": 50,
    "top_k":         5,
    "retrieval_mode": "dense",
    "rerank":        False,
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline RAG evaluation")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="RAG backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/eval_dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-judge generation metrics (faster)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval (no LLM generation at all)",
    )
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    client = RAGClient(base_url=args.server)
    if not client.is_alive():
        logger.error("Cannot reach server at %s — is it running?", args.server)
        sys.exit(1)

    logger.info("Connected to server: %s", args.server)

    # ── Apply baseline config on server ───────────────────────────────────────
    logger.info("Applying baseline config: %s", BASELINE_CONFIG)
    client.update_config(
        chunk_size=BASELINE_CONFIG["chunk_size"],
        chunk_overlap=BASELINE_CONFIG["chunk_overlap"],
        top_k=BASELINE_CONFIG["top_k"],
        retrieval_mode=BASELINE_CONFIG["retrieval_mode"],
        reranker_enabled=False,
    )

    # ── Print index info ───────────────────────────────────────────────────────
    info = client.index_info()
    logger.info(
        "Index: %d chunks across %d documents",
        info["num_chunks"], info["num_documents"],
    )
    if info["num_chunks"] == 0:
        logger.warning(
            "Index is empty! Please upload documents first:\n"
            "  python -m src.experiments.ingest_docs --server %s --docs <path_to_docs>",
            args.server,
        )
        sys.exit(1)

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = Evaluator(
        client=client,
        dataset_path=args.dataset,
        top_k=BASELINE_CONFIG["top_k"],
        mode=BASELINE_CONFIG["retrieval_mode"],
        rerank=BASELINE_CONFIG["rerank"],
        run_generation=not args.retrieval_only,
        run_judge=not args.no_judge,
    )

    results = evaluator.run(
        experiment_name=EXPERIMENT_NAME,
        extra_config=BASELINE_CONFIG,
    )

    logger.info("Baseline evaluation complete.")
    return results


if __name__ == "__main__":
    main()
