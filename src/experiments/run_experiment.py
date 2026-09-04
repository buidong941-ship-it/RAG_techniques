"""
src/experiments/run_experiment.py
──────────────────────────────────
Generic experiment runner for any RAG technique.

Usage:
    python -m src.experiments.run_experiment \\
        --server http://<server>:8000 \\
        --name hybrid_retrieval \\
        --mode hybrid \\
        --top-k 10 \\
        --rerank

    # Or with a full config JSON:
    python -m src.experiments.run_experiment \\
        --server http://<server>:8000 \\
        --name chunk_size_200 \\
        --config '{"chunk_size": 200, "chunk_overlap": 20, "top_k": 5}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.client import RAGClient
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a RAG experiment")
    parser.add_argument("--server",   default="http://localhost:8000")
    parser.add_argument("--dataset",  default="data/evaluation/eval_dataset.json")
    parser.add_argument("--name",     required=True, help="Experiment name (used as results folder)")
    parser.add_argument("--mode",     default=None,  choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--top-k",    type=int, default=None)
    parser.add_argument("--rerank",   action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--config",
        default=None,
        help="JSON string of server config overrides, e.g. '{\"chunk_size\": 200}'",
    )
    args = parser.parse_args()

    client = RAGClient(base_url=args.server)
    if not client.is_alive():
        logger.error("Cannot reach server at %s", args.server)
        sys.exit(1)

    # Build config overrides
    extra_config: dict = {}
    if args.config:
        extra_config = json.loads(args.config)
    if args.mode:
        extra_config["retrieval_mode"] = args.mode
    if args.top_k:
        extra_config["top_k"] = args.top_k
    if args.rerank:
        extra_config["reranker_enabled"] = True

    # Apply config overrides on server
    if extra_config:
        logger.info("Applying config overrides: %s", extra_config)
        client.update_config(**extra_config)

    # Run evaluation
    evaluator = Evaluator(
        client=client,
        dataset_path=args.dataset,
        top_k=args.top_k,
        mode=args.mode,
        rerank=args.rerank,
        run_generation=not args.retrieval_only,
        run_judge=not args.no_judge,
    )

    evaluator.run(experiment_name=args.name, extra_config=extra_config)
    logger.info("Experiment '%s' complete.", args.name)


if __name__ == "__main__":
    main()
