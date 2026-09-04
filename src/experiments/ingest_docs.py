"""
src/experiments/ingest_docs.py
────────────────────────────────
Helper script to ingest all documents in a local folder to the server.

Usage:
    python -m src.experiments.ingest_docs \\
        --server http://<server>:8000 \\
        --docs server/data/documents/

    # With custom chunking:
    python -m src.experiments.ingest_docs \\
        --server http://<server>:8000 \\
        --docs /path/to/pdfs \\
        --chunk-size 300 \\
        --chunk-overlap 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.client import RAGClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG server")
    parser.add_argument("--server",       default="http://localhost:8000")
    parser.add_argument("--docs",         required=True, help="Path to documents folder")
    parser.add_argument("--chunk-size",   type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    client = RAGClient(base_url=args.server)
    if not client.is_alive():
        logger.error("Cannot reach server at %s", args.server)
        sys.exit(1)

    docs_path = Path(args.docs)
    if not docs_path.exists():
        logger.error("Docs path does not exist: %s", docs_path)
        sys.exit(1)

    files = [f for f in docs_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logger.warning("No supported files found in %s", docs_path)
        sys.exit(0)

    logger.info("Found %d files to ingest", len(files))
    total_chunks = 0

    for f in files:
        try:
            result = client.ingest_file(
                file_path=f,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            total_chunks += result["num_chunks"]
            logger.info(
                "  ✓ %s → %d chunks (%.2fs)",
                result["doc_id"], result["num_chunks"], result["embedding_time"]
            )
        except Exception as e:
            logger.error("  ✗ Failed to ingest %s: %s", f.name, e)

    logger.info("Done. Total chunks indexed: %d", total_chunks)
    info = client.index_info()
    logger.info("Index now: %d chunks, %d documents", info["num_chunks"], info["num_documents"])


if __name__ == "__main__":
    main()
