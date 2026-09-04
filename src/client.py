"""
src/client.py
─────────────
Python client for the RAG backend server.

Wraps all HTTP calls in typed, easy-to-use methods.
Used by experiment scripts and the evaluation framework.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RAGClient:
    """
    Synchronous HTTP client for the RAG backend.

    Args:
        base_url: e.g. "http://192.168.1.100:8000"
        timeout:  Request timeout in seconds (generation can be slow)
    """

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        return response.json()

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
        return response.json()

    def _delete(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(url)
            response.raise_for_status()
        return response.json()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """
        Upload a local file to the server and ingest it.

        Uses base64 encoding to send file content in JSON.
        """
        file_path = Path(file_path)
        content_b64 = base64.b64encode(file_path.read_bytes()).decode()
        payload: dict[str, Any] = {
            "filename": file_path.name,
            "content_b64": content_b64,
        }
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            payload["chunk_overlap"] = chunk_overlap

        logger.info("Ingesting %s …", file_path.name)
        result = self._post("/ingest", payload)
        logger.info(
            "Ingested %s → %d chunks in %.2fs",
            result["doc_id"], result["num_chunks"], result["embedding_time"]
        )
        return result

    def ingest_server_file(
        self,
        filename: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a file already present in server/data/documents/.
        (Useful when documents are uploaded directly to server via SCP/SFTP.)
        """
        payload: dict[str, Any] = {"filename": filename}
        if chunk_size is not None:
            payload["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            payload["chunk_overlap"] = chunk_overlap
        return self._post("/ingest", payload)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """Retrieve chunks without generating an answer."""
        payload: dict[str, Any] = {"query": query, "rerank": rerank}
        if top_k is not None:
            payload["top_k"] = top_k
        if mode is not None:
            payload["mode"] = mode
        if filters:
            payload["filters"] = filters
        return self._post("/retrieve", payload)

    # ── RAG query ─────────────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict[str, Any] | None = None,
        rerank: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Full RAG query: retrieve + generate."""
        payload: dict[str, Any] = {"query": query, "rerank": rerank}
        if top_k is not None:
            payload["top_k"] = top_k
        if mode is not None:
            payload["mode"] = mode
        if filters:
            payload["filters"] = filters
        if system_prompt:
            payload["system_prompt"] = system_prompt
        return self._post("/query", payload)

    # ── Index management ──────────────────────────────────────────────────────

    def index_info(self) -> dict[str, Any]:
        """Get FAISS index statistics."""
        return self._get("/index/info")

    def clear_index(self) -> dict[str, Any]:
        """Delete the entire index (careful!)."""
        return self._delete("/index")

    # ── Config ────────────────────────────────────────────────────────────────

    def update_config(self, **kwargs) -> dict[str, Any]:
        """
        Update server config at runtime.

        Example:
            client.update_config(top_k=10, retrieval_mode="hybrid")
        """
        return self._post("/config", kwargs)

    # ── LLM judge (for evaluation) ────────────────────────────────────────────

    def judge(self, prompt: str) -> str:
        """
        Send a raw prompt to the server's LLM judge.
        Returns the model's text response.
        """
        result = self._post("/_judge", {"prompt": prompt})
        return result.get("response", "")

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False
