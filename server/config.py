"""
server/config.py
────────────────
Centralised configuration for the RAG backend.

All values can be overridden via environment variables using the RAG_ prefix,
or by placing them in a .env file at the project root.

Example:
    RAG_OLLAMA_MODEL=qwen2.5:14b
    RAG_TOP_K=8
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Paths (server-side, relative to project root) ───────────────────────
    data_dir: Path = Path("server/data")
    documents_dir: Path = Path("server/data/documents")
    faiss_index_dir: Path = Path("server/data/faiss_index")

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"        # "cpu" if no GPU
    embedding_batch_size: int = 32
    embedding_dim: int = 1024             # BGE-M3 output dim

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_enabled: bool = False        # toggled per experiment

    # ── Chunking defaults ────────────────────────────────────────────────────
    chunk_size: int = 500                 # characters
    chunk_overlap: int = 50

    # ── Retrieval defaults ────────────────────────────────────────────────────
    top_k: int = 5
    retrieval_mode: str = "dense"         # "dense" | "sparse" | "hybrid"
    hybrid_alpha: float = 0.5             # weight for dense in hybrid (RRF)

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_judge_model: str = "qwen2.5:7b"
    ollama_timeout: int = 120             # seconds

    # ── Generation ───────────────────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.1              # low for factual RAG answers

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.data_dir, self.documents_dir, self.faiss_index_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton used throughout the server
settings = Settings()
