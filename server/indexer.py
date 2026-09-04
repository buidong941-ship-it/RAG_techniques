"""
server/indexer.py
─────────────────
Document ingestion pipeline:
  PDF/text → text extraction → chunking → embedding → FAISS index

Design principles:
  - Plain Python, no LangChain/LlamaIndex
  - Each step is a standalone function — easy to swap out for experiments
  - FAISS index is persisted to disk and hot-reloaded on startup
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import fitz                          # PyMuPDF
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from server.config import settings
from server.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


# ── Globals (loaded once at startup) ─────────────────────────────────────────

_embedding_model: BGEM3FlagModel | None = None

# In-memory store: chunk_id → Chunk (FAISS only stores vectors)
_chunk_store:  dict[str, Chunk] = {}
# Ordered list of chunk_ids (index i corresponds to FAISS vector i)
_chunk_ids:    list[str] = []
# FAISS index
_faiss_index:  faiss.Index | None = None


# ── Persistence paths ─────────────────────────────────────────────────────────

def _faiss_path() -> Path:
    return settings.faiss_index_dir / "index.faiss"

def _meta_path() -> Path:
    return settings.faiss_index_dir / "metadata.json"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_embedding_model() -> BGEM3FlagModel:
    """Load BGE-M3; cached in module-level global."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        use_fp16 = settings.embedding_device == "cuda"
        _embedding_model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=use_fp16,
            device=settings.embedding_device,
        )
        logger.info("Embedding model loaded.")
    return _embedding_model


# ── FAISS index helpers ───────────────────────────────────────────────────────

def _build_faiss_index(dim: int) -> faiss.Index:
    """Create a flat inner-product index (cosine sim when vectors are normalized)."""
    index = faiss.IndexFlatIP(dim)
    return index

def _get_or_create_index() -> faiss.Index:
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = _build_faiss_index(settings.embedding_dim)
    return _faiss_index


def save_index() -> None:
    """Persist FAISS index + metadata to disk."""
    settings.ensure_dirs()
    idx = _get_or_create_index()
    faiss.write_index(idx, str(_faiss_path()))
    meta = {
        "chunk_ids": _chunk_ids,
        "chunks": {k: v.model_dump() for k, v in _chunk_store.items()},
    }
    with open(_meta_path(), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Index saved: %d vectors", idx.ntotal)


def load_index() -> None:
    """Load persisted FAISS index + metadata from disk (called at startup)."""
    global _faiss_index, _chunk_ids, _chunk_store
    if _faiss_path().exists() and _meta_path().exists():
        logger.info("Loading existing FAISS index from disk…")
        _faiss_index = faiss.read_index(str(_faiss_path()))
        with open(_meta_path(), encoding="utf-8") as f:
            meta = json.load(f)
        _chunk_ids = meta["chunk_ids"]
        _chunk_store = {k: Chunk(**v) for k, v in meta["chunks"].items()}
        logger.info("Index loaded: %d vectors, %d chunks", _faiss_index.ntotal, len(_chunk_store))
    else:
        logger.info("No existing index found — starting fresh.")
        _faiss_index = _build_faiss_index(settings.embedding_dim)


def clear_index() -> None:
    """Wipe the in-memory index and persisted files."""
    global _faiss_index, _chunk_ids, _chunk_store
    _faiss_index = _build_faiss_index(settings.embedding_dim)
    _chunk_ids = []
    _chunk_store = {}
    for p in [_faiss_path(), _meta_path()]:
        p.unlink(missing_ok=True)
    logger.info("Index cleared.")


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract text page by page using PyMuPDF.

    Returns a list of dicts: {"page": int, "text": str}
    """
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": page_num, "text": text})
    return pages


def extract_text_from_txt(txt_path: Path) -> list[dict[str, Any]]:
    """Extract text from a plain text file (treated as single 'page')."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    return [{"page": 1, "text": text}]


def extract_text(file_path: Path) -> list[dict[str, Any]]:
    """Dispatcher: extract text based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in {".txt", ".md"}:
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """
    Sliding-window character-level chunking.

    Returns list of dicts: {"text": str, "char_start": int, "char_end": int}

    NOTE: This is the BASELINE chunker. Experiments will swap in:
      - Semantic chunking (split on embedding similarity breakpoints)
      - Proposition chunking (LLM-based)
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "char_start": start,
                "char_end": end,
            })
        if end >= text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using BGE-M3.

    Returns float32 array of shape (N, embedding_dim), L2-normalized.
    """
    model = load_embedding_model()
    result = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = np.array(result["dense_vecs"], dtype=np.float32)
    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(vectors)
    return vectors


# ── Full ingestion pipeline ───────────────────────────────────────────────────

def ingest_document(
    file_path: Path,
    doc_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """
    Full pipeline: extract → chunk → embed → add to FAISS.

    Returns stats dict.
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    doc_id        = doc_id        or file_path.name

    logger.info("Ingesting document: %s (chunk_size=%d, overlap=%d)", doc_id, chunk_size, chunk_overlap)

    # 1. Extract text
    pages = extract_text(file_path)

    # 2. Chunk
    all_raw_chunks: list[dict[str, Any]] = []
    for page_info in pages:
        raw_chunks = chunk_text(page_info["text"], chunk_size, chunk_overlap)
        for rc in raw_chunks:
            rc["page"] = page_info["page"]
        all_raw_chunks.extend(raw_chunks)

    if not all_raw_chunks:
        raise ValueError(f"No text extracted from {file_path}")

    texts = [c["text"] for c in all_raw_chunks]

    # 3. Embed
    t0 = time.perf_counter()
    vectors = embed_texts(texts)
    embedding_time = time.perf_counter() - t0

    # 4. Add to FAISS & chunk store
    index = _get_or_create_index()
    new_chunks: list[Chunk] = []

    for i, (raw, vec) in enumerate(zip(all_raw_chunks, vectors)):
        chunk_id = f"{doc_id}::chunk_{len(_chunk_ids) + i}"
        chunk = Chunk(
            chunk_id=chunk_id,
            text=raw["text"],
            score=0.0,  # placeholder, set during retrieval
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_idx=len(_chunk_ids) + i,
                page=raw.get("page"),
                char_start=raw.get("char_start"),
                char_end=raw.get("char_end"),
            ),
        )
        new_chunks.append(chunk)
        _chunk_store[chunk_id] = chunk
        _chunk_ids.append(chunk_id)

    index.add(vectors)
    save_index()

    logger.info(
        "Ingestion complete: %d chunks, %.2fs embedding time",
        len(new_chunks), embedding_time,
    )
    return {
        "doc_id": doc_id,
        "num_chunks": len(new_chunks),
        "embedding_time": embedding_time,
    }


# ── Public accessors (used by retriever) ─────────────────────────────────────

def get_index() -> faiss.Index:
    return _get_or_create_index()

def get_chunk_store() -> dict[str, Chunk]:
    return _chunk_store

def get_chunk_ids() -> list[str]:
    return _chunk_ids

def get_index_info() -> dict[str, Any]:
    idx = _get_or_create_index()
    docs = list({c.metadata.doc_id for c in _chunk_store.values()})
    index_size = 0.0
    if _faiss_path().exists():
        index_size = _faiss_path().stat().st_size / (1024 * 1024)
    return {
        "num_chunks": idx.ntotal,
        "num_documents": len(docs),
        "documents": docs,
        "embedding_model": settings.embedding_model,
        "faiss_index_type": type(idx).__name__,
        "index_size_mb": round(index_size, 2),
    }
