"""
server/main.py
──────────────
FastAPI application — the RAG backend.

Endpoints:
  POST /ingest          Upload & index a document
  POST /query           Full RAG query (retrieve + generate)
  POST /retrieve        Retrieval only (no generation)
  GET  /index/info      Index statistics
  DELETE /index         Clear the entire index
  POST /config          Dynamically update server settings
  GET  /health          Health check
"""

from __future__ import annotations

import base64
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.config import settings
from server.indexer import (
    ingest_document,
    clear_index,
    get_index_info,
    load_index,
)
from server.retriever import retrieve as _retrieve
from server.reranker import rerank
from server.generator import generate
from server.models import (
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    QueryRequest,
    QueryResponse,
    ConfigUpdateRequest,
    ConfigResponse,
    IndexInfoResponse,
    Chunk,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    settings.ensure_dirs()
    load_index()
    logger.info("RAG server started on %s:%d", settings.host, settings.port)
    yield
    # ── Shutdown (add cleanup here if needed) ─────────────────────────────────


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Backend",
    description="Local RAG server with FAISS + BGE-M3 + Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["indexing"])
async def ingest_endpoint(request: IngestRequest):
    """
    Ingest a document.

    Two modes:
    1. content_b64 provided → decode and save to a temp file, then ingest
    2. content_b64 omitted  → assume file is already in server/data/documents/
    """
    settings.ensure_dirs()

    if request.content_b64:
        # Decode and write to temp file
        try:
            file_bytes = base64.b64decode(request.content_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 content")

        tmp_path = settings.documents_dir / request.filename
        tmp_path.write_bytes(file_bytes)
        file_path = tmp_path
    else:
        file_path = settings.documents_dir / request.filename
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{request.filename}' not found in {settings.documents_dir}",
            )

    try:
        stats = ingest_document(
            file_path=file_path,
            doc_id=request.filename,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        doc_id=stats["doc_id"],
        num_chunks=stats["num_chunks"],
        embedding_time=stats["embedding_time"],
    )


@app.post("/ingest/upload", response_model=IngestResponse, tags=["indexing"])
async def ingest_upload(
    file: UploadFile = File(...),
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """Ingest via multipart file upload (convenient for curl / UI)."""
    settings.ensure_dirs()
    file_bytes = await file.read()
    file_path = settings.documents_dir / file.filename
    file_path.write_bytes(file_bytes)

    try:
        stats = ingest_document(
            file_path=file_path,
            doc_id=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as e:
        logger.exception("Upload ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        doc_id=stats["doc_id"],
        num_chunks=stats["num_chunks"],
        embedding_time=stats["embedding_time"],
    )


# ── Retrieve ──────────────────────────────────────────────────────────────────

@app.post("/retrieve", response_model=RetrieveResponse, tags=["retrieval"])
async def retrieve_endpoint(request: RetrieveRequest):
    """Retrieve relevant chunks without generating an answer."""
    chunks, retrieval_time = _retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode.value if request.mode else None,
        filters=request.filters or None,
    )

    if request.rerank and chunks:
        chunks, _ = rerank(request.query, chunks, top_k=request.top_k or settings.top_k)

    return RetrieveResponse(
        query=request.query,
        chunks=chunks,
        retrieval_time=retrieval_time,
        num_results=len(chunks),
    )


# ── Query (RAG) ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_endpoint(request: QueryRequest):
    """Full RAG: retrieve → (optionally rerank) → generate."""
    t_total_start = time.perf_counter()

    # 1. Retrieve
    chunks, retrieval_time = _retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode.value if request.mode else None,
        filters=request.filters or None,
    )

    # 2. Rerank (optional)
    rerank_time = 0.0
    if (request.rerank or settings.reranker_enabled) and chunks:
        chunks, rerank_time = rerank(
            request.query, chunks, top_k=request.top_k or settings.top_k
        )

    if not chunks:
        return QueryResponse(
            query=request.query,
            answer="Không tìm thấy thông tin liên quan trong tài liệu.",
            chunks=[],
            retrieval_time=retrieval_time,
            generation_time=0.0,
            total_time=time.perf_counter() - t_total_start,
            num_retrieved=0,
            model=settings.ollama_model,
        )

    # 3. Generate
    try:
        gen_result = generate(
            query=request.query,
            chunks=chunks,
            system_prompt=request.system_prompt,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    total_time = time.perf_counter() - t_total_start

    return QueryResponse(
        query=request.query,
        answer=gen_result["answer"],
        chunks=chunks,
        retrieval_time=retrieval_time + rerank_time,
        generation_time=gen_result["total_time"],
        total_time=total_time,
        num_retrieved=len(chunks),
        model=gen_result["model"],
    )


# ── Index management ──────────────────────────────────────────────────────────

@app.get("/index/info", response_model=IndexInfoResponse, tags=["indexing"])
async def index_info():
    """Return statistics about the current FAISS index."""
    return IndexInfoResponse(**get_index_info())


@app.delete("/index", tags=["indexing"])
async def delete_index():
    """Clear the entire index (irreversible)."""
    clear_index()
    return {"message": "Index cleared successfully"}


# ── Dynamic config ────────────────────────────────────────────────────────────

@app.post("/config", response_model=ConfigResponse, tags=["system"])
async def update_config(request: ConfigUpdateRequest):
    """
    Update server settings at runtime without restarting.

    Useful for experiment runs: change top_k, chunk_size, retrieval_mode, etc.
    Note: chunk_size / chunk_overlap changes only affect NEW ingestion.
    """
    updates = request.model_dump(exclude_none=True)
    for key, value in updates.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    logger.info("Config updated: %s", updates)
    return ConfigResponse(current_config=settings.model_dump())


# ── LLM Judge (internal, used by evaluator) ───────────────────────────────────

class JudgeRequest(BaseModel):
    prompt: str

@app.post("/_judge", tags=["internal"])
async def judge_endpoint(request: JudgeRequest):
    """Internal endpoint: send a raw prompt to the LLM judge."""
    from server.generator import judge as _judge
    try:
        response = _judge(request.prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"response": response}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
