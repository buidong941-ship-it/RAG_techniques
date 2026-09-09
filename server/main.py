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
    PipelineConfig,
    PipelineQueryRequest,
    CompareRequest,
    BenchmarkRunRequest,
    BenchmarkStatusResponse,
    PIPELINE_PRESETS,
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
            chunker=request.chunker,
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
    chunker: str = "sliding_window",
):
    """Ingest via multipart file upload (convenient for curl / UI).

    chunker options: sliding_window (default) | semantic | proposition
    NOTE: proposition is slow — one LLM call per pre-chunk.
    """
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
            chunker=chunker,
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


# ── Pipeline endpoints ───────────────────────────────────────────────────────────────

@app.get("/pipeline/presets", tags=["pipeline"])
async def list_presets():
    """Return all named pipeline presets."""
    return {"presets": PIPELINE_PRESETS}


@app.post("/pipeline/query", tags=["pipeline"])
async def pipeline_query(request: PipelineQueryRequest):
    """
    Run a query through a fully configurable RAG pipeline.
    Specify query_transformer, retriever, reranker, context_processor, etc.
    """
    from server.pipeline import run_pipeline
    try:
        result = run_pipeline(query=request.query, config=request.config, generate=True)
        return result
    except Exception as e:
        logger.exception("Pipeline query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/compare", tags=["pipeline"])
async def pipeline_compare(request: CompareRequest):
    """
    Run the same query through multiple pipeline configs simultaneously.
    Returns results list in same order as configs.
    """
    import asyncio
    from server.pipeline import run_pipeline

    async def _run_one(config: PipelineConfig) -> dict:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: run_pipeline(request.query, config, generate=True)
            )
        except Exception as e:
            return {"error": str(e), "pipeline_config": config.model_dump()}

    results = await asyncio.gather(*[_run_one(cfg) for cfg in request.configs])
    return {"query": request.query, "results": list(results)}


# ── Benchmark endpoints (SSE streaming) ───────────────────────────────────────────

import asyncio
import json as _json
from fastapi.responses import StreamingResponse

# In-memory benchmark state
_benchmark_state: dict[str, Any] = {
    "status": "idle",
    "experiment_name": "",
    "progress": 0,
    "total": 0,
    "current_metrics": {},
    "result_path": None,
    "error": None,
}


@app.get("/benchmark/status", response_model=BenchmarkStatusResponse, tags=["benchmark"])
async def benchmark_status():
    """Get current benchmark run status."""
    return BenchmarkStatusResponse(**_benchmark_state)


@app.get("/benchmark/list", tags=["benchmark"])
async def benchmark_list():
    """List all experiment result files."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return {"experiments": []}
    experiments = []
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir():
            metrics_file = exp_dir / "metrics.json"
            experiments.append({
                "name": exp_dir.name,
                "has_metrics": metrics_file.exists(),
                "metrics_path": str(metrics_file) if metrics_file.exists() else None,
            })
    return {"experiments": experiments}


@app.get("/benchmark/results/{experiment_name}", tags=["benchmark"])
async def benchmark_results(experiment_name: str):
    """Get metrics.json for a specific experiment."""
    metrics_file = Path("experiments/results") / experiment_name / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail=f"No results for '{experiment_name}'")
    import json as _j
    with open(metrics_file, encoding="utf-8") as f:
        return _j.load(f)


@app.post("/benchmark/run", tags=["benchmark"])
async def benchmark_run(request: BenchmarkRunRequest):
    """
    Start a benchmark run and stream progress via Server-Sent Events.

    Returns SSE stream:
      data: {"type": "progress", "sample": N, "total": T, "metrics": {...}}
      data: {"type": "done", "metrics": {...}, "result_path": "..."}
      data: {"type": "error", "message": "..."}
    """
    if _benchmark_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A benchmark is already running")

    async def _stream():
        from server.pipeline import run_pipeline
        import json as _j

        _benchmark_state.update({
            "status": "running",
            "experiment_name": request.experiment_name,
            "progress": 0, "total": 0,
            "current_metrics": {}, "result_path": None, "error": None,
        })

        try:
            # Load dataset
            dataset_path = Path(request.dataset_path or "data/evaluation/eval_dataset.json")
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            with open(dataset_path, encoding="utf-8") as f:
                dataset = _j.load(f)

            if not dataset:
                raise ValueError("Eval dataset is empty")

            _benchmark_state["total"] = len(dataset)
            top_k = request.top_k or request.config.top_k or settings.top_k
            config = request.config
            if top_k:
                config = config.model_copy(update={"top_k": top_k})

            results_dir = Path("experiments/results") / request.experiment_name
            results_dir.mkdir(parents=True, exist_ok=True)

            per_sample: list[dict] = []
            running_metrics: dict[str, list[float]] = {}

            for i, sample in enumerate(dataset):
                question = sample.get("question", "")
                ground_truth = sample.get("ground_truth", "")

                # Run pipeline
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda q=question: run_pipeline(q, config, generate=True)
                )

                sample_record: dict[str, Any] = {
                    "sample_idx": i,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": result.get("answer", ""),
                    "retrieved_ids": [c["chunk_id"] for c in result.get("chunks", [])],
                    "retrieval_time": result.get("retrieval_time", 0),
                    "generation_time": result.get("generation_time", 0),
                }

                # LLM judge metrics
                if request.run_judge and sample_record["answer"]:
                    from server.evaluation_helpers import judge_sample
                    loop2 = asyncio.get_event_loop()
                    gen_metrics = await loop2.run_in_executor(
                        None,
                        lambda: judge_sample(
                            question=question,
                            answer=sample_record["answer"],
                            ground_truth=ground_truth,
                            context="\n".join(c["text"] for c in result.get("chunks", [])),
                        )
                    )
                    sample_record.update(gen_metrics)
                    for k, v in gen_metrics.items():
                        if isinstance(v, (int, float)):
                            running_metrics.setdefault(k, []).append(v)

                per_sample.append(sample_record)
                _benchmark_state["progress"] = i + 1

                # Compute running averages
                avg_metrics = {k: sum(vs)/len(vs) for k, vs in running_metrics.items()}
                _benchmark_state["current_metrics"] = avg_metrics

                progress_event = _j.dumps({
                    "type": "progress",
                    "sample": i + 1,
                    "total": len(dataset),
                    "question": question[:80],
                    "metrics": avg_metrics,
                })
                yield f"data: {progress_event}\n\n"

            # Save results
            agg = {k: sum(vs)/len(vs) for k, vs in running_metrics.items()}
            output = {
                "experiment": request.experiment_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config.model_dump(),
                "num_samples": len(dataset),
                "generation_metrics": agg,
                "per_sample": per_sample,
            }
            metrics_path = results_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                _j.dump(output, f, ensure_ascii=False, indent=2)

            result_path = str(metrics_path)
            _benchmark_state.update({"status": "done", "result_path": result_path})

            done_event = _j.dumps({"type": "done", "metrics": agg, "result_path": result_path})
            yield f"data: {done_event}\n\n"

        except Exception as e:
            logger.exception("Benchmark failed")
            _benchmark_state.update({"status": "error", "error": str(e)})
            err_event = _j.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_event}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
