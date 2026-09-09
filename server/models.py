"""
server/models.py
────────────────
Pydantic v2 schemas for all API request / response bodies.

These models are shared between the FastAPI layer and the client SDK.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class RetrievalMode(str, Enum):
    dense  = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


# ── Chunk / Document ────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata attached to every stored chunk."""
    doc_id:     str             = Field(..., description="Unique document identifier (filename)")
    chunk_idx:  int             = Field(..., description="Index of this chunk within the document")
    page:       int | None      = Field(None, description="PDF page number (1-indexed)")
    section:    str | None      = Field(None, description="Section/heading if available")
    char_start: int | None      = Field(None, description="Character offset in original text")
    char_end:   int | None      = Field(None, description="Character offset end in original text")
    extra:      dict[str, Any]  = Field(default_factory=dict, description="Technique-specific metadata")


class Chunk(BaseModel):
    """A single retrieved chunk with its content and metadata."""
    chunk_id:   str
    text:       str
    score:      float           = Field(..., description="Similarity / relevance score")
    metadata:   ChunkMetadata


# ── Ingest ───────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Body for /ingest when sending document content as base64 (optional path mode)."""
    filename:       str
    content_b64:    str | None  = Field(None, description="Base64-encoded file bytes; omit if file already on server")
    chunk_size:     int | None  = Field(None, description="Override default chunk_size")
    chunk_overlap:  int | None  = Field(None, description="Override default chunk_overlap")
    chunker:        str         = Field("sliding_window", description="sliding_window | semantic | proposition")


class IngestResponse(BaseModel):
    doc_id:         str
    num_chunks:     int
    embedding_time: float       = Field(..., description="Seconds spent embedding")
    message:        str         = "OK"


# ── Retrieve ──────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:          str
    top_k:          int | None              = None
    mode:           RetrievalMode | None    = None
    filters:        dict[str, Any]          = Field(default_factory=dict)
    rerank:         bool                    = False


class RetrieveResponse(BaseModel):
    query:          str
    chunks:         list[Chunk]
    retrieval_time: float
    num_results:    int


# ── Query (RAG) ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:          str
    top_k:          int | None              = None
    mode:           RetrievalMode | None    = None
    filters:        dict[str, Any]          = Field(default_factory=dict)
    rerank:         bool                    = False
    # Allows injecting a custom system prompt for experiments
    system_prompt:  str | None              = None


class QueryResponse(BaseModel):
    query:          str
    answer:         str
    chunks:         list[Chunk]
    retrieval_time: float
    generation_time: float
    total_time:     float
    num_retrieved:  int
    model:          str


# ── Config (dynamic update) ───────────────────────────────────────────────────

class ConfigUpdateRequest(BaseModel):
    """Partially update runtime settings without restarting the server."""
    chunk_size:     int | None          = None
    chunk_overlap:  int | None          = None
    top_k:          int | None          = None
    retrieval_mode: RetrievalMode | None = None
    hybrid_alpha:   float | None        = None
    reranker_enabled: bool | None       = None
    ollama_model:   str | None          = None
    temperature:    float | None        = None


class ConfigResponse(BaseModel):
    current_config: dict[str, Any]


# ── Index info ────────────────────────────────────────────────────────────────

class IndexInfoResponse(BaseModel):
    num_chunks:     int
    num_documents:  int
    documents:      list[str]
    embedding_model: str
    faiss_index_type: str
    index_size_mb:  float


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """Full configuration for a pluggable RAG pipeline run."""

    # Strategy selectors
    chunker:            str = Field("sliding_window", description="sliding_window | semantic | proposition")
    query_transformer:  str = Field("passthrough",    description="passthrough | multi_query | hyde | step_back")
    retriever:          str = Field("dense",          description="dense | sparse | hybrid")
    reranker:           str = Field("none",           description="none | bge")
    context_processor:  str = Field("passthrough",    description="passthrough | window_expand | compress | rse | contextual_headers")

    # Retrieval params
    top_k:          int | None          = Field(None, description="Override default top_k")
    filters:        dict[str, Any]      = Field(default_factory=dict)
    hybrid_alpha:   float | None        = None

    # Technique-specific params
    multi_query_n:  int                 = Field(3,    description="Number of query variants for multi_query")
    window_size:    int                 = Field(2,    description="Expand window ±k for window_expand")

    # Generation
    system_prompt:  str | None          = None

    # Metadata
    experiment_name: str | None         = None


# ── Preset pipeline configs ───────────────────────────────────────────────────

PIPELINE_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "experiment_name":  "baseline",
        "chunker":          "sliding_window",
        "query_transformer":"passthrough",
        "retriever":        "dense",
        "reranker":         "none",
        "context_processor":"passthrough",
    },
    "hybrid_rerank": {
        "experiment_name":  "hybrid_rerank",
        "chunker":          "sliding_window",
        "query_transformer":"passthrough",
        "retriever":        "hybrid",
        "reranker":         "bge",
        "context_processor":"passthrough",
    },
    "hyde": {
        "experiment_name":  "hyde",
        "query_transformer":"hyde",
        "retriever":        "dense",
        "reranker":         "none",
        "context_processor":"passthrough",
    },
    "multi_query": {
        "experiment_name":  "multi_query",
        "query_transformer":"multi_query",
        "retriever":        "hybrid",
        "reranker":         "bge",
        "context_processor":"passthrough",
        "multi_query_n":    3,
    },
    "step_back": {
        "experiment_name":  "step_back",
        "query_transformer":"step_back",
        "retriever":        "hybrid",
        "reranker":         "none",
        "context_processor":"passthrough",
    },
    "semantic_chunks": {
        "experiment_name":  "semantic_chunks",
        "chunker":          "semantic",
        "query_transformer":"passthrough",
        "retriever":        "dense",
        "reranker":         "bge",
        "context_processor":"passthrough",
    },
    "full_stack": {
        "experiment_name":  "full_stack",
        "query_transformer":"hyde",
        "retriever":        "hybrid",
        "reranker":         "bge",
        "context_processor":"window_expand",
        "window_size":      2,
    },
}


class PipelineQueryRequest(BaseModel):
    """Run a single query through a configurable pipeline."""
    query:  str
    config: PipelineConfig = Field(default_factory=PipelineConfig)


class PipelineQueryResponse(BaseModel):
    query:            str
    answer:           str
    chunks:           list[dict[str, Any]]
    query_variants:   list[str]
    retrieval_time:   float
    generation_time:  float
    total_time:       float
    model:            str | None = None
    pipeline_config:  dict[str, Any]


class CompareRequest(BaseModel):
    """Run same query through multiple pipeline configs side by side."""
    query:   str
    configs: list[PipelineConfig] = Field(
        ..., min_length=1, max_length=4,
        description="Up to 4 pipeline configs to compare",
    )


class CompareResponse(BaseModel):
    query:   str
    results: list[dict[str, Any]]   # one entry per config


# ── Benchmark ─────────────────────────────────────────────────────────────────

class BenchmarkRunRequest(BaseModel):
    """Trigger a full evaluation benchmark run."""
    experiment_name:  str
    config:           PipelineConfig = Field(default_factory=PipelineConfig)
    dataset_path:     str | None     = Field(None, description="Path relative to project root; default: data/evaluation/eval_dataset.json")
    run_judge:        bool           = Field(True,  description="Whether to run LLM-as-judge generation metrics")
    top_k:            int | None     = None


class BenchmarkStatusResponse(BaseModel):
    experiment_name:  str
    status:           str            # running | done | error | idle
    progress:         int            = 0
    total:            int            = 0
    current_metrics:  dict[str, Any] = Field(default_factory=dict)
    result_path:      str | None     = None
    error:            str | None     = None
