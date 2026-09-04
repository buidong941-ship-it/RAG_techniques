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
