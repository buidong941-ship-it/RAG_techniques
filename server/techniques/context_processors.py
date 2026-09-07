"""
server/techniques/context_processors.py
─────────────────────────────────────────
Context post-processing strategies — applied after retrieval & reranking,
before feeding chunks to the LLM generator.

Strategies:
  passthrough         — baseline, chunks unchanged
  window_expand       — expand each chunk by ±k sentences from neighbours
  compress            — LLM summarises each chunk to keep only relevant parts
  rse                 — Relevant Segment Extraction: merge consecutive relevant chunks
  contextual_headers  — prepend document/section heading to each chunk
"""

from __future__ import annotations

import logging
from typing import Any

from server.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_COMPRESS_PROMPT = """\
Bạn được cung cấp một đoạn văn bản và một câu hỏi. \
Hãy rút gọn đoạn văn bản, chỉ giữ lại những thông tin TRỰC TIẾP liên quan đến câu hỏi. \
Nếu đoạn văn không có thông tin liên quan, trả về chuỗi rỗng.
Không thêm thông tin mới. Giữ nguyên ngôn ngữ gốc.

Câu hỏi: {question}

Đoạn văn:
{text}

Phần liên quan:"""


# ── Passthrough ───────────────────────────────────────────────────────────────

def passthrough_process(
    chunks: list[Chunk],
    query: str,
    **_: Any,
) -> list[Chunk]:
    """Baseline — return chunks unchanged."""
    return chunks


# ── Window Expand ─────────────────────────────────────────────────────────────

def window_expand_process(
    chunks: list[Chunk],
    query: str,
    window_size: int = 2,
    **_: Any,
) -> list[Chunk]:
    """
    Expand each retrieved chunk by adding ±window_size neighbouring chunks
    from the same document.

    This provides more context around each relevant passage without changing
    which chunks were retrieved.
    """
    from server.indexer import get_chunk_store, get_chunk_ids

    chunk_store = get_chunk_store()
    all_ids = get_chunk_ids()

    # Build a quick lookup: (doc_id, chunk_idx) → chunk_id
    pos_map: dict[tuple[str, int], str] = {}
    for cid in all_ids:
        c = chunk_store.get(cid)
        if c:
            pos_map[(c.metadata.doc_id, c.metadata.chunk_idx)] = cid

    expanded: list[Chunk] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        doc_id = chunk.metadata.doc_id
        idx = chunk.metadata.chunk_idx

        # Collect neighbouring chunk ids
        neighbour_ids: list[str] = []
        for offset in range(-window_size, window_size + 1):
            nid = pos_map.get((doc_id, idx + offset))
            if nid and nid not in seen_ids:
                neighbour_ids.append(nid)
                seen_ids.add(nid)

        if not neighbour_ids:
            if chunk.chunk_id not in seen_ids:
                expanded.append(chunk)
                seen_ids.add(chunk.chunk_id)
            continue

        # Merge neighbour texts in order
        neighbour_chunks = [
            chunk_store[nid] for nid in neighbour_ids if nid in chunk_store
        ]
        neighbour_chunks.sort(key=lambda c: c.metadata.chunk_idx)
        merged_text = " ".join(c.text for c in neighbour_chunks)

        # Create a new chunk with merged text, keeping original score
        merged = Chunk(
            chunk_id=chunk.chunk_id,
            text=merged_text,
            score=chunk.score,
            metadata=chunk.metadata.model_copy(
                update={"extra": {**chunk.metadata.extra, "window_expanded": True}}
            ),
        )
        expanded.append(merged)

    return expanded


# ── Contextual Compression ────────────────────────────────────────────────────

def compress_process(
    chunks: list[Chunk],
    query: str,
    min_length: int = 20,
    **_: Any,
) -> list[Chunk]:
    """
    LLM-based contextual compression: keep only the query-relevant parts of each chunk.

    Chunks where the LLM finds nothing relevant are dropped.
    NOTE: One LLM call per chunk — use sparingly on small top-k.
    """
    from server.generator import judge as llm_call

    compressed: list[Chunk] = []

    for chunk in chunks:
        prompt = _COMPRESS_PROMPT.format(question=query, text=chunk.text)
        try:
            result = llm_call(prompt).strip()
            if result and len(result) >= min_length:
                new_chunk = chunk.model_copy(
                    update={
                        "text": result,
                        "metadata": chunk.metadata.model_copy(
                            update={"extra": {**chunk.metadata.extra, "compressed": True}}
                        ),
                    }
                )
                compressed.append(new_chunk)
            else:
                logger.debug("Compression dropped chunk %s (no relevant content)", chunk.chunk_id)
        except Exception as e:
            logger.warning("Compression failed for chunk %s: %s — keeping original", chunk.chunk_id, e)
            compressed.append(chunk)

    if not compressed:
        logger.warning("All chunks compressed away — returning originals")
        return chunks

    return compressed


# ── Relevant Segment Extraction (RSE) ────────────────────────────────────────

def rse_process(
    chunks: list[Chunk],
    query: str,
    max_segment_size: int = 3,
    **_: Any,
) -> list[Chunk]:
    """
    Relevant Segment Extraction: merge consecutive chunks from the same document
    into coherent segments.

    Groups adjacent chunks (by doc_id + chunk_idx) and merges them, preserving
    score of the highest-scoring chunk in each group.
    """
    if not chunks:
        return chunks

    # Sort by (doc_id, chunk_idx)
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (c.metadata.doc_id, c.metadata.chunk_idx),
    )

    segments: list[Chunk] = []
    current_group: list[Chunk] = [sorted_chunks[0]]

    for chunk in sorted_chunks[1:]:
        prev = current_group[-1]
        # Check if this chunk is consecutive to the previous one
        same_doc = chunk.metadata.doc_id == prev.metadata.doc_id
        consecutive = chunk.metadata.chunk_idx == prev.metadata.chunk_idx + 1
        within_limit = len(current_group) < max_segment_size

        if same_doc and consecutive and within_limit:
            current_group.append(chunk)
        else:
            segments.append(_merge_group(current_group))
            current_group = [chunk]

    segments.append(_merge_group(current_group))

    # Re-sort by original relevance score descending
    segments.sort(key=lambda c: c.score, reverse=True)
    return segments


def _merge_group(group: list[Chunk]) -> Chunk:
    """Merge a group of consecutive chunks into one."""
    if len(group) == 1:
        return group[0]
    merged_text = " ".join(c.text for c in group)
    best_score = max(c.score for c in group)
    anchor = group[0]
    return Chunk(
        chunk_id=anchor.chunk_id,
        text=merged_text,
        score=best_score,
        metadata=anchor.metadata.model_copy(
            update={"extra": {**anchor.metadata.extra, "rse_merged": len(group)}}
        ),
    )


# ── Contextual Headers ────────────────────────────────────────────────────────

def contextual_headers_process(
    chunks: list[Chunk],
    query: str,
    **_: Any,
) -> list[Chunk]:
    """
    Prepend the document name and section heading (if available) to each chunk.

    This gives the LLM explicit context about where each passage comes from,
    improving answer attribution and reducing hallucination.
    """
    enriched: list[Chunk] = []
    for chunk in chunks:
        meta = chunk.metadata
        header_parts = [f"[Document: {meta.doc_id}"]
        if meta.section:
            header_parts.append(f"Section: {meta.section}")
        if meta.page:
            header_parts.append(f"Page: {meta.page}")
        header = " | ".join(header_parts) + "]"
        enriched_text = f"{header}\n{chunk.text}"

        enriched.append(
            chunk.model_copy(
                update={
                    "text": enriched_text,
                    "metadata": meta.model_copy(
                        update={"extra": {**meta.extra, "has_header": True}}
                    ),
                }
            )
        )
    return enriched


# ── Dispatcher ────────────────────────────────────────────────────────────────

def process_context(
    chunks: list[Chunk],
    query: str,
    strategy: str = "passthrough",
    **kwargs: Any,
) -> list[Chunk]:
    """
    Apply the chosen context processing strategy.

    Args:
        chunks:   Retrieved (and optionally reranked) chunks.
        query:    Original user query.
        strategy: One of passthrough | window_expand | compress | rse | contextual_headers.
        **kwargs: Strategy-specific parameters.
    Returns:
        Processed chunk list ready for the generator.
    """
    strategy = strategy.lower()
    if strategy == "passthrough":
        return passthrough_process(chunks, query, **kwargs)
    elif strategy == "window_expand":
        return window_expand_process(chunks, query, **kwargs)
    elif strategy == "compress":
        return compress_process(chunks, query, **kwargs)
    elif strategy == "rse":
        return rse_process(chunks, query, **kwargs)
    elif strategy == "contextual_headers":
        return contextual_headers_process(chunks, query, **kwargs)
    else:
        raise ValueError(f"Unknown context processor: {strategy!r}")
