"""
server/techniques/chunkers.py
──────────────────────────────
Chunking strategies — pluggable via PipelineConfig.chunker.

Strategies:
  sliding_window  — baseline fixed-size chunks with overlap (default)
  semantic        — split at embedding similarity breakpoints
  proposition     — LLM extracts atomic facts (one fact per chunk)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: str = "sliding_window",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> list[str]:
    """
    Dispatch to the appropriate chunking strategy.

    Args:
        text:          Full document text.
        strategy:      One of sliding_window | semantic | proposition.
        chunk_size:    Target chunk size (chars). Used by sliding_window.
        chunk_overlap: Overlap between consecutive chunks (chars).
        **kwargs:      Strategy-specific options.
    Returns:
        List of text chunks (strings).
    """
    if strategy == "sliding_window":
        return sliding_window_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return semantic_chunk(text, **kwargs)
    elif strategy == "proposition":
        return proposition_chunk(text, **kwargs)
    else:
        raise ValueError(f"Unknown chunker strategy: {strategy!r}")


# ── Strategy 1: Sliding Window (baseline) ────────────────────────────────────

def sliding_window_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Split text into fixed-size character windows with overlap.
    Respects sentence boundaries where possible.
    """
    if not text.strip():
        return []

    # Split into sentences for cleaner boundaries
    sentences = _split_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)
        if current_len + slen > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last sentences that fit in chunk_overlap chars
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_buf.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_buf
            current_len = overlap_len
        current.append(sentence)
        current_len += slen

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ── Strategy 2: Semantic Chunking ─────────────────────────────────────────────

def semantic_chunk(
    text: str,
    breakpoint_threshold: float = 0.3,
    min_chunk_size: int = 100,
    **_: Any,
) -> list[str]:
    """
    Split at embedding cosine-similarity breakpoints between sentences.

    Sentences where the similarity to the next sentence drops below
    `breakpoint_threshold` are treated as chunk boundaries.

    Uses the global BGE-M3 embedding model from indexer.
    """
    from server.indexer import embed_texts  # lazy import to avoid circular

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return [text.strip()] if text.strip() else []

    # Embed all sentences at once (batch)
    embeddings = embed_texts(sentences)  # (N, dim), L2-normalised

    # Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    # Find breakpoints: low similarity = topic shift
    threshold = np.mean(similarities) - breakpoint_threshold * np.std(similarities)

    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current)) >= min_chunk_size:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


# ── Strategy 3: Proposition Chunking ─────────────────────────────────────────

_PROPOSITION_PROMPT = """\
Bạn là chuyên gia phân tích văn bản. Hãy tách đoạn văn bản sau thành các mệnh đề nguyên tử (atomic propositions).

Quy tắc:
- Mỗi mệnh đề là một sự kiện, thông tin hoặc khẳng định độc lập, đầy đủ ý nghĩa khi đứng một mình.
- Giữ nguyên tên riêng, số liệu, ngày tháng.
- Mỗi mệnh đề trên một dòng riêng, bắt đầu bằng dấu "- ".
- Không thêm giải thích, không đặt lại tiêu đề.
- Nếu đoạn văn quá ngắn (< 2 câu), giữ nguyên và trả về như một mệnh đề duy nhất.

VĂN BẢN:
{text}

MỆNH ĐỀ:"""


def proposition_chunk(
    text: str,
    pre_chunk_size: int = 800,
    pre_chunk_overlap: int = 80,
    min_proposition_length: int = 20,
    **_: Any,
) -> list[str]:
    """
    LLM-based proposition chunking:
      1. Pre-chunk the text with sliding window (to fit LLM context).
      2. For each pre-chunk, call LLM to extract atomic propositions.
      3. Return flat list of propositions.

    NOTE: Expensive — one LLM call per pre-chunk. Best for small datasets.
    """
    from server.generator import judge as llm_call  # reuse judge helper

    # Step 1: pre-chunk to fit LLM context window
    pre_chunks = sliding_window_chunk(text, pre_chunk_size, pre_chunk_overlap)
    if not pre_chunks:
        return []

    propositions: list[str] = []

    for i, pre_chunk in enumerate(pre_chunks):
        logger.debug("Proposition chunking: pre-chunk %d/%d", i + 1, len(pre_chunks))
        prompt = _PROPOSITION_PROMPT.format(text=pre_chunk)
        try:
            response = llm_call(prompt)
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    prop = line[2:].strip()
                    if len(prop) >= min_proposition_length:
                        propositions.append(prop)
        except Exception as e:
            logger.warning("Proposition chunking failed for pre-chunk %d: %s", i, e)
            # Fallback: keep the pre-chunk as-is
            propositions.append(pre_chunk)

    return propositions if propositions else pre_chunks


# ── Sentence splitter ─────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that handles Vietnamese + English text.
    Splits on '. ', '! ', '? ', '.\n', etc.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Split on sentence-ending punctuation followed by space or EOL
    parts = re.split(r"(?<=[.!?…])\s+", text)
    # Filter out empty strings
    return [p.strip() for p in parts if p.strip()]
