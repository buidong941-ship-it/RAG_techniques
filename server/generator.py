"""
server/generator.py
───────────────────
LLM generation via Ollama REST API.

Responsibilities:
  - Build the RAG prompt (system + context + question)
  - Call Ollama /api/chat (streaming disabled for simplicity)
  - Return answer text + token usage info

Design: plain httpx calls, no LangChain. This keeps the generation logic
transparent and easy to modify for experiments (e.g. changing prompts,
adding chain-of-thought, compression, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from server.config import settings
from server.models import Chunk

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
Bạn là một trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.
Hãy trả lời câu hỏi một cách chính xác và ngắn gọn, CHỈ dựa vào thông tin trong phần [TÀI LIỆU THAM KHẢO].
Nếu thông tin không có trong tài liệu, hãy nói rõ là bạn không tìm thấy thông tin đó.
Trả lời bằng cùng ngôn ngữ với câu hỏi."""

RAG_USER_TEMPLATE = """\
[TÀI LIỆU THAM KHẢO]
{context}

[CÂU HỎI]
{question}

[TRẢ LỜI]"""


def build_context(chunks: list[Chunk]) -> str:
    """
    Concatenate retrieved chunks into a context string.

    Each chunk is prefixed with its source info for traceability.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        source_info = f"[Nguồn {i}: {meta.doc_id}"
        if meta.page:
            source_info += f", trang {meta.page}"
        source_info += "]"
        parts.append(f"{source_info}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(
    query: str,
    chunks: list[Chunk],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the Ollama chat messages list."""
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    context = build_context(chunks)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=query)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]


# ── Ollama call ────────────────────────────────────────────────────────────────

def generate(
    query: str,
    chunks: list[Chunk],
    system_prompt: str | None = None,
    model: str | None = None,
    judge_mode: bool = False,
) -> dict[str, Any]:
    """
    Call Ollama and return the generated answer.

    Args:
        query:         The user question.
        chunks:        Retrieved context chunks.
        system_prompt: Optional override for the system message.
        model:         Override the Ollama model (default: settings.ollama_model).
        judge_mode:    If True, uses ollama_judge_model instead.

    Returns dict with keys: answer, model, prompt_tokens, completion_tokens, total_time
    """
    model = model or (
        settings.ollama_judge_model if judge_mode else settings.ollama_model
    )
    messages = build_messages(query, chunks, system_prompt)

    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature":  settings.temperature,
            "num_predict":  settings.max_new_tokens,
        },
    }

    url = f"{settings.ollama_base_url}/api/chat"
    t0 = time.perf_counter()

    try:
        with httpx.Client(timeout=settings.ollama_timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama timed out after {settings.ollama_timeout}s")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {e.response.text}")

    elapsed = time.perf_counter() - t0
    data = response.json()

    answer = data.get("message", {}).get("content", "").strip()
    usage  = data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    return {
        "answer":            answer,
        "model":             model,
        "prompt_tokens":     usage[0],
        "completion_tokens": usage[1],
        "total_time":        elapsed,
    }


# ── LLM-as-judge helper ───────────────────────────────────────────────────────

def judge(prompt: str, model: str | None = None) -> str:
    """
    Call Ollama with a raw prompt (for evaluation / judge tasks).
    Returns the model's response text.
    """
    model = model or settings.ollama_judge_model
    payload = {
        "model":   model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    url = f"{settings.ollama_base_url}/api/chat"
    with httpx.Client(timeout=settings.ollama_timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()
