"""
server/techniques/query_transformers.py
────────────────────────────────────────
Query transformation strategies — applied before retrieval.

Strategies:
  passthrough   — baseline, returns query unchanged
  multi_query   — LLM generates N query variants, retrieves for each, merges via RRF
  hyde          — LLM generates a hypothetical answer document, embeds it for search
  step_back     — LLM rephrases to a more general/abstract question
"""

from __future__ import annotations

import logging
from typing import Any

from server.config import settings

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_MULTI_QUERY_PROMPT = """\
Bạn là trợ lý tìm kiếm thông tin. Hãy tạo ra {n} cách diễn đạt khác nhau cho câu hỏi sau.
Mỗi cách diễn đạt nên tiếp cận từ góc độ khác nhau để tìm được nhiều thông tin liên quan hơn.
Chỉ trả về các câu hỏi, mỗi câu trên một dòng, bắt đầu bằng số thứ tự "1. ", "2. ", v.v.
Không giải thích thêm.

Câu hỏi gốc: {question}

Các cách diễn đạt khác:"""

_HYDE_PROMPT = """\
Hãy viết một đoạn văn ngắn (3-5 câu) như thể bạn đang trả lời câu hỏi sau.
Đây là tài liệu giả định dùng để tìm kiếm thông tin liên quan. Viết đầy đủ, cụ thể.
Không nói "Tôi không biết" hay "Không có thông tin".

Câu hỏi: {question}

Đoạn văn giả định:"""

_STEP_BACK_PROMPT = """\
Câu hỏi sau đây có thể quá cụ thể. Hãy tạo ra một câu hỏi tổng quát hơn, \
ở mức độ cao hơn (step-back question) mà khi trả lời sẽ giúp trả lời câu hỏi gốc.
Chỉ trả về câu hỏi mới, không giải thích.

Câu hỏi gốc: {question}

Câu hỏi tổng quát hơn:"""


# ── Passthrough ───────────────────────────────────────────────────────────────

def passthrough_transform(query: str, **_: Any) -> list[str]:
    """Baseline — return the original query unchanged."""
    return [query]


# ── Multi-Query ───────────────────────────────────────────────────────────────

def multi_query_transform(
    query: str,
    n: int = 3,
    **_: Any,
) -> list[str]:
    """
    Generate N alternative phrasings of the query using LLM.

    The caller (pipeline.py) will retrieve for EACH query and merge via RRF.
    Returns list of queries (including original as first element).
    """
    from server.generator import judge as llm_call

    prompt = _MULTI_QUERY_PROMPT.format(question=query, n=n)
    try:
        response = llm_call(prompt)
        variants: list[str] = []
        for line in response.splitlines():
            line = line.strip()
            # Strip leading "1. ", "2. " etc.
            if line and line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) == 2:
                    variants.append(parts[1].strip())
        if not variants:
            logger.warning("multi_query got empty variants, falling back to original")
            return [query]
        # Always include original query
        queries = [query] + variants[:n]
        logger.info("multi_query generated %d queries", len(queries))
        return queries
    except Exception as e:
        logger.warning("multi_query transform failed: %s — using passthrough", e)
        return [query]


# ── HyDE (Hypothetical Document Embedding) ────────────────────────────────────

def hyde_transform(query: str, **_: Any) -> list[str]:
    """
    Generate a hypothetical answer document and return it as the search query.

    The embedding of the hypothetical doc is typically closer to real relevant
    passages than the raw query embedding.

    Returns a single-element list containing the hypothetical document text
    (which will be embedded instead of the original query for FAISS search).
    """
    from server.generator import judge as llm_call

    prompt = _HYDE_PROMPT.format(question=query)
    try:
        hypothesis = llm_call(prompt).strip()
        if not hypothesis:
            logger.warning("HyDE returned empty — using original query")
            return [query]
        logger.info("HyDE hypothesis: %s…", hypothesis[:80])
        # Return hypothesis as the search text; original query kept for display
        return [hypothesis]
    except Exception as e:
        logger.warning("HyDE transform failed: %s — using passthrough", e)
        return [query]


# ── Step-Back ─────────────────────────────────────────────────────────────────

def step_back_transform(query: str, **_: Any) -> list[str]:
    """
    Rephrase the query to a more general/abstract version (step-back prompting).

    Returns both the step-back query and the original query so retrieval can
    use both and merge the results.
    """
    from server.generator import judge as llm_call

    prompt = _STEP_BACK_PROMPT.format(question=query)
    try:
        step_back = llm_call(prompt).strip()
        if not step_back or step_back == query:
            logger.warning("step_back returned same query — using original only")
            return [query]
        logger.info("step_back: %s → %s", query[:60], step_back[:60])
        # Return step-back first (higher priority), then original
        return [step_back, query]
    except Exception as e:
        logger.warning("step_back transform failed: %s — using passthrough", e)
        return [query]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def transform_query(
    query: str,
    strategy: str = "passthrough",
    **kwargs: Any,
) -> list[str]:
    """
    Apply the chosen query transformation strategy.

    Returns a list of queries to retrieve for. The pipeline will retrieve
    for each query and merge the results via RRF.
    """
    strategy = strategy.lower()
    if strategy == "passthrough":
        return passthrough_transform(query, **kwargs)
    elif strategy == "multi_query":
        return multi_query_transform(query, **kwargs)
    elif strategy == "hyde":
        return hyde_transform(query, **kwargs)
    elif strategy == "step_back":
        return step_back_transform(query, **kwargs)
    else:
        raise ValueError(f"Unknown query transformer: {strategy!r}")
