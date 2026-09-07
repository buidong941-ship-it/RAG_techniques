"""
server/evaluation_helpers.py
─────────────────────────────
Lightweight LLM-as-judge helpers used by the benchmark SSE endpoint.
Adapted from src/evaluation/generation_metrics.py but callable server-side.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from server.generator import judge as _llm

logger = logging.getLogger(__name__)

# ── Score extraction ──────────────────────────────────────────────────────────

def _extract_score(text: str, low: float = 0.0, high: float = 1.0) -> float:
    """Extract first float/int from LLM response, clamped to [low, high]."""
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if not matches:
        return (low + high) / 2
    val = float(matches[0])
    # If score looks like it's on a 0-10 scale, normalise
    if val > high and val <= 10:
        val = val / 10
    return max(low, min(high, val))


# ── Judge prompts ─────────────────────────────────────────────────────────────

_CORRECTNESS_PROMPT = """\
Đánh giá mức độ chính xác của câu trả lời so với đáp án chuẩn.
Chỉ trả về một số từ 0 đến 1 (0 = hoàn toàn sai, 1 = hoàn toàn đúng).

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}
Câu trả lời cần đánh giá: {answer}

Điểm (0-1):"""

_FAITHFULNESS_PROMPT = """\
Đánh giá mức độ trung thực của câu trả lời — câu trả lời có dựa trên ngữ cảnh được cung cấp không?
Chỉ trả về một số từ 0 đến 1 (0 = bịa đặt hoàn toàn, 1 = hoàn toàn dựa trên ngữ cảnh).

Ngữ cảnh: {context}
Câu trả lời: {answer}

Điểm (0-1):"""

_CONTEXT_RELEVANCE_PROMPT = """\
Đánh giá mức độ liên quan của ngữ cảnh được truy xuất so với câu hỏi.
Chỉ trả về một số từ 0 đến 1 (0 = không liên quan, 1 = rất liên quan).

Câu hỏi: {question}
Ngữ cảnh: {context}

Điểm (0-1):"""

_ANSWER_RELEVANCE_PROMPT = """\
Đánh giá mức độ câu trả lời có trực tiếp trả lời câu hỏi không.
Chỉ trả về một số từ 0 đến 1 (0 = không trả lời câu hỏi, 1 = trả lời trực tiếp và đầy đủ).

Câu hỏi: {question}
Câu trả lời: {answer}

Điểm (0-1):"""


# ── Main judge function ───────────────────────────────────────────────────────

def judge_sample(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
) -> dict[str, float]:
    """
    Run LLM-as-judge on a single QA sample.
    Returns dict of metric_name → score (0-1).
    """
    metrics: dict[str, float] = {}

    def _score(prompt: str) -> float:
        try:
            resp = _llm(prompt)
            return _extract_score(resp)
        except Exception as e:
            logger.warning("Judge call failed: %s", e)
            return 0.5  # neutral fallback

    if ground_truth:
        metrics["answer_correctness"] = _score(
            _CORRECTNESS_PROMPT.format(
                question=question, ground_truth=ground_truth, answer=answer
            )
        )

    if context:
        metrics["faithfulness"] = _score(
            _FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer)
        )
        metrics["context_relevance"] = _score(
            _CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context[:2000])
        )

    metrics["answer_relevance"] = _score(
        _ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    )

    return metrics
