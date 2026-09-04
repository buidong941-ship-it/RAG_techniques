"""
src/evaluation/generation_metrics.py
─────────────────────────────────────
Generation quality metrics using LLM-as-judge (Ollama).

Metrics:
  - answer_correctness  How correct is the answer vs ground truth?
  - faithfulness        Is every claim in the answer supported by the context?
  - context_relevance   How relevant is the retrieved context to the question?
  - answer_relevance    Does the answer actually address the question?

Each metric is scored 0–10 by the LLM judge, then normalised to [0, 1].

Also includes:
  - failure_mode_diagnosis  Is a wrong answer due to retrieval or generation?

NOTE: These are SLOW (one LLM call per metric per sample).
      Run in batch and cache results to disk.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ── Judge prompt templates ─────────────────────────────────────────────────────

_SCORE_INSTRUCTION = (
    "Chỉ trả lời bằng một số nguyên từ 0 đến 10. Không giải thích thêm."
)

ANSWER_CORRECTNESS_PROMPT = """\
Bạn là một giám khảo đánh giá câu trả lời của AI.

Câu hỏi: {question}
Đáp án chuẩn (ground truth): {ground_truth}
Câu trả lời của AI: {answer}

Đánh giá mức độ chính xác của câu trả lời AI so với đáp án chuẩn.
0 = hoàn toàn sai, 10 = hoàn toàn đúng và đầy đủ.
{score_instruction}"""

FAITHFULNESS_PROMPT = """\
Bạn là một giám khảo đánh giá tính trung thực của câu trả lời AI.

Ngữ cảnh được cung cấp:
{context}

Câu trả lời của AI: {answer}

Đánh giá xem câu trả lời có được hỗ trợ bởi ngữ cảnh không.
0 = câu trả lời bịa đặt thông tin không có trong ngữ cảnh,
10 = mọi thông tin trong câu trả lời đều có nguồn gốc từ ngữ cảnh.
{score_instruction}"""

CONTEXT_RELEVANCE_PROMPT = """\
Bạn là một giám khảo đánh giá chất lượng retrieval.

Câu hỏi: {question}
Ngữ cảnh được truy xuất:
{context}

Đánh giá mức độ liên quan của ngữ cảnh so với câu hỏi.
0 = ngữ cảnh hoàn toàn không liên quan, 10 = ngữ cảnh rất liên quan và chứa đủ thông tin.
{score_instruction}"""

ANSWER_RELEVANCE_PROMPT = """\
Bạn là một giám khảo đánh giá mức độ trả lời câu hỏi.

Câu hỏi: {question}
Câu trả lời của AI: {answer}

Đánh giá xem câu trả lời có trực tiếp trả lời câu hỏi không.
0 = câu trả lời lạc đề hoàn toàn, 10 = câu trả lời đầy đủ và đúng trọng tâm.
{score_instruction}"""


# ── Score extraction ───────────────────────────────────────────────────────────

def _extract_score(text: str) -> float | None:
    """Extract the first integer 0–10 from LLM judge response."""
    match = re.search(r"\b(10|[0-9])\b", text.strip())
    if match:
        return int(match.group()) / 10.0  # normalise to [0, 1]
    return None


# ── Individual metrics ────────────────────────────────────────────────────────

def _call_judge(prompt: str, judge_fn) -> float:
    """Call the judge function and parse the score. Returns -1.0 on failure."""
    try:
        response = judge_fn(prompt)
        score = _extract_score(response)
        if score is None:
            logger.warning("Judge returned unparseable response: %r", response[:100])
            return -1.0
        return score
    except Exception as e:
        logger.error("Judge call failed: %s", e)
        return -1.0


def answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    judge_fn,
) -> float:
    prompt = ANSWER_CORRECTNESS_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        score_instruction=_SCORE_INSTRUCTION,
    )
    return _call_judge(prompt, judge_fn)


def faithfulness(
    answer: str,
    context: str,
    judge_fn,
) -> float:
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        answer=answer,
        score_instruction=_SCORE_INSTRUCTION,
    )
    return _call_judge(prompt, judge_fn)


def context_relevance(
    question: str,
    context: str,
    judge_fn,
) -> float:
    prompt = CONTEXT_RELEVANCE_PROMPT.format(
        question=question,
        context=context,
        score_instruction=_SCORE_INSTRUCTION,
    )
    return _call_judge(prompt, judge_fn)


def answer_relevance(
    question: str,
    answer: str,
    judge_fn,
) -> float:
    prompt = ANSWER_RELEVANCE_PROMPT.format(
        question=question,
        answer=answer,
        score_instruction=_SCORE_INSTRUCTION,
    )
    return _call_judge(prompt, judge_fn)


# ── Failure mode diagnosis ────────────────────────────────────────────────────

def diagnose_failure(
    retrieval_recall: float,
    generation_correctness: float,
    recall_threshold: float = 0.5,
    correctness_threshold: float = 0.5,
) -> str:
    """
    Categorise what went wrong for a given sample.

    Returns one of:
      "retrieval_failure"   — Retriever didn't find the right context
      "generation_failure"  — Retriever found context but LLM still got it wrong
      "ok"                  — Both retrieval and generation are adequate
      "both_failure"        — Both failed
    """
    retrieval_ok    = retrieval_recall    >= recall_threshold
    generation_ok   = generation_correctness >= correctness_threshold

    if not retrieval_ok and not generation_ok:
        return "both_failure"
    elif not retrieval_ok:
        return "retrieval_failure"
    elif not generation_ok:
        return "generation_failure"
    else:
        return "ok"


# ── Batch computation ─────────────────────────────────────────────────────────

def compute_generation_metrics(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
    judge_fn,
) -> dict[str, float]:
    """
    Compute all generation metrics for a single sample.

    judge_fn: callable(prompt: str) -> str  (e.g. generator.judge)
    """
    return {
        "answer_correctness": answer_correctness(question, answer, ground_truth, judge_fn),
        "faithfulness":       faithfulness(answer, context, judge_fn),
        "context_relevance":  context_relevance(question, context, judge_fn),
        "answer_relevance":   answer_relevance(question, answer, judge_fn),
    }


def aggregate_generation_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Average generation metrics across the eval set, excluding -1.0 (failed) scores."""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    result = {}
    for key in keys:
        valid = [m[key] for m in per_query_metrics if m[key] >= 0]
        result[key] = sum(valid) / len(valid) if valid else -1.0
        result[f"{key}_coverage"] = len(valid) / len(per_query_metrics)
    return result
