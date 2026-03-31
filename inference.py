"""
model/inference.py
------------------
Core inference layer: ask_question(context, question) → answer + confidence.
Includes caching, multi-context retrieval, and confidence thresholding.
"""

import logging
import functools
import time
from typing import Union

logger = logging.getLogger(__name__)

# ── Lazy-loaded pipeline (loaded once, reused across requests) ───────────────
_pipeline_cache: dict = {}


def get_pipeline(model_path: str = "distilbert-base-cased-distilled-squad"):
    """Return a cached pipeline instance (loads only on first call)."""
    if model_path not in _pipeline_cache:
        from model.qa_model import build_pipeline
        logger.info("Initialising QA pipeline …")
        _pipeline_cache[model_path] = build_pipeline(model_path)
    return _pipeline_cache[model_path]


# ════════════════════════════════════════════════════════════════════════════
# Main inference function
# ════════════════════════════════════════════════════════════════════════════

def ask_question(
    context:    str,
    question:   str,
    model_path: str  = "distilbert-base-cased-distilled-squad",
    top_k:      int  = 3,           # return top-k candidate answers
    min_score:  float = 0.01,       # discard answers below this confidence
    max_answer_len: int = 100,      # maximum answer length in tokens
    handle_impossible: bool = True, # handle questions with no valid answer
) -> dict:
    """
    Ask a question given a context paragraph.

    Parameters
    ----------
    context   : The passage of text to search for an answer.
    question  : The natural-language question.
    model_path: HuggingFace model ID or local checkpoint directory.
    top_k     : Number of candidate answer spans to consider.
    min_score : Minimum confidence score to report an answer.
    max_answer_len : Maximum token length of a valid answer.
    handle_impossible : If True, return a graceful "no answer" response
                        when confidence is below min_score.

    Returns
    -------
    {
        "answer":     str,    # best answer string
        "score":      float,  # confidence  0 → 1
        "start":      int,    # char start offset in context
        "end":        int,    # char end offset
        "candidates": list,   # top_k answers with individual scores
        "latency_ms": float,  # wall-clock inference time
    }
    """
    # ── Input validation ────────────────────────────────────────────────────
    context  = (context  or "").strip()
    question = (question or "").strip()

    if not context:
        return _error_response("Context cannot be empty.")
    if not question:
        return _error_response("Question cannot be empty.")
    if len(context) < 10:
        return _error_response("Context is too short to extract a meaningful answer.")

    # ── Run inference ────────────────────────────────────────────────────────
    pipe = get_pipeline(model_path)
    t0   = time.perf_counter()

    try:
        raw = pipe(
            question=question,
            context=context,
            top_k=top_k,
            max_answer_len=max_answer_len,
            handle_impossible_answer=handle_impossible,
        )
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return _error_response(f"Model inference failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000

    # pipeline returns a list when top_k > 1, a dict when top_k == 1
    candidates = raw if isinstance(raw, list) else [raw]

    # ── Filter & format ──────────────────────────────────────────────────────
    best = candidates[0]

    if best["score"] < min_score or best.get("answer") in ("", None):
        if handle_impossible:
            return {
                "answer":     "I could not find a confident answer in the provided context.",
                "score":      float(best["score"]),
                "start":      None,
                "end":        None,
                "candidates": [],
                "latency_ms": round(latency_ms, 2),
                "error":      None,
            }

    logger.debug("Q: %s | A: %s (%.4f) | %.1f ms",
                 question, best["answer"], best["score"], latency_ms)

    return {
        "answer":     best["answer"],
        "score":      round(float(best["score"]), 6),
        "start":      best.get("start"),
        "end":        best.get("end"),
        "candidates": [
            {"answer": c["answer"], "score": round(float(c["score"]), 6)}
            for c in candidates
        ],
        "latency_ms": round(latency_ms, 2),
        "error":      None,
    }


def ask_question_multi_context(
    contexts:   list[str],
    question:   str,
    model_path: str = "distilbert-base-cased-distilled-squad",
) -> dict:
    """
    Run the QA model over MULTIPLE context paragraphs and return the
    highest-scoring answer across all of them.

    Useful for a simple retriever-reader pipeline:
        1. Retrieve top-N context chunks with TF-IDF / BM25 / embeddings
        2. Call this function to find the best answer among them
    """
    if not contexts:
        return _error_response("No contexts provided.")

    best_result = None
    for i, ctx in enumerate(contexts):
        result = ask_question(ctx, question, model_path=model_path)
        if result.get("error"):
            continue
        if best_result is None or result["score"] > best_result["score"]:
            best_result = {**result, "source_context_index": i}

    return best_result or _error_response("No valid answer found across all contexts.")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _error_response(message: str) -> dict:
    return {
        "answer":     message,
        "score":      0.0,
        "start":      None,
        "end":        None,
        "candidates": [],
        "latency_ms": 0.0,
        "error":      message,
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

    ctx = (
        "Machine learning (ML) is a field of inquiry devoted to understanding "
        "and building methods that 'learn', that is, methods that leverage data "
        "to improve performance on some set of tasks. It is seen as a part of "
        "artificial intelligence."
    )

    tests = [
        "What is machine learning?",
        "Who invented machine learning?",   # unanswerable → low score
        "",                                  # empty question → validation error
    ]

    for q in tests:
        res = ask_question(ctx, q)
        print(f"\nQ: {q!r}")
        print(f"A: {res['answer']}")
        print(f"   confidence={res['score']:.4f}  latency={res['latency_ms']} ms")
        if res["error"]:
            print(f"   [error]: {res['error']}")
