"""
evaluation/metrics.py
---------------------
Computes standard QA evaluation metrics:
  • Exact Match (EM)  — 1 if predicted == gold (after normalisation), else 0
  • F1               — token-level overlap between prediction and gold
  • Full evaluation loop over a dataset
"""

import re
import string
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Text normalisation (mirrors the official SQuAD eval script)
# ════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation & articles, collapse whitespace."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse spaces
    return " ".join(text.split())


# ════════════════════════════════════════════════════════════════════════════
# Per-sample metrics
# ════════════════════════════════════════════════════════════════════════════

def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalised strings match exactly, else 0.0."""
    return float(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1: harmonic mean of precision and recall over word tokens.
    This is the standard SQuAD F1 metric.
    """
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1_and_em(prediction: str, gold_answers: list[str]) -> tuple[float, float]:
    """
    When multiple gold answers exist, take the best F1 / EM over all of them.
    (Standard SQuAD evaluation protocol.)
    """
    f1 = max(f1_score(prediction, g) for g in gold_answers)
    em = max(exact_match(prediction, g) for g in gold_answers)
    return f1, em


# ════════════════════════════════════════════════════════════════════════════
# Dataset-level evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_dataset(
    samples: list[dict],
    qa_fn,                        # callable: (context, question) → result dict
    model_path: str = "distilbert-base-cased-distilled-squad",
    verbose: bool = True,
) -> dict:
    """
    Run the QA pipeline over every sample and compute aggregate metrics.

    Parameters
    ----------
    samples    : list of dicts with keys: context, question, answer_text
    qa_fn      : inference function matching ask_question's signature
    model_path : model to evaluate
    verbose    : print per-sample results

    Returns
    -------
    {"exact_match": float, "f1": float, "num_samples": int,
     "per_sample": list}
    """
    total_em, total_f1 = 0.0, 0.0
    per_sample = []

    for i, sample in enumerate(samples):
        context  = sample["context"]
        question = sample["question"]
        gold     = sample.get("answer_text", sample.get("answers", {}).get("text", [""])[0])

        result = qa_fn(context=context, question=question, model_path=model_path)
        prediction = result.get("answer", "")

        # gold may be a single string or a list (multiple annotations)
        gold_list = gold if isinstance(gold, list) else [gold]
        f1, em    = best_f1_and_em(prediction, gold_list)

        total_em += em
        total_f1 += f1

        row = {
            "id":         sample.get("id", str(i)),
            "question":   question,
            "gold":       gold_list[0],
            "prediction": prediction,
            "score":      result.get("score", 0.0),
            "em":         em,
            "f1":         f1,
        }
        per_sample.append(row)

        if verbose:
            status = "✓" if em == 1.0 else ("~" if f1 > 0.5 else "✗")
            print(f"[{status}] Q: {question[:60]}")
            print(f"     Gold: {gold_list[0]!r}")
            print(f"     Pred: {prediction!r}  (model_score={result.get('score', 0):.4f}, F1={f1:.2f})")

    n = len(samples)
    summary = {
        "exact_match": round(total_em / n * 100, 2) if n else 0.0,
        "f1":          round(total_f1 / n * 100, 2) if n else 0.0,
        "num_samples": n,
        "per_sample":  per_sample,
    }

    print(f"\n{'═'*50}")
    print(f"  Exact Match : {summary['exact_match']:.1f}%")
    print(f"  F1 Score    : {summary['f1']:.1f}%")
    print(f"  Samples     : {n}")
    print(f"{'═'*50}")

    return summary


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../")))

    from data.dataset_loader import load_squad_json
    from model.inference import ask_question

    DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/sample_dataset.json")
    samples   = load_squad_json(DATA_FILE)

    print(f"\nEvaluating {len(samples)} samples …\n")
    metrics = evaluate_dataset(samples, qa_fn=ask_question, verbose=True)
