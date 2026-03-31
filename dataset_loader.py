"""
data/dataset_loader.py
----------------------
Loads and preprocesses QA datasets in SQuAD format.
Supports both local JSON files and HuggingFace datasets.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Load raw JSON (SQuAD format)
# ──────────────────────────────────────────────

def load_squad_json(filepath: str) -> list[dict]:
    """
    Parse a SQuAD-style JSON file into a flat list of
    {id, context, question, answer_text, answer_start} dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue                         # skip unanswerable Qs
                answer = qa["answers"][0]            # take first gold answer
                samples.append({
                    "id":           qa["id"],
                    "context":      context,
                    "question":     qa["question"],
                    "answer_text":  answer["text"],
                    "answer_start": answer["answer_start"],
                })

    logger.info("Loaded %d QA samples from %s", len(samples), filepath)
    return samples


# ──────────────────────────────────────────────
# 2. Convert to HuggingFace Dataset
# ──────────────────────────────────────────────

def samples_to_hf_dataset(samples: list[dict],
                           val_split: float = 0.2) -> DatasetDict:
    """
    Convert flat list of samples to a train/validation DatasetDict
    in the format expected by the Trainer API.
    """
    # HuggingFace expects 'answers' as {"text": [...], "answer_start": [...]}
    hf_samples = [
        {
            "id":       s["id"],
            "context":  s["context"],
            "question": s["question"],
            "answers":  {
                "text":         [s["answer_text"]],
                "answer_start": [s["answer_start"]],
            },
        }
        for s in samples
    ]

    full_ds = Dataset.from_list(hf_samples)

    if len(full_ds) < 4:
        # Too small to split — use everything for both splits
        logger.warning("Dataset too small to split; using all data for train & validation.")
        return DatasetDict({"train": full_ds, "validation": full_ds})

    split = full_ds.train_test_split(test_size=val_split, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ──────────────────────────────────────────────
# 3. Load from HuggingFace Hub (SQuAD)
# ──────────────────────────────────────────────

def load_hf_squad(subset: str = "plain_text",
                  train_samples: Optional[int] = 1000,
                  val_samples: Optional[int] = 200) -> DatasetDict:
    """
    Download the real SQuAD dataset from the HuggingFace Hub.
    Pass None for *_samples to use the full split.
    """
    from datasets import load_dataset
    logger.info("Downloading SQuAD from HuggingFace Hub …")
    ds = load_dataset("squad", subset)

    if train_samples:
        ds["train"] = ds["train"].select(range(min(train_samples, len(ds["train"]))))
    if val_samples:
        ds["validation"] = ds["validation"].select(
            range(min(val_samples, len(ds["validation"])))
        )

    logger.info("SQuAD — train: %d | validation: %d",
                len(ds["train"]), len(ds["validation"]))
    return ds


# ──────────────────────────────────────────────
# 4. Tokenise for extractive QA (DistilBERT / BERT)
# ──────────────────────────────────────────────

def tokenize_dataset(dataset: DatasetDict,
                     tokenizer,
                     max_length: int = 384,
                     doc_stride: int = 128) -> DatasetDict:
    """
    Tokenise contexts + questions and compute start/end token positions
    for the gold-answer spans.  Works with sliding-window (doc_stride)
    for long contexts.
    """

    def preprocess(examples):
        # Tokenise with overflow / stride so long contexts are handled
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map   = tokenized.pop("overflow_to_sample_mapping")
        offset_map   = tokenized.pop("offset_mapping")

        start_positions, end_positions = [], []

        for i, offsets in enumerate(offset_map):
            sample_idx  = sample_map[i]
            answers     = examples["answers"][sample_idx]
            cls_index   = tokenized["input_ids"][i].index(tokenizer.cls_token_id)

            # If no answers, point to CLS token (marks "no answer")
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            ans_start_char = answers["answer_start"][0]
            ans_end_char   = ans_start_char + len(answers["text"][0])

            # Find which tokens fall inside the answer span
            seq_ids        = tokenized.sequence_ids(i)
            ctx_start      = seq_ids.index(1)          # first context token
            ctx_end        = len(seq_ids) - 1
            while seq_ids[ctx_end] != 1:
                ctx_end -= 1

            # Answer outside this window → point to CLS
            if (offsets[ctx_start][0] > ans_end_char or
                    offsets[ctx_end][1] < ans_start_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                tok_start = ctx_start
                while tok_start <= ctx_end and offsets[tok_start][0] <= ans_start_char:
                    tok_start += 1
                start_positions.append(tok_start - 1)

                tok_end = ctx_end
                while tok_end >= ctx_start and offsets[tok_end][1] >= ans_end_char:
                    tok_end -= 1
                end_positions.append(tok_end + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"]   = end_positions
        return tokenized

    tokenized_ds = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    logger.info("Tokenisation complete.")
    return tokenized_ds


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    DATA_FILE = Path(__file__).parent / "sample_dataset.json"
    samples   = load_squad_json(str(DATA_FILE))
    ds        = samples_to_hf_dataset(samples)

    print("\n=== Dataset Preview ===")
    print(ds)
    print("\nFirst training example:")
    for k, v in ds["train"][0].items():
        print(f"  {k}: {v}")
