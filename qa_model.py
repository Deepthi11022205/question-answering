"""
model/qa_model.py
-----------------
Loads a pretrained extractive-QA model (DistilBERT by default),
optionally fine-tunes it on a custom dataset, and saves/loads checkpoints.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    pipeline,
)

logger = logging.getLogger(__name__)

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "distilbert-base-cased-distilled-squad"
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# 1. Load tokenizer + model
# ════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    """
    Downloads (or loads from cache) the tokenizer and model weights.

    Why distilbert-base-cased-distilled-squad?
    ──────────────────────────────────────────
    • 40 % smaller than BERT-base, 60 % faster at inference
    • Already fine-tuned on SQuAD 1.1 → strong out-of-the-box accuracy
    • Cased version preserves named-entity capitalisation
    • Great starting point for further fine-tuning on domain data
    """
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model: %s", model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info("Model device: %s | params: %s",
                device, f"{sum(p.numel() for p in model.parameters()):,}")

    return tokenizer, model


# ════════════════════════════════════════════════════════════════════════════
# 2. Fine-tune on custom / SQuAD data
# ════════════════════════════════════════════════════════════════════════════

def fine_tune(
    tokenized_datasets,           # DatasetDict with train + validation splits
    model_name: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    fp16: bool = False,           # enable on GPU for ~2× speedup
):
    """
    Fine-tune a QA model with the HuggingFace Trainer API.

    Steps
    ─────
    1. Load the base model & tokenizer
    2. Define TrainingArguments (lr, epochs, batch size, etc.)
    3. Create a Trainer instance
    4. Call trainer.train()
    5. Save model + tokenizer to output_dir
    """
    output_dir = output_dir or str(CHECKPOINT_DIR / "finetuned-qa")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(model_name)

    # ── Training hyper-parameters ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=fp16 and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(Path(output_dir) / "logs"),
        logging_steps=50,
        report_to="none",          # disable wandb / tensorboard by default
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    logger.info("Starting fine-tuning for %d epoch(s) …", num_epochs)
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuned model saved to %s", output_dir)

    return trainer, output_dir


# ════════════════════════════════════════════════════════════════════════════
# 3. Build inference pipeline
# ════════════════════════════════════════════════════════════════════════════

def build_pipeline(model_path: str = DEFAULT_MODEL):
    """
    Returns a HuggingFace `pipeline` object ready for inference.
    Pass the checkpoint directory path to use a fine-tuned model.
    """
    device = 0 if torch.cuda.is_available() else -1   # -1 = CPU
    qa_pipe = pipeline(
        "question-answering",
        model=model_path,
        tokenizer=model_path,
        device=device,
    )
    logger.info("Inference pipeline ready (model=%s)", model_path)
    return qa_pipe


# ════════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    pipe = build_pipeline()

    context  = ("Python is a high-level, general-purpose programming language. "
                "Guido van Rossum began working on Python in the late 1980s.")
    question = "Who created Python?"

    result = pipe(question=question, context=context)
    print("\n=== Smoke Test ===")
    print(f"Q: {question}")
    print(f"A: {result['answer']}  (score: {result['score']:.4f})")
