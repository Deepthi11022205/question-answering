"""
app.py
------
Top-level entry point for the QA System.

Usage
─────
  # Start the API server
  python app.py serve

  # Ask a single question (CLI demo)
  python app.py ask

  # Run evaluation on the sample dataset
  python app.py eval

  # Fine-tune on sample data (demo — real training needs more data)
  python app.py train
"""

import argparse
import json
import logging
import os
import sys

# ── Project root on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "logs", "app.log"), mode="a"),
    ],
)
logger = logging.getLogger("app")


# ════════════════════════════════════════════════════════════════════════════
# Subcommand: serve
# ════════════════════════════════════════════════════════════════════════════

def cmd_serve(args):
    import uvicorn
    logger.info("Starting QA API on http://0.0.0.0:%d", args.port)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# ════════════════════════════════════════════════════════════════════════════
# Subcommand: ask (interactive CLI)
# ════════════════════════════════════════════════════════════════════════════

def cmd_ask(args):
    from model.inference import ask_question

    print("\n╔══════════════════════════════════════╗")
    print("║       QA System — Interactive CLI    ║")
    print("╚══════════════════════════════════════╝")
    print("(Type 'quit' to exit)\n")

    default_context = (
        "Python is a high-level, general-purpose programming language created "
        "by Guido van Rossum. Its design philosophy emphasises code readability."
    )
    context = input(f"Context [{default_context[:40]}…]: ").strip() or default_context

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            print("Please enter a question.")
            continue

        result = ask_question(context=context, question=question)
        print(f"\n  Answer     : {result['answer']}")
        print(f"  Confidence : {result['score']:.4f}")
        print(f"  Latency    : {result['latency_ms']} ms")
        if result["error"]:
            print(f"  ⚠ {result['error']}")


# ════════════════════════════════════════════════════════════════════════════
# Subcommand: eval
# ════════════════════════════════════════════════════════════════════════════

def cmd_eval(args):
    from data.dataset_loader import load_squad_json
    from model.inference import ask_question
    from evaluation.metrics import evaluate_dataset

    data_path = args.data or os.path.join(ROOT, "data", "sample_dataset.json")
    logger.info("Evaluating on %s …", data_path)
    samples = load_squad_json(data_path)
    metrics = evaluate_dataset(samples, qa_fn=ask_question, verbose=True)

    out_path = os.path.join(ROOT, "evaluation", "results.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "per_sample"}, f, indent=2)
    logger.info("Metrics saved to %s", out_path)


# ════════════════════════════════════════════════════════════════════════════
# Subcommand: train
# ════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    from data.dataset_loader import load_squad_json, samples_to_hf_dataset, tokenize_dataset
    from model.qa_model import fine_tune, load_model_and_tokenizer

    data_path  = args.data or os.path.join(ROOT, "data", "sample_dataset.json")
    output_dir = args.output or os.path.join(ROOT, "model", "checkpoints", "finetuned-qa")

    logger.info("Loading data from %s …", data_path)
    samples = load_squad_json(data_path)
    ds      = samples_to_hf_dataset(samples)

    model_name = args.model or "distilbert-base-cased-distilled-squad"
    tokenizer, _ = load_model_and_tokenizer(model_name)

    logger.info("Tokenising …")
    tokenized_ds = tokenize_dataset(ds, tokenizer)

    logger.info("Fine-tuning for %d epoch(s) …", args.epochs)
    trainer, saved_path = fine_tune(
        tokenized_datasets=tokenized_ds,
        model_name=model_name,
        output_dir=output_dir,
        num_epochs=args.epochs,
    )
    print(f"\n✓ Fine-tuned model saved to: {saved_path}")


# ════════════════════════════════════════════════════════════════════════════
# CLI parser
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="QA System — serve / ask / eval / train",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--port",   type=int,  default=8000)
    p_serve.add_argument("--reload", action="store_true", help="Hot-reload on code changes")

    # ask
    sub.add_parser("ask", help="Interactive CLI demo")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate on a dataset")
    p_eval.add_argument("--data", type=str, help="Path to SQuAD-format JSON")

    # train
    p_train = sub.add_parser("train", help="Fine-tune on a custom dataset")
    p_train.add_argument("--data",   type=str, help="Path to SQuAD-format JSON")
    p_train.add_argument("--output", type=str, help="Checkpoint output directory")
    p_train.add_argument("--model",  type=str, default="distilbert-base-cased-distilled-squad")
    p_train.add_argument("--epochs", type=int, default=3)

    args = parser.parse_args()
    dispatch = {"serve": cmd_serve, "ask": cmd_ask, "eval": cmd_eval, "train": cmd_train}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
