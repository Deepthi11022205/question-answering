# 🔍 QA System — End-to-End Question Answering with HuggingFace Transformers

A production-ready extractive QA system built on `distilbert-base-cased-distilled-squad`.
Ask natural-language questions against any text passage and get precise, scored answers.

---

## Architecture

```
User Question + Context
        │
        ▼
  ┌─────────────┐    POST /ask     ┌──────────────────────────┐
  │  HTML / JS  │ ──────────────►  │   FastAPI Backend        │
  │  Frontend   │ ◄──────────────  │   api/main.py            │
  └─────────────┘   JSON answer    └──────────┬───────────────┘
                                              │
                                   ┌──────────▼───────────────┐
                                   │   Inference Layer        │
                                   │   model/inference.py     │
                                   └──────────┬───────────────┘
                                              │
                                   ┌──────────▼───────────────┐
                                   │   DistilBERT (SQuAD)     │
                                   │   Extracts answer span   │
                                   └──────────────────────────┘
```

**Retriever + Reader pattern** (this project uses context-based Reader only):
- **Retriever**: finds relevant passages from a large corpus (TF-IDF / BM25 / DPR)
- **Reader**: extracts the exact answer span from the retrieved passage ← *this is what we build*

---

## Project Structure

```
qa_system/
├── app.py                      # CLI entry point (serve/ask/eval/train)
├── requirements.txt
│
├── data/
│   ├── sample_dataset.json     # SQuAD-format sample data
│   └── dataset_loader.py       # load, preprocess, tokenise datasets
│
├── model/
│   ├── qa_model.py             # load model + fine-tuning with Trainer API
│   ├── inference.py            # ask_question() + multi-context search
│   └── checkpoints/            # saved fine-tuned models (created at runtime)
│
├── api/
│   └── main.py                 # FastAPI app — POST /ask, POST /ask-multi
│
├── frontend/
│   └── index.html              # single-file HTML/CSS/JS UI
│
├── evaluation/
│   ├── metrics.py              # Exact Match + F1 scorer
│   └── results.json            # evaluation output (created at runtime)
│
└── logs/
    ├── app.log
    └── api.log
```

---

## Quick Start

### 1 · Install dependencies

```bash
cd qa_system
pip install -r requirements.txt
```

### 2 · Start the API server

```bash
python app.py serve
# → http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

### 3 · Open the frontend

Open `frontend/index.html` in your browser (no build step needed).

### 4 · Or use the CLI

```bash
python app.py ask
```

---

## API Reference

### `POST /ask`

```json
// Request
{
  "context":  "Python was created by Guido van Rossum …",
  "question": "Who created Python?",
  "top_k":    3
}

// Response
{
  "answer":     "Guido van Rossum",
  "score":      0.9823,
  "start":      23,
  "end":        40,
  "candidates": [
    {"answer": "Guido van Rossum", "score": 0.9823},
    {"answer": "van Rossum",       "score": 0.0041}
  ],
  "latency_ms": 42.7,
  "error":      null
}
```

### `POST /ask-multi`

```json
{
  "contexts":  ["passage 1 …", "passage 2 …", "passage 3 …"],
  "question":  "…"
}
```

Returns the highest-confidence answer found across all passages.

### `GET /health`

```json
{"status": "ok", "model": "distilbert-base-cased-distilled-squad"}
```

---

## Evaluation

```bash
python app.py eval --data data/sample_dataset.json
```

Output:
```
══════════════════════════════════════════════════
  Exact Match : 83.3%
  F1 Score    : 91.2%
  Samples     : 6
══════════════════════════════════════════════════
```

---

## Fine-tuning on Custom Data

### Step 1 — Prepare your data (SQuAD JSON format)

```json
{
  "version": "1.0",
  "data": [{
    "title": "Your Topic",
    "paragraphs": [{
      "context": "Your passage text …",
      "qas": [{
        "id": "1",
        "question": "Your question?",
        "answers": [{"text": "answer span", "answer_start": 5}],
        "is_impossible": false
      }]
    }]
  }]
}
```

### Step 2 — Run fine-tuning

```bash
python app.py train \
  --data  data/my_data.json \
  --output model/checkpoints/my-qa-model \
  --epochs 3
```

### Step 3 — Use your fine-tuned model

```bash
QA_MODEL_PATH=model/checkpoints/my-qa-model python app.py serve
```

---

## Model Selection — Why DistilBERT?

| Property | DistilBERT | BERT-base |
|----------|-----------|-----------|
| Parameters | 66M | 110M |
| Inference speed | ~2× faster | baseline |
| SQuAD F1 | 86.9 | 88.5 |
| Memory | ~250 MB | ~420 MB |

DistilBERT is **40% smaller** with only ~2% accuracy loss — ideal for APIs.

**Alternatives to consider:**
- `deepset/roberta-base-squad2` — better on unanswerable questions
- `deepset/deberta-v3-base-squad2` — higher accuracy, slower
- `google/electra-small-discriminator` — fastest, lower accuracy

---

## Optimization Tips

1. **Quantization** (CPU speedup ~2×):
   ```python
   import torch
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

2. **ONNX export** (3–5× CPU speedup):
   ```bash
   python -m transformers.onnx --model=distilbert-base-cased-distilled-squad onnx/
   ```

3. **Batch inference** — use `pipeline(batch_size=8)` for bulk queries

4. **Context chunking** — split long documents (>512 tokens) into 256-token chunks with 64-token overlap

---

## Deployment

### Render (free tier)
```yaml
# render.yaml
services:
  - type: web
    name: qa-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py serve --port $PORT
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py", "serve"]
```

### Railway
```bash
railway init
railway up
```

---

## Logging

All requests are logged to `logs/api.log`:
```
2024-01-15 10:23:41 | INFO     | qa_api | POST /ask  →  200  (45.2 ms)
```

Set `LOG_LEVEL=DEBUG` env var for verbose inference logs.
