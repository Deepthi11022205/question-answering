"""
api/main.py
-----------
FastAPI backend for the QA system.

Endpoints
─────────
POST /ask          → answer a question given inline context
POST /ask-multi    → answer against multiple context chunks
GET  /health       → liveness probe
GET  /docs         → Swagger UI (built-in)
"""

import os
import sys
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure project root is on the path
ROOT = os.path.abspath(os.path.join(__file__, "../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.inference import ask_question, ask_question_multi_context

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "logs", "api.log"), mode="a"),
    ],
)
logger = logging.getLogger("qa_api")

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("QA_MODEL_PATH", "distilbert-base-cased-distilled-squad")


# ════════════════════════════════════════════════════════════════════════════
# App lifecycle — warm up model on startup
# ════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up QA model …")
    ask_question(
        context="Python is a programming language.",
        question="What is Python?",
        model_path=MODEL_PATH,
    )
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="QA System API",
    description="Extractive Question-Answering powered by DistilBERT/SQuAD",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow all origins for local dev; restrict in production) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    context:  str = Field(..., min_length=10, description="Passage to search for an answer")
    question: str = Field(..., min_length=3,  description="Natural-language question")
    top_k:    int = Field(default=3, ge=1, le=10, description="Number of candidate answers")

    @field_validator("context", "question")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip()


class AskMultiRequest(BaseModel):
    contexts:  list[str] = Field(..., min_length=1, description="List of context passages")
    question:  str       = Field(..., min_length=3,  description="Natural-language question")

    @field_validator("question")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip()


class Candidate(BaseModel):
    answer: str
    score:  float


class AskResponse(BaseModel):
    answer:     str
    score:      float
    start:      int  | None
    end:        int  | None
    candidates: list[Candidate]
    latency_ms: float
    error:      str  | None = None


# ════════════════════════════════════════════════════════════════════════════
# Middleware — log every request
# ════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s  →  %d  (%.1f ms)",
                request.method, request.url.path, response.status_code, elapsed)
    return response


# ════════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/ask", response_model=AskResponse, tags=["QA"])
def ask(payload: AskRequest):
    """
    Answer a question from a single context paragraph.

    - **context**: The passage to search (10 – ~5000 chars recommended)
    - **question**: What you want to know
    - **top_k**: How many candidate spans to consider internally
    """
    result = ask_question(
        context=payload.context,
        question=payload.question,
        model_path=MODEL_PATH,
        top_k=payload.top_k,
    )

    if result.get("error") and result["score"] == 0.0:
        # Hard validation error (empty input etc.)
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.post("/ask-multi", response_model=AskResponse, tags=["QA"])
def ask_multi(payload: AskMultiRequest):
    """
    Answer a question by searching across **multiple** context passages.
    Returns the highest-confidence answer found in any passage.
    """
    result = ask_question_multi_context(
        contexts=payload.contexts,
        question=payload.question,
        model_path=MODEL_PATH,
    )

    if result.get("error") and result["score"] == 0.0:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ════════════════════════════════════════════════════════════════════════════
# Global exception handler
# ════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ════════════════════════════════════════════════════════════════════════════
# Dev server entrypoint
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
