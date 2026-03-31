"""
api/main.py — v2  (question-only + knowledge base)
"""
import os, sys, logging, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

ROOT = os.path.abspath(os.path.join(__file__, "../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.inference import ask_question, answer as auto_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "logs", "api.log"), mode="a"),
    ],
)
logger = logging.getLogger("qa_api")
MODEL_PATH = os.environ.get("QA_MODEL_PATH", "distilbert-base-cased-distilled-squad")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up QA model ...")
    ask_question(context="Python is a programming language.", question="What is Python?", model_path=MODEL_PATH)
    logger.info("Model ready.")
    yield

app = FastAPI(title="QA System API", description="Just ask a question — no context needed!", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    @field_validator("question")
    @classmethod
    def strip(cls, v): return v.strip()

class AskRequest(BaseModel):
    context:  str = Field(..., min_length=10)
    question: str = Field(..., min_length=3)
    top_k:    int = Field(default=3, ge=1, le=10)
    @field_validator("context", "question")
    @classmethod
    def strip(cls, v): return v.strip()

class AddDocRequest(BaseModel):
    id:      str = Field(..., min_length=1)
    title:   str = Field(..., min_length=1)
    content: str = Field(..., min_length=20)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, (time.perf_counter()-start)*1000)
    return response

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/query", tags=["QA - Question Only"])
def query(payload: QueryRequest):
    """MAIN ENDPOINT — just send a question, system finds the answer automatically."""
    result = auto_answer(question=payload.question, model_path=MODEL_PATH)
    if result.get("error") and result["score"] == 0.0:
        raise HTTPException(status_code=422, detail=result["error"])
    return result

@app.post("/ask", tags=["QA - With Context"])
def ask(payload: AskRequest):
    """Provide your own context + question."""
    result = ask_question(context=payload.context, question=payload.question, model_path=MODEL_PATH, top_k=payload.top_k)
    if result.get("error") and result["score"] == 0.0:
        raise HTTPException(status_code=422, detail=result["error"])
    return result

@app.post("/kb/add", tags=["Knowledge Base"])
def kb_add(payload: AddDocRequest):
    """Add a new document to the knowledge base."""
    from model.retriever import add_document
    try:
        add_document(payload.id, payload.title, payload.content)
        return {"status": "added", "id": payload.id}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.get("/kb/list", tags=["Knowledge Base"])
def kb_list():
    """List all documents in the knowledge base."""
    from model.retriever import _load_knowledge_base, _documents
    _load_knowledge_base()
    return {"count": len(_documents), "documents": [{"id": d["id"], "title": d["title"], "preview": d["content"][:120]+"..."} for d in _documents]}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
