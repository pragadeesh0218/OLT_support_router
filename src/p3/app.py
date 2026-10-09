"""Phase 3 API.   uvicorn src.p3.app:app --reload     (UI at http://localhost:8000)

POST /search  {"query": "..."}  -> {"decision": "article"|"human", "article": {...url...}, ...}
POST /feedback {"query": "...", "article_id": "...", "helpful": true}
"""
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import LOG_QUESTIONS, STATIC_DIR
from ..model import mask_pii
from .articles import P3_DIR
from .router import build_router

state = {"router": None, "n": 0, "article": 0, "human": 0, "helpful": 0, "not_helpful": 0, "lat": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["router"] = build_router()
    yield


app = FastAPI(title="Support article finder", version="0.3.0", lifespan=lifespan)


class Search(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class Feedback(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    article_id: str | None = None
    helpful: bool


def _log(name: str, row: dict):
    P3_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
    with open(P3_DIR / name, "a") as f:
        f.write(json.dumps(row) + "\n")


@app.post("/search")
def search(body: Search):
    out = state["router"].answer(body.query)
    state["n"] += 1
    state["article" if out["decision"] == "article" else "human"] += 1
    state["lat"] = (state["lat"] + [out["latency_ms"]])[-500:]
    if LOG_QUESTIONS:                      # masked, only if your privacy policy allows it
        _log("queries.jsonl", {"query": mask_pii(body.query), "decision": out["decision"], "reason": out["reason"],
                               "article_id": (out["article"] or {}).get("id"), "confidence": out["confidence"]})
    return out


@app.post("/feedback")
def feedback(body: Feedback):
    state["helpful" if body.helpful else "not_helpful"] += 1
    _log("feedback.jsonl", {"query": mask_pii(body.query), "article_id": body.article_id, "helpful": body.helpful})
    return {"status": "saved"}


@app.get("/stats")
def stats():
    lat = sorted(state["lat"])
    r = state["router"]
    return {"searches": state["n"], "answered_with_article": state["article"], "sent_to_human": state["human"],
            "helpful_votes": state["helpful"], "not_helpful_votes": state["not_helpful"],
            "latency_ms_p50": lat[len(lat) // 2] if lat else None, "articles_indexed": len(r.articles),
            "corpus": r.articles[0].get("source", "live") if r.articles else None,
            "policy_threshold": r.policy.threshold, "notes": r.notes}


@app.get("/health")
def health():
    return {"status": "ok", "ready": state["router"] is not None}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "p3.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
