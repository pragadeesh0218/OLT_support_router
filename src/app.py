"""FastAPI service.  Run:  uvicorn src.app:app --reload"""
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import hashlib
import json
from datetime import datetime, timezone

from .config import CATEGORIES, LOG_QUESTIONS, RAW_DIR, STATIC_DIR
from .model import Router, mask_pii

state = {"router": None, "stats": {"requests": 0, "auto": 0, "human": 0, "feedback": 0, "latency_ms": []}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["router"] = Router.load()
    yield


app = FastAPI(title="Tax Support Question Router", version="0.1.0", lifespan=lifespan)


def log_question(text: str, result: dict):
    """Phase 2 live-data loop: keep PII-masked production questions for later labeling.
    Only enable (LOG_QUESTIONS=1) if your privacy policy / consent allows it."""
    masked = mask_pii(text)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    row = {"id": "prod-" + hashlib.sha1(masked.lower().encode()).hexdigest()[:16], "source": "production",
           "title": masked, "text": masked, "created": datetime.now(timezone.utc).isoformat(),
           "url": None, "license": "first-party", "model_guess": result["category"]}
    with open(RAW_DIR / "production.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Feedback(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    correct_category: str


@app.post("/classify")
def classify(body: Question):
    t0 = perf_counter()
    result = state["router"].classify(body.question)
    ms = (perf_counter() - t0) * 1000
    st = state["stats"]
    st["requests"] += 1
    st["auto" if result["decision"] == "auto_route" else "human"] += 1
    st["latency_ms"] = (st["latency_ms"] + [ms])[-500:]
    result["latency_ms"] = round(ms, 2)
    result["threshold"] = state["router"].threshold
    if LOG_QUESTIONS:
        log_question(body.question, result)
    return result


@app.post("/feedback")
def feedback(body: Feedback):
    if body.correct_category not in CATEGORIES:
        raise HTTPException(422, f"correct_category must be one of {list(CATEGORIES)}")
    state["router"].learn(body.question, body.correct_category)
    state["stats"]["feedback"] += 1
    return {"status": "learned", "category": body.correct_category}


@app.get("/categories")
def categories():
    return CATEGORIES


@app.get("/metrics")
def metrics():
    st = state["stats"]
    lat = sorted(st["latency_ms"])
    p50 = lat[len(lat) // 2] if lat else None
    p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) >= 20 else (lat[-1] if lat else None)
    return {"requests": st["requests"], "auto_routed": st["auto"], "sent_to_human": st["human"],
            "feedback_received": st["feedback"], "threshold": state["router"].threshold,
            "latency_ms_p50": None if p50 is None else round(p50, 2),
            "latency_ms_p95": None if p95 is None else round(p95, 2)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["router"] is not None}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
