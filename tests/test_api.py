import json
import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")

from src import config, model
from src.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # keep feedback written by tests away from the real feedback file
    fb = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(config, "FEEDBACK_PATH", fb)
    monkeypatch.setattr(model, "FEEDBACK_PATH", fb)
    with TestClient(app) as c:
        yield c, fb


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize("text,expected", [
    ("my refund has not come after 5 weeks", "refund_status"),
    ("w2 photo upload keeps failing", "w2_1099_import"),
    ("i sold bitcoin how do i report it", "investments_rental"),
    ("how do i file form 1040-x", "amended_return"),
])
def test_routes_clear_questions(client, text, expected):
    c, _ = client
    r = c.post("/classify", json={"question": text}).json()
    assert r["category"] == expected
    assert r["decision"] == "auto_route"
    assert 1 <= len(r["top_articles"]) <= 3


def test_chitchat_goes_to_human(client):
    c, _ = client
    r = c.post("/classify", json={"question": "hello"}).json()
    assert r["decision"] == "human_review"
    assert r["top_articles"] == []


def test_response_shape(client):
    c, _ = client
    r = c.post("/classify", json={"question": "where is my refund"}).json()
    for key in ["category", "confidence", "queue", "decision", "alternatives", "top_articles", "threshold"]:
        assert key in r


def test_pii_is_masked():
    assert "123-45-6789" not in model.clean("my ssn is 123-45-6789 and mail a@b.com")
    assert "a@b.com" not in model.clean("mail a@b.com")


def test_empty_question_rejected(client):
    c, _ = client
    assert c.post("/classify", json={"question": ""}).status_code == 422


def test_feedback_validates_and_persists_masked_text(client):
    c, fb = client
    assert c.post("/feedback", json={"question": "x", "correct_category": "nope"}).status_code == 422
    ok = c.post("/feedback", json={"question": "my ssn 123-45-6789 refund?", "correct_category": "refund_status"})
    assert ok.status_code == 200
    saved = json.loads(fb.read_text().splitlines()[0])
    assert "123-45-6789" not in saved["text"] and saved["label"] == "refund_status"


def test_low_confidence_keeps_prediction_but_defers_to_human(client):
    c, _ = client
    r = c.post("/classify", json={"question": "how do i file a 1040-x"}).json()
    assert r["decision"] == "human_review"      # below the confidence line
    assert r["confidence"] < r["threshold"]
    assert r["category"] == "amended_return"    # best guess is still reported for the agent
