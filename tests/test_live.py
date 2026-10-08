"""Phase 2 tests. All HTTP is mocked and the text fixtures below exist only to exercise the code;
no test data is ever used to train the real model."""
import json
import warnings

import pandas as pd
import pytest
import requests

warnings.filterwarnings("ignore")

from src import config
from src.live import build_dataset, collect_irs_faq, collect_kb, collect_stackexchange, label_cli, train_live
from src.live.http_utils import PoliteFetcher, RobotsDisallowed
from src.live.weak_labels import WeakLabeler
from src.model import Router


# ---------------------------------------------------------------- fake HTTP ------------
class Resp:
    def __init__(self, status=200, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """routes: url -> Resp | callable(params) -> Resp | list of Resp (served in order)."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        r = self.routes.get(url, Resp(404))
        if isinstance(r, list):
            r = r.pop(0)
        return r(params) if callable(r) else r


def fetcher(tmp_path, routes, **kw):
    return PoliteFetcher(session=FakeSession(routes), min_delay=0, cache_dir=tmp_path / "cache",
                         sleep=lambda s: None, **kw)


# ---------------------------------------------------------------- http_utils -----------
def test_robots_disallow_blocks_fetch(tmp_path):
    f = fetcher(tmp_path, {"https://x.org/robots.txt": Resp(200, "User-agent: *\nDisallow: /private"),
                           "https://x.org/private/a": Resp(200, "secret"), "https://x.org/ok": Resp(200, "fine")})
    assert f.get("https://x.org/ok") == "fine"
    with pytest.raises(RobotsDisallowed):
        f.get("https://x.org/private/a")


def test_missing_robots_means_allowed_but_error_means_blocked(tmp_path):
    f = fetcher(tmp_path, {"https://a.org/p": Resp(200, "hi")})            # robots.txt -> 404
    assert f.get("https://a.org/p") == "hi"
    g = fetcher(tmp_path / "g", {"https://b.org/robots.txt": Resp(503), "https://b.org/p": Resp(200, "hi")})
    with pytest.raises(RobotsDisallowed):                                    # cannot read robots -> be conservative
        g.get("https://b.org/p")


def test_cache_and_retry(tmp_path):
    sess_routes = {"https://a.org/p": [Resp(503), Resp(200, "ok")]}
    f = fetcher(tmp_path, sess_routes)
    assert f.get("https://a.org/p") == "ok"                  # retried after 503
    n = len(f.session.calls)
    assert f.get("https://a.org/p") == "ok"                  # served from the disk cache
    assert len(f.session.calls) == n


# ---------------------------------------------------------------- Stack Exchange -------
def se_item(i, title="How do I report &amp; pay?", body="<p>Details <b>here</b></p>"):
    return {"question_id": i, "title": title, "body": body, "creation_date": 1700000000 + i,
            "link": f"https://money.stackexchange.com/q/{i}", "tags": ["taxes"],
            "owner": {"display_name": "Someone", "link": "https://x/u/1"}}


def test_stackexchange_rows_dedupe_and_privacy(tmp_path):
    def api(params):
        page = params["page"]
        return Resp(200, data={"items": [se_item(1), se_item(2)] if page == 1 else [se_item(2), se_item(3)],
                               "has_more": page == 1})
    f = fetcher(tmp_path, {collect_stackexchange.API: api}, respect_robots=False)
    rows = collect_stackexchange.collect(f, tags=["taxes"], max_pages=3)
    assert [r["id"] for r in rows] == ["se-1", "se-2", "se-3"]           # question 2 only once
    assert rows[0]["title"] == "How do I report & pay?" and rows[0]["text"] == "Details here"
    assert "author" not in rows[0] and rows[0]["license"] == "CC BY-SA 4.0" and rows[0]["url"].endswith("/q/1")
    kept = collect_stackexchange.to_row(se_item(9), keep_authors=True)
    assert kept["author"]["name"] == "Someone"


# ---------------------------------------------------------------- IRS FAQ --------------
FAQ_HTML = """<html><body><nav><h2>Menu?</h2></nav><main>
<h2>How do I check my refund?</h2><p>Use Where's My Refund.</p><p>It updates daily.</p>
<h3>Can I amend my return?</h3><p>Yes, with Form 1040-X.</p>
<h2>About this page</h2><p>no question here</p>
<a href="/faqs/refunds">r</a><a href="/faqs/private/x">p</a><a href="https://example.com/faqs/z">ext</a>
</main></body></html>"""


def test_extract_qa_questions_only():
    qa = collect_irs_faq.extract_qa(FAQ_HTML, "https://www.irs.gov/faqs")
    assert [q for q, _ in qa] == ["How do I check my refund?", "Can I amend my return?"]   # nav + non-questions ignored
    assert "Where's My Refund" in qa[0][1]


def test_irs_crawl_respects_robots_and_scope(tmp_path):
    routes = {
        "https://www.irs.gov/robots.txt": Resp(200, "User-agent: *\nDisallow: /faqs/private"),
        "https://www.irs.gov/faqs": Resp(200, FAQ_HTML),
        "https://www.irs.gov/faqs/refunds": Resp(200, "<main><h2>When will I get my refund?</h2><p>Soon.</p></main>"),
        "https://www.irs.gov/faqs/private/x": Resp(200, "<main><h2>Hidden question?</h2></main>"),
    }
    rows = collect_irs_faq.crawl(fetcher(tmp_path, routes), max_pages=10)
    titles = {r["title"] for r in rows}
    assert "When will I get my refund?" in titles and "Hidden question?" not in titles
    assert all(r["source"] == "irs_faq" for r in rows)


# ---------------------------------------------------------------- KB text --------------
def test_kb_build_fetched_thin_blocked(tmp_path):
    kb = [{"id": "a", "title": "Refunds", "category": "refund_status", "url": "https://k.org/a", "text": "old"},
          {"id": "b", "title": "Thin", "category": "efile_pin", "url": "https://k.org/b", "text": "old b"},
          {"id": "c", "title": "Blocked", "category": "state_return", "url": "https://k.org/private/c", "text": "old c"}]
    routes = {"https://k.org/robots.txt": Resp(200, "User-agent: *\nDisallow: /private"),
              "https://k.org/a": Resp(200, "<main>" + "refund text " * 60 + "</main>"),
              "https://k.org/b": Resp(200, "<main>loading...</main>")}
    out, rep = collect_kb.build(fetcher(tmp_path, routes), kb)
    by = {a["id"]: a for a in out}
    assert by["a"]["text_source"] == "fetched" and "refund text" in by["a"]["text"]
    assert by["b"]["text_source"] == "summary" and by["b"]["text"] == "old b"        # JS page -> keeps fallback
    assert rep["thin"] == ["https://k.org/b"] and rep["blocked"] == ["https://k.org/private/c"]


# ---------------------------------------------------------------- dataset builder -----
def raw_rows():
    r = lambda i, t, body="": {"id": i, "source": "t", "title": t, "text": body, "created": "2025-03-01T00:00:00+00:00", "url": None}
    return pd.DataFrame([
        r("1", "Where is my refund? my ssn is 123-45-6789"),
        r("2", "where is my refund my ssn is 123-45-6789?"),              # duplicate of 1
        r("3", "hi"),                                                    # too short
        r("4", "税金を申告する方法を教えてください"),                       # not English
        r("5", "My W-2 upload is not working"),
        r("6", "When is the extension deadline this year"),
        r("7", "How do I report freelance income"),
        r("8", "How can I amend a filed return"),
        r("9", "Can I deduct my home office"),
    ])


KB = json.load(open(config.KB_PATH))


def test_build_dataset_cleans_masks_dedupes():
    out = build_dataset.build(raw_rows(), KB, already_labeled={"9"})
    ids = set(out["id"])
    assert ids == {"1", "5", "6", "7", "8"}                              # dup, short, non-English, already-labeled gone
    assert not out["text"].str.contains("123-45-6789").any()
    assert {"suggested_label", "suggested_score", "reason"} <= set(out.columns)


def test_build_dataset_interleaves_classes():
    out = build_dataset.build(raw_rows(), KB)
    assert len(set(out["suggested_label"].head(3))) == 3                # first 3 rows come from different topics


def test_weak_labels_suggestions():
    wl = WeakLabeler(KB)
    assert wl.label("where is my refund after 5 weeks")["label"] == "refund_status"
    assert wl.label("irs rejected my return with a code")["label"] == "efile_pin"
    assert wl.label("i sold bitcoin this year")["label"] == "investments_rental"
    assert wl.label("hello there")["label"] == "other_human"


# ---------------------------------------------------------------- labeling CLI --------
def test_label_cli_accept_pick_skip_undo(tmp_path):
    un, lab = tmp_path / "un.csv", tmp_path / "lab.csv"
    pd.DataFrame([{"id": f"r{i}", "source": "t", "created": None, "url": None, "text": f"question {i}",
                   "suggested_label": "refund_status", "suggested_score": 0.5, "reason": ""} for i in range(1, 5)]
                 ).to_csv(un, index=False)
    answers = iter(["", "3", "s", "u", ""])      # r1 accept | r2 -> w2 | r3 skip | r4: undo r2, then accept
    out = label_cli.run(un, lab, input_fn=lambda _: next(answers), out=lambda *_: None)
    saved = pd.read_csv(lab)
    assert list(saved["id"]) == ["r1", "r4"] and list(saved["label"]) == ["refund_status", "refund_status"]
    # resume: already-labeled rows are not shown again
    shown = []
    label_cli.run(un, lab, input_fn=lambda _: "q", out=shown.append)
    assert any("2 to label" in str(s) for s in shown)


# ---------------------------------------------------------------- live training -------
def labeled_fixture(per_class=20):
    vocab = {"refund_status": ["refund", "deposit", "weeks", "irs"], "efile_pin": ["reject", "pin", "efile", "agi"],
             "w2_1099_import": ["upload", "scan", "w2", "photo"], "extension_deadline": ["extension", "deadline", "late", "april"]}
    rows = []
    for label, words in vocab.items():
        for i in range(per_class):
            rows.append({"id": f"{label}-{i}", "label": label, "source": "t",
                         "text": f"my {words[i % 4]} {words[(i + 1) % 4]} problem number {i}",
                         "created": f"2025-0{1 + i % 9}-15T00:00:00+00:00"})
    return pd.DataFrame(rows)


def test_training_refuses_without_enough_labels():
    with pytest.raises(train_live.NotEnoughLabels):
        train_live.train(labeled_fixture(2))             # 8 labels < 150


def test_training_produces_usable_bundle():
    res = train_live.train(labeled_fixture(), min_labels=40)
    b, r = res["bundle"], res["report"]
    assert b["classes"] == sorted(["refund_status", "efile_pin", "w2_1099_import", "extension_deadline"])
    assert 0 < b["threshold"] < 1 and r["temporal_split"] is not None
    assert max(r["cv_macro_f1"].values()) > 0.8           # fixture is separable by construction
    router = Router(clf=b["clf"], threshold=b["threshold"])
    out = router.classify("my refund deposit problem")
    assert out["category"] == "refund_status"


def test_thin_and_dropped_classes_reported():
    df = pd.concat([labeled_fixture(), pd.DataFrame([{"id": "x1", "label": "state_return", "text": "state form", "source": "t", "created": None}])])
    res = train_live.train(df, min_labels=40)
    assert "state_return" in res["report"]["dropped_classes"]


# ---------------------------------------------------------------- production logging --
def test_production_log_is_masked(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src import app as app_module
    monkeypatch.setattr(app_module, "LOG_QUESTIONS", True)
    monkeypatch.setattr(app_module, "RAW_DIR", tmp_path)
    fb = tmp_path / "fb.jsonl"
    monkeypatch.setattr("src.model.FEEDBACK_PATH", fb)
    with TestClient(app_module.app) as c:
        c.post("/classify", json={"question": "my ssn 123-45-6789 refund missing"})
    row = json.loads((tmp_path / "production.jsonl").read_text().splitlines()[0])
    assert "123-45-6789" not in json.dumps(row) and row["source"] == "production" and row["id"].startswith("prod-")
