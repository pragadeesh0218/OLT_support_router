"""Phase 3 tests. Articles/HTML below are tiny test fixtures only; HTTP is mocked."""
import json
import warnings

import numpy as np
import pytest
import requests

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import HashingVectorizer

from src.live.http_utils import PoliteFetcher
from src.p3 import articles as art_mod
from src.p3.articles import field_texts, load_articles, slug_of
from src.p3.collect_faq_pairs import extract_faq_pairs
from src.p3.crawl_articles import crawl, is_article, parse_article, parse_sitemap
from src.p3.evaluate import evaluate
from src.p3.policy import needs_agent
from src.p3.pseudo_queries import build_eval_sets, casualize
from src.p3.retrievers import build_stack
from src.p3.router import SearchRouter

BASE = "https://kb.olt.com/KB/articles/article_detail/"


def A(slug, title, subtitle, tags, heads, body, cat):
    return {"id": slug, "url": BASE + slug, "title": title, "subtitle": subtitle, "tags": tags, "headings": heads,
            "body": body, "category": cat, "source": "test"}


FIX = [
    A("where-is-my-refund", "Where Is My Refund and How Long Does It Take", "Check the status of your federal and state refund after e-filing",
      ["refund status", "direct deposit", "irs"], ["Check refund status online", "Why your refund may be delayed"],
      "After the IRS accepts your return the refund is issued, usually by direct deposit within three weeks. You can check the refund status online. A delayed refund can be caused by errors, identity checks or a paper return.", "refund"),
    A("efile-rejected-codes", "E-File Rejected: Common Reject Codes and How to Fix Them", "Fix a rejected electronic return and submit it again",
      ["e-file rejected", "reject code", "prior year agi"], ["Prior year AGI mismatch", "Resubmit your return"],
      "The IRS rejects an electronic return when something does not match, such as the prior year adjusted gross income, a dependent already claimed, or a Social Security number. Correct the error and resubmit.", "efile"),
    A("import-w2-1099", "How to Import a W-2 or 1099 by Photo or PDF", "Upload tax forms instead of typing the numbers",
      ["w-2", "1099", "upload", "import"], ["Upload a PDF", "Take a photo", "Fix wrong numbers after import"],
      "You can upload a PDF or take a photo of your W-2 or 1099 and the import fills in wages and withholding. Review every box after the import because blurry scans can read the wrong number.", "import"),
    A("state-tax-return", "Filing a State Tax Return", "Which states are supported and what a state return costs",
      ["state return", "part-year resident", "nonresident"], ["Which states are supported", "Part year residents"],
      "Most states require their own income tax return. If you moved during the year you file as a part year resident, and if you work in another state you may file as a nonresident.", "state"),
    A("home-office-deduction", "Home Office Deduction for Remote Workers and the Self-Employed", "Claim the space you use regularly and exclusively for work",
      ["home office", "deduction", "square footage"], ["Simplified method", "Regular method"],
      "The simplified method multiplies the square footage of your workspace by a fixed rate. The regular method uses actual expenses such as rent, utilities and insurance, shared by the office percentage.", "deductions"),
    A("standard-vs-itemized", "Standard Deduction vs Itemized Deductions", "Pick the larger deduction",
      ["standard deduction", "itemized", "schedule a"], ["When to itemize", "Standard deduction amounts"],
      "Most people take the standard deduction. Itemize on Schedule A only when mortgage interest, state taxes, charity and medical costs add up to more than the standard amount.", "deductions"),
    A("child-tax-credit", "Child Tax Credit and Credit for Other Dependents", "Credits for qualifying children and other dependents",
      ["child tax credit", "dependents", "schedule 8812"], ["Who is a qualifying child", "Additional child tax credit"],
      "A qualifying child under seventeen with a Social Security number can earn the child tax credit. Other dependents may qualify for a smaller credit. Schedule 8812 figures the refundable part.", "credits"),
    A("amend-return-1040x", "How to Amend a Filed Tax Return with Form 1040-X", "Fix mistakes after your return was accepted",
      ["amended return", "1040-x", "correct mistake"], ["When to amend", "Track an amended return"],
      "File Form 1040-X to correct income, deductions, credits or filing status after the original return was accepted. Do not amend for math errors, the IRS fixes those.", "amend"),
    A("tax-extension-4868", "File a Tax Extension with Form 4868", "Get more time to file, not more time to pay",
      ["extension", "4868", "deadline", "late filing penalty"], ["How to request an extension", "Late filing penalties"],
      "Form 4868 gives six extra months to file but taxes owed are still due in April. Missing the deadline without an extension can bring a late filing penalty.", "extension"),
    A("crypto-gains", "Reporting Cryptocurrency Sales and Gains", "Bitcoin and other digital assets on your return",
      ["crypto", "bitcoin", "capital gains", "1099-da"], ["Cost basis for crypto", "Report on Form 8949"],
      "Selling, swapping or spending cryptocurrency is a taxable event. Work out the cost basis and holding period for each sale and report gains and losses on Form 8949 and Schedule D.", "investments"),
    A("rental-income-depreciation", "Rental Property Income and Depreciation", "Report rent and deduct property costs",
      ["rental income", "schedule e", "depreciation", "landlord"], ["Rental expenses", "Depreciate the building"],
      "Landlords report rent on Schedule E and deduct repairs, insurance and interest. The building is depreciated over twenty seven and a half years.", "rental"),
    A("gig-worker-taxes", "Gig Worker and Freelancer Taxes: Schedule C and Self-Employment Tax", "Uber, DoorDash and freelance income",
      ["uber", "freelance", "schedule c", "1099-nec"], ["Track mileage", "Pay estimated taxes"],
      "Gig workers get a 1099-NEC or 1099-K and report profit on Schedule C. Self-employment tax covers Social Security and Medicare. Track mileage and pay estimated taxes each quarter.", "selfemp"),
]


# ------------------------------------------------------------------ fake HTTP -----------
class Resp:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, routes):
        self.routes = routes

    def get(self, url, params=None, headers=None, timeout=None):
        return self.routes.get(url, Resp(404))


def fetcher(tmp_path, routes):
    return PoliteFetcher(session=FakeSession(routes), min_delay=0, cache_dir=tmp_path / "c", sleep=lambda s: None)


# ------------------------------------------------------------------ articles / crawl ----
def test_slug_and_fields():
    assert slug_of(BASE + "where-is-my-refund") == "where-is-my-refund"
    assert slug_of("https://www.olt.com/main/home/states.php?x=1") == "states"
    assert field_texts(FIX[0])["slug"] == "where is my refund"


SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://kb.olt.com/KB/articles/article_detail/a-one</loc><lastmod>2025-01-02</lastmod></url>
<url><loc>https://kb.olt.com/KB/articles/indiv_article_detail/b-two</loc></url>
<url><loc>https://kb.olt.com/KB/articles/article_detail/secret</loc></url>
<url><loc>https://kb.olt.com/KB/articles/article_detail/thin</loc></url>
<url><loc>https://www.olt.com/main/home/states.php?a=1&amp;b=2</loc></url></urlset>"""


def page(title, desc, kw, body):
    return (f'<html><head><title>{title} | OLT</title><meta name="description" content="{desc}">'
            f'<meta name="keywords" content="{kw}"></head><body><nav><h2>Site menu items</h2></nav><main>'
            f'<h1>{title}</h1><h2>First step to take</h2><p>{body}</p><h3>Second step to take</h3></main>'
            f'<footer>footer text</footer></body></html>')


def test_sitemap_and_article_filter():
    entries = parse_sitemap(SITEMAP)
    assert len(entries) == 5 and entries[0]["lastmod"] == "2025-01-02"
    assert [is_article(e["url"]) for e in entries] == [True, True, True, True, False]
    assert entries[4]["url"].endswith("?a=1&b=2")                    # &amp; unescaped


def test_parse_article_fields():
    a = parse_article(page("Claim a Refund", "How to claim it", "refund, deposit", "x " * 200), BASE + "claim-a-refund")
    assert a["title"] == "Claim a Refund" and a["subtitle"] == "How to claim it" and a["tags"] == ["refund", "deposit"]
    assert a["headings"] == ["First step to take", "Second step to take"] and "Site menu" not in a["body"]
    assert a["id"] == "claim-a-refund"


def test_crawl_handles_blocked_thin_ok(tmp_path):
    routes = {"https://www.olt.com/sitemap.xml": Resp(200, SITEMAP),
              "https://kb.olt.com/robots.txt": Resp(200, "User-agent: *\nDisallow: /KB/articles/article_detail/secret"),
              "https://kb.olt.com/KB/articles/article_detail/a-one": Resp(200, page("A one title", "d", "k1, k2", "word " * 120)),
              "https://kb.olt.com/KB/articles/indiv_article_detail/b-two": Resp(200, page("B two title", "d", "k", "text " * 120)),
              "https://kb.olt.com/KB/articles/article_detail/thin": Resp(200, "<html><body><h1>Thin</h1><main>loading</main></body></html>")}
    rows, rep, n = crawl(fetcher(tmp_path, routes))
    assert n == 4 and [r["id"] for r in rows] == ["a-one", "b-two"] and rows[0]["lastmod"] == "2025-01-02"
    assert rep["blocked"] and rep["thin"] and not rep["failed"]


def test_faq_pairs_need_article_links():
    html = ('<main><h3>Is filing free?</h3><p>Yes for simple returns. See <a href="https://kb.olt.com/KB/articles/article_detail/free-filing">this guide</a></p>'
            '<h3>What is your phone number?</h3><p>Call us.</p><h3>Not a question</h3><p><a href="https://kb.olt.com/KB/articles/article_detail/x">x</a></p></main>')
    pairs = extract_faq_pairs(html, "https://www.olt.com/main/home/default.php")
    assert pairs == [{"question": "Is filing free?", "article_urls": [BASE + "free-filing"], "page": "https://www.olt.com/main/home/default.php"}]


def test_load_articles_merges_faq(tmp_path):
    ap, fp = tmp_path / "a.jsonl", tmp_path / "f.jsonl"
    ap.write_text("".join(json.dumps(a) + "\n" for a in FIX[:2]))
    fp.write_text(json.dumps({"question": "Why was my return rejected?", "article_urls": [FIX[1]["url"] + "/"]}) + "\n")
    arts = load_articles(ap, fp)
    assert arts[1]["faq"] == ["Why was my return rejected?"] and "faq" not in arts[0]
    assert load_articles(tmp_path / "missing.jsonl", tmp_path / "none.jsonl")[0]["source"] == "seed"


# ------------------------------------------------------------------ retrieval -----------
QUERIES = [("where is my refund after 3 weeks", "where-is-my-refund"), ("my efile got rejected prior year agi", "efile-rejected-codes"),
           ("upload my w2 photo", "import-w2-1099"), ("how do i claim home office", "home-office-deduction"),
           ("extension form 4868 deadline", "tax-extension-4868"), ("i sold bitcoin", "crypto-gains")]


def top_id(run_or_scores, arts=FIX):
    s = run_or_scores["fused"] if isinstance(run_or_scores, dict) else run_or_scores
    return arts[int(np.argmax(s))]["id"]


@pytest.mark.parametrize("name", ["bm25", "lsa", "nb"])
def test_each_retriever_finds_obvious_answers(name):
    st = build_stack(FIX)
    hits = sum(top_id(st[name].scores(q)) == want for q, want in QUERIES)
    assert hits >= 5, f"{name}: {hits}/6"


def test_hybrid_and_dense_with_fake_encoder():
    hv = HashingVectorizer(n_features=512, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    st = build_stack(FIX, encoder=lambda texts: hv.transform([t.lower() for t in texts]).toarray())
    assert set(st) == {"bm25", "lsa", "nb", "dense", "hybrid"}
    assert sum(top_id(st["hybrid"].run(q)) == want for q, want in QUERIES) >= 5


def test_eval_sets_hide_their_own_field_and_run():
    sets = build_eval_sets(FIX)
    assert {"title", "subtitle", "tags", "heading", "body_sentence"} <= set(sets)
    assert "title" in sets["title"]["exclude"] and "body" in sets["body_sentence"]["exclude"]
    res = evaluate(FIX)
    assert res["title/raw"]["hybrid"]["top3"] >= 0.75 and res["tags/raw"]["hybrid"]["top3"] >= 0.75


def test_casualize_is_deterministic():
    import random
    assert casualize("How do I claim the home office deduction", random.Random(1)) == casualize("How do I claim the home office deduction", random.Random(1))


# ------------------------------------------------------------------ router / policy -----
@pytest.fixture(scope="module")
def router():
    return SearchRouter(FIX)


def test_exact_title_returns_that_article(router):
    out = router.answer(FIX[7]["title"])
    assert out["decision"] == "article" and out["article"]["url"] == FIX[7]["url"] and out["human"] is None


@pytest.mark.parametrize("q", ["hello", "asdf qwer zxcv", "what is the weather today", "", "   "])
def test_nonsense_goes_to_human(router, q):
    out = router.answer(q)
    assert out["decision"] == "human" and out["article"] is None and out["human"]["url"].startswith("https://")


def test_agent_topics_skip_search(router):
    assert needs_agent("I received an IRS audit notice")
    out = router.answer("someone stole my identity and filed my taxes")
    assert out["decision"] == "human" and out["reason"] == "needs_agent_topic"


def test_policy_report_sane(router):
    r = router.policy.report
    assert r["n_train"] > 100 and 0 < r["threshold"] <= 0.95
    assert r["target_reached"] and r["precision_at_threshold"] >= 0.9


# ------------------------------------------------------------------ API ------------------
def test_api_search_feedback_and_masked_logs(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.p3 import app as app_mod
    monkeypatch.setattr(app_mod, "build_router", lambda: SearchRouter(FIX))
    monkeypatch.setattr(app_mod, "P3_DIR", tmp_path)
    monkeypatch.setattr(app_mod, "LOG_QUESTIONS", True)
    with TestClient(app_mod.app) as c:
        r = c.post("/search", json={"query": FIX[8]["title"] + " ssn 123-45-6789"}).json()
        assert r["decision"] in ("article", "human") and "latency_ms" in r
        assert c.post("/search", json={"query": ""}).status_code == 422
        assert c.post("/feedback", json={"query": "ssn 123-45-6789 refund", "article_id": "x", "helpful": False}).json() == {"status": "saved"}
        s = c.get("/stats").json()
        assert s["searches"] == 1 and s["not_helpful_votes"] == 1 and s["articles_indexed"] == len(FIX)
        assert c.get("/").status_code == 200
    logs = (tmp_path / "queries.jsonl").read_text() + (tmp_path / "feedback.jsonl").read_text()
    assert "123-45-6789" not in logs
