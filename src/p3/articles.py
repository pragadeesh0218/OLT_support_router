"""Article store: live crawl (data/p3/articles.jsonl) or a small seed built from the Phase 1 KB index."""
import json
import re
from urllib.parse import urlparse

from ..config import DATA_DIR, KB_PATH

P3_DIR = DATA_DIR / "p3"
ARTICLES_PATH = P3_DIR / "articles.jsonl"
FAQ_PATH = P3_DIR / "faq_pairs.jsonl"
# repeat-count weights when fields are merged into one searchable document
FIELD_WEIGHTS = {"title": 3, "slug": 2, "subtitle": 2, "tags": 2, "headings": 2, "faq": 3, "body": 1}


def norm_url(url: str) -> str:
    p = urlparse(url or "")
    return f"{p.netloc.lower()}{p.path.rstrip('/')}".lower()


def slug_of(url: str) -> str:
    seg = urlparse(url or "").path.rstrip("/").split("/")[-1]
    return re.sub(r"\.(php|html?|aspx?)$", "", seg)


def field_texts(a: dict) -> dict:
    return {
        "title": a.get("title", "") or "",
        "slug": re.sub(r"[-_]+", " ", slug_of(a.get("url", ""))),
        "subtitle": a.get("subtitle", "") or "",
        "tags": " ".join(a.get("tags", []) or []),
        "headings": " . ".join(a.get("headings", []) or []),
        "faq": " . ".join(a.get("faq", []) or []),
        "body": a.get("body", "") or "",
    }


def _seed():
    kb = json.loads(KB_PATH.read_text())
    return [{"id": k["id"], "url": k["url"], "title": k["title"], "subtitle": "", "tags": [], "headings": [],
             "body": k["text"], "category": k.get("category"), "source": "seed"} for k in kb]


def load_articles(path=None, faq_path=None) -> list:
    path = path or ARTICLES_PATH
    if path.exists():
        arts = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        arts = _seed()
    by_url = {norm_url(a["url"]): a for a in arts}
    fp = faq_path or FAQ_PATH
    if fp.exists():
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for u in row.get("article_urls", []):
                a = by_url.get(norm_url(u))
                if a is not None and row["question"] not in a.setdefault("faq", []):
                    a["faq"].append(row["question"])
    return arts
