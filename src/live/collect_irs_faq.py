"""Collect real FAQ questions from IRS.gov (US government work, generally public domain).

    python -m src.live.collect_irs_faq --max-pages 80

Wording is formal (written by the IRS, not by customers), so use it mainly for category coverage and retrieval text.
Casual customer phrasing should come from production logs (LOG_QUESTIONS=1) or other permitted sources.
The crawler respects robots.txt, waits between requests and stays under /faqs.
NOTE: written against generic HTML patterns (question-like headings); check the first run's output by eye.
"""
import argparse
import hashlib
import json
from collections import deque
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..config import RAW_DIR
from .http_utils import PoliteFetcher, RobotsDisallowed

START = ["https://www.irs.gov/faqs"]
QUESTION_TAGS = ["h2", "h3", "h4", "h5", "summary", "button", "dt", "strong"]


def extract_qa(html: str, url: str):
    """Return [(question, answer_snippet)] from headings/accordion titles that end in '?'."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header"]):
        t.decompose()
    out, seen = [], set()
    for el in soup.find_all(QUESTION_TAGS):
        q = " ".join(el.get_text(" ").split())
        if not q.endswith("?") or not (3 <= len(q.split()) <= 40) or q in seen:
            continue
        seen.add(q)
        ans, node = [], el
        for _ in range(6):                                  # collect a few following blocks as the answer
            node = node.find_next_sibling()
            if node is None or node.name in QUESTION_TAGS:
                break
            ans.append(" ".join(node.get_text(" ").split()))
        out.append((q, " ".join(ans)[:400]))
    return out


def links(html: str, base: str):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        u = urljoin(base, a["href"]).split("#")[0].split("?")[0]
        p = urlparse(u)
        if p.netloc.endswith("irs.gov") and p.path.startswith("/faqs"):
            yield u


def crawl(fetcher, start=START, max_pages=80):
    queue, visited, rows = deque(start), set(), []
    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            html = fetcher.get(url)
        except RobotsDisallowed:
            print("skipped (robots.txt):", url)
            continue
        for i, (q, a) in enumerate(extract_qa(html, url)):
            rows.append({"id": "irs-" + hashlib.sha1((url + q).encode()).hexdigest()[:12], "source": "irs_faq", "title": q, "text": a,
                         "created": None, "url": url, "license": "US government work (check page notes)"})
        for u in links(html, url):
            if u not in visited:
                queue.append(u)
    print(f"visited {len(visited)} pages, collected {len(rows)} questions")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=80)
    a = ap.parse_args()
    rows = crawl(PoliteFetcher(min_delay=1.5), START, a.max_pages)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "irs_faq.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", out)


if __name__ == "__main__":
    main()
