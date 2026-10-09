"""Real (question -> article) pairs from FAQ blocks whose answers link to KB articles.

    python -m src.p3.collect_faq_pairs

Fetches OLT FAQ-style pages, finds question-like headings (ending in '?'), and keeps the KB article links that appear
in the answer. Writes data/p3/faq_pairs.jsonl. Used as extra index text and as a small REAL test set.
"""
import json
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..live.http_utils import PoliteFetcher, RobotsDisallowed
from .articles import FAQ_PATH, P3_DIR
from .crawl_articles import is_article

PAGES = ["https://www.olt.com/main/home/default.php", "https://www.olt.com/main/home/custservicefaq.php"]
QTAGS = ["h2", "h3", "h4", "h5", "summary", "button", "dt", "strong", "a"]


def _kb_links(nodes, base):
    out = []
    for n in nodes:
        for a in ([n] if n.name == "a" else []) + n.find_all("a", href=True):
            u = urljoin(base, a.get("href", "")).split("#")[0]
            if is_article(u):
                out.append(u)
    return list(dict.fromkeys(out))


def extract_faq_pairs(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    pairs, seen = [], set()
    for el in soup.find_all(QTAGS):
        q = " ".join(el.get_text(" ").split())
        if el.name == "a" or not q.endswith("?") or not 3 <= len(q.split()) <= 30 or q in seen:
            continue
        seen.add(q)
        sibs, node = [], el
        for _ in range(6):
            node = node.find_next_sibling()
            if node is None or node.name in QTAGS[:5]:
                break
            sibs.append(node)
        links = _kb_links(sibs, base_url)
        if not links and el.parent is not None:          # accordion: answer sits next to the heading's wrapper
            nxt = el.parent.find_next_sibling()
            links = _kb_links([nxt], base_url) if nxt is not None else []
        if links:
            pairs.append({"question": q, "article_urls": links, "page": base_url})
    return pairs


def main():
    f = PoliteFetcher(min_delay=1.5)
    rows = []
    for page in PAGES:
        try:
            rows += extract_faq_pairs(f.get(page), page)
        except RobotsDisallowed:
            print("blocked by robots.txt:", page)
        except Exception as e:
            print(f"skipped {page}: {type(e).__name__}")
    P3_DIR.mkdir(parents=True, exist_ok=True)
    FAQ_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} FAQ question -> article pairs written to {FAQ_PATH}")


if __name__ == "__main__":
    main()
