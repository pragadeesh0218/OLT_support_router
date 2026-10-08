"""Replace the hand-written KB keyword summaries with the REAL article text.

    python -m src.live.collect_kb

Reads data/kb_articles.json (titles, urls, categories), fetches each page (robots.txt respected) and writes
data/live/kb_live.json with the page text in the `text` field. Use it with:  ROUTER_KB=data/live/kb_live.json
Check the site's terms first (or ask the owner). Pages that load content with JavaScript return little text;
those are reported so you can paste the text manually or use a headless browser.
"""
import json
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ..config import KB_PATH, LIVE_DIR
from .http_utils import PoliteFetcher, RobotsDisallowed

MIN_USEFUL_CHARS = 300


def page_text(html: str, limit: int = 4000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return " ".join(main.get_text(" ").split())[:limit]


def build(fetcher, kb):
    out, report, cache = [], {"ok": [], "thin": [], "blocked": [], "failed": []}, {}
    for art in kb:
        art = dict(art)
        url = art["url"]
        if url not in cache:                      # several articles may share a URL
            try:
                cache[url] = page_text(fetcher.get(url))
            except RobotsDisallowed:
                cache[url] = None
                report["blocked"].append(url)
            except Exception as e:                # network / HTTP errors: keep going
                cache[url] = None
                report["failed"].append(f"{url} ({type(e).__name__})")
        text = cache[url]
        if text and len(text) >= MIN_USEFUL_CHARS:
            art["text_source"] = "fetched"
            art["text"] = f'{art["title"]}. {text}'
            art["fetched_at"] = datetime.now(timezone.utc).isoformat()
            report["ok"].append(url)
        else:
            art["text_source"] = "summary"          # keep the old keyword summary as fallback
            if text is not None:
                report["thin"].append(url)
        out.append(art)
    return out, report


def main():
    kb = json.loads(KB_PATH.read_text())
    out, report = build(PoliteFetcher(min_delay=1.5), kb)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "kb_live.json").write_text(json.dumps(out, indent=1))
    for k, v in report.items():
        print(f"{k:8s}: {len(v)}")
    for k in ("thin", "blocked", "failed"):
        for u in report[k]:
            print(f"  {k}: {u}")


if __name__ == "__main__":
    main()
