"""Crawl every KB article listed in OLT's sitemap.   python -m src.p3.crawl_articles [--limit 20]

sitemap.xml -> article URLs -> fetch (robots.txt respected, rate-limited, cached) -> data/p3/articles.jsonl
Check the site's terms (or ask OLT) before using the text. Parsing uses generic HTML patterns:
look at data/p3/articles.jsonl after the first run and adjust parse_article if fields come out empty.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from ..live.http_utils import PoliteFetcher, RobotsDisallowed
from .articles import ARTICLES_PATH, P3_DIR, slug_of

SITEMAP = "https://www.olt.com/sitemap.xml"
MIN_BODY = 200


def parse_sitemap(xml_text: str):
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    out = []
    for node in root.iter():
        if node.tag.endswith("}url") or node.tag == "url":
            loc = lastmod = None
            for ch in node:
                tag = ch.tag.split("}")[-1]
                if tag == "loc":
                    loc = (ch.text or "").strip()
                elif tag == "lastmod":
                    lastmod = (ch.text or "").strip()
            if loc:
                out.append({"url": loc, "lastmod": lastmod})
    return out


def child_sitemaps(xml_text: str):
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    return [n.findtext("{*}loc", "").strip() for n in root.iter() if n.tag.split("}")[-1] == "sitemap"]


def is_article(url: str) -> bool:
    return "kb.olt.com/KB/articles/" in url and "article_detail/" in url


def _meta(soup, name):
    m = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": f"og:{name}"})
    return (m.get("content") or "").strip() if m else ""


def parse_article(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "") or _meta(soup, "title") \
        or (soup.title.get_text(strip=True) if soup.title else "")
    subtitle = _meta(soup, "description")
    tags = [t.strip() for t in re.split(r"[,;]", _meta(soup, "keywords")) if t.strip()]
    for t in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "aside"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    heads = [h.get_text(" ", strip=True) for h in main.find_all(["h2", "h3"])]
    heads = [h for h in dict.fromkeys(heads) if 2 <= len(h.split()) <= 20 and h.lower() != title.lower()]
    body = " ".join(main.get_text(" ").split())[:6000]
    return {"id": slug_of(url), "url": url, "title": title, "subtitle": subtitle, "tags": tags, "headings": heads,
            "body": body, "category": None, "source": "crawl"}


def crawl(fetcher, sitemap=SITEMAP, limit=None):
    xml = fetcher.get(sitemap)
    entries = parse_sitemap(xml)
    for child in child_sitemaps(xml):               # sitemap index files
        entries += parse_sitemap(fetcher.get(child))
    urls = list(dict.fromkeys(e["url"] for e in entries if is_article(e["url"])))
    lastmod = {e["url"]: e["lastmod"] for e in entries}
    if limit:
        urls = urls[:limit]
    rows, report = [], {"ok": 0, "thin": [], "blocked": [], "failed": []}
    for u in urls:
        try:
            a = parse_article(fetcher.get(u), u)
        except RobotsDisallowed:
            report["blocked"].append(u); continue
        except Exception as e:
            report["failed"].append(f"{u} ({type(e).__name__})"); continue
        a["lastmod"] = lastmod.get(u)
        if len(a["body"]) < MIN_BODY or not a["title"]:
            report["thin"].append(u); continue
        rows.append(a); report["ok"] += 1
    return rows, report, len(urls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows, report, n = crawl(PoliteFetcher(min_delay=1.5), limit=a.limit)
    P3_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"article URLs in sitemap: {n} | saved: {report['ok']} | thin: {len(report['thin'])} | "
          f"blocked: {len(report['blocked'])} | failed: {len(report['failed'])}")
    for k in ("thin", "blocked", "failed"):
        for u in report[k][:10]:
            print(f"  {k}: {u}")
    print("wrote", ARTICLES_PATH)


if __name__ == "__main__":
    main()
