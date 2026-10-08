"""Collect real tax questions from Money Stack Exchange via the official API.

    python -m src.live.collect_stackexchange --max-pages 3
    STACKEXCHANGE_KEY=xxxx python -m src.live.collect_stackexchange --max-pages 10   (higher quota)

Licence: user content is CC BY-SA 4.0 (attribution + share-alike). We keep the question link for attribution.
Read https://stackexchange.com/legal/api-terms-of-use yourself before using this data: the bulk data dump
excludes LLM training, and the API terms are not something this script can clear for you.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ..config import RAW_DIR
from .http_utils import PoliteFetcher

API = "https://api.stackexchange.com/2.3/questions"
# Tags are queried one by one (the API ANDs tags). Tags that do not exist just return 0 items.
DEFAULT_TAGS = ["taxes", "income-tax", "tax-deduction", "tax-refund", "self-employment", "capital-gains",
                "tax-filing", "irs", "w-2", "1099", "tax-return", "estimated-taxes"]


def html_to_text(html: str) -> str:
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())


def to_row(item: dict, keep_authors: bool = False) -> dict:
    row = {
        "id": f"se-{item['question_id']}",
        "source": "stackexchange_money",
        "title": html_to_text(item.get("title", "")),
        "text": html_to_text(item.get("body", ""))[:600],
        "created": datetime.fromtimestamp(item["creation_date"], tz=timezone.utc).isoformat(),
        "url": item.get("link"),
        "license": "CC BY-SA 4.0",
        "tags": item.get("tags", []),
    }
    if keep_authors and item.get("owner"):          # needed if you REDISTRIBUTE the data (attribution)
        row["author"] = {"name": item["owner"].get("display_name"), "link": item["owner"].get("link")}
    return row


def collect(fetcher, tags=DEFAULT_TAGS, max_pages=3, pagesize=100, key=None, keep_authors=False):
    seen, rows = set(), []
    for tag in tags:
        for page in range(1, max_pages + 1):
            params = {"site": "money", "tagged": tag, "page": page, "pagesize": pagesize,
                      "order": "desc", "sort": "creation", "filter": "withbody"}
            if key:
                params["key"] = key
            data = fetcher.get(API, params=params, as_json=True)
            for item in data.get("items", []):
                if item["question_id"] not in seen:
                    seen.add(item["question_id"])
                    rows.append(to_row(item, keep_authors))
            if data.get("backoff"):
                fetcher.sleep(int(data["backoff"]))
            if not data.get("has_more"):
                break
        print(f"tag {tag!r}: total unique so far {len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    ap.add_argument("--keep-authors", action="store_true")
    a = ap.parse_args()
    fetcher = PoliteFetcher(respect_robots=False, min_delay=0.5)   # official API: quota + backoff, not robots
    rows = collect(fetcher, a.tags, a.max_pages, key=os.getenv("STACKEXCHANGE_KEY"), keep_authors=a.keep_authors)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "stackexchange.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
