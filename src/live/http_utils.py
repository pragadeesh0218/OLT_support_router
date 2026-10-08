"""Polite HTTP: robots.txt check, per-host rate limit, retry/backoff, on-disk cache.

Every live collector goes through this so we never hammer a site and never fetch what robots.txt forbids.
"""
import hashlib
import json
import time
from urllib import robotparser
from urllib.parse import urlparse

import requests

from ..config import CACHE_DIR

USER_AGENT = "support-router-research/0.1 (student project; contact: 22138009@student.hindustanuniv.ac.in)"


class RobotsDisallowed(Exception):
    pass


class PoliteFetcher:
    def __init__(self, session=None, min_delay=1.0, respect_robots=True, cache_dir=CACHE_DIR,
                 max_retries=3, sleep=time.sleep, user_agent=USER_AGENT):
        self.session = session or requests.Session()
        self.min_delay, self.respect_robots = min_delay, respect_robots
        self.cache_dir, self.max_retries, self.sleep, self.ua = cache_dir, max_retries, sleep, user_agent
        self._last_hit = {}
        self._robots = {}

    # -- robots.txt -----------------------------------------------------------------
    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(host + "/robots.txt", headers={"User-Agent": self.ua}, timeout=15)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                elif r.status_code == 404:
                    rp.parse([])                        # no robots.txt -> everything allowed
                else:
                    rp.parse(["User-agent: *", "Disallow: /"])   # 401/403/5xx -> be conservative
            except requests.RequestException:
                rp.parse(["User-agent: *", "Disallow: /"])
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.ua, url)

    # -- fetching ---------------------------------------------------------------------
    def _cache_path(self, url, params):
        key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def get(self, url: str, params=None, as_json=False, use_cache=True):
        if not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        path = self._cache_path(url, params)
        if use_cache and path.exists():
            cached = json.loads(path.read_text())
            return cached["json"] if as_json else cached["text"]

        host = urlparse(url).netloc
        for attempt in range(self.max_retries + 1):
            wait = self.min_delay - (time.monotonic() - self._last_hit.get(host, -1e9))
            if wait > 0:
                self.sleep(wait)
            self._last_hit[host] = time.monotonic()
            r = self.session.get(url, params=params, headers={"User-Agent": self.ua}, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                self.sleep(min(60, 2 ** (attempt + 1)))
                continue
            r.raise_for_status()
            break
        data = r.json() if as_json else None
        text = None if as_json else r.text
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "text": text, "json": data}))
        return data if as_json else text
