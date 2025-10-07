"""Helpers for polite web scraping.

- Respect robots.txt (using urllib.robotparser)
- Respect crawl-delay when present
- Polite GET with configurable delay and simple backoff for 429/503
- Simple article extraction helper using BeautifulSoup

Usage:
    from webscraper_utils import polite_get, fetch_article, is_allowed
    resp = polite_get(url)
    text = fetch_article(url)

This file is intentionally self-contained and optional. Use headless browsers only when
robots.txt and the site's Terms of Service allow automated access.
"""
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
import time
from typing import Optional, Dict
from bs4 import BeautifulSoup

# Simple in-memory cache for RobotFileParser objects
_ROBOTS_CACHE: Dict[str, RobotFileParser] = {}

DEFAULT_USER_AGENT = "PoliteScraperBot/1.0 (+https://github.com/)"


def _get_robots_parser_for(url: str, user_agent: str = DEFAULT_USER_AGENT) -> RobotFileParser:
    # Parse the URL and build the robots.txt location for the site
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    # Return cached parser if we've already fetched this domain's robots.txt
    if robots_url in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[robots_url]

    # Create a new RobotFileParser and try to fetch robots.txt via requests
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        # Fetch robots.txt with a short timeout and the provided User-Agent
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=10)
        if resp.status_code == 200:
            # Parse rules into the RobotFileParser
            rp.parse(resp.text.splitlines())
        else:
            # Non-200 responses -> treat as empty robots.txt (permissive)
            rp.parse([])
    except Exception:
        # Network or parsing errors -> treat as empty robots.txt (permissive)
        rp.parse([])

    # Cache and return the parser for future calls
    _ROBOTS_CACHE[robots_url] = rp
    return rp


def is_allowed(url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
    # Return True if the robots.txt rules allow `user_agent` to fetch `url`
    rp = _get_robots_parser_for(url, user_agent)
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        # On parser failure, be conservative and disallow
        return False


def get_crawl_delay(url: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[float]:
    # Return the crawl-delay specified in robots.txt for this user agent (seconds)
    rp = _get_robots_parser_for(url, user_agent)
    try:
        return rp.crawl_delay(user_agent)
    except Exception:
        # If unavailable or parser fails, return None
        return None


def polite_get(url: str,
               user_agent: str = DEFAULT_USER_AGENT,
               session: Optional[requests.Session] = None,
               default_delay: float = 1.0,
               max_retries: int = 3,
               timeout: float = 10.0) -> Optional[requests.Response]:
    # Don't fetch if robots.txt disallows this URL for our User-Agent
    if not is_allowed(url, user_agent=user_agent):
        print(f"Fetching disallowed by robots.txt: {url}")
        return None

    # Respect crawl-delay if provided, otherwise sleep a default short delay
    crawl = get_crawl_delay(url, user_agent=user_agent)
    delay = crawl if (isinstance(crawl, (int, float)) and crawl > 0) else default_delay
    time.sleep(delay)

    # Use an existing Session if provided to reuse TCP connections
    s = session or requests.Session()
    headers = {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}

    # Exponential backoff loop for transient failures (network errors, 429/503)
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            # Attempt the GET with timeout
            resp = s.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            # Network-level error: wait and retry
            print(f"Request exception for {url}: {e}")
            time.sleep(backoff)
            backoff *= 2
            continue

        # Success: return the response
        if resp.status_code == 200:
            return resp

        # Rate-limited or service unavailable: backoff and retry
        if resp.status_code in (429, 503):
            wait = backoff * 2
            print(f"Received {resp.status_code} for {url}; backing off {wait}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
            backoff *= 2
            continue

        # Other HTTP statuses: consider non-retryable and return None
        print(f"Non-retryable status {resp.status_code} for {url}")
        return None

    # Exhausted retries
    print(f"Exceeded retries for {url}")
    return None


def fetch_article(url: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[str]:
    # Use polite_get to fetch the page (this also respects robots.txt)
    resp = polite_get(url, user_agent=user_agent)
    if resp is None:
        return None

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml")

    # Prefer a semantic <article> tag when available
    article = soup.find("article")
    if not article:
        # Fallback heuristics: try common article/body container classes
        candidates = [
            lambda s: s.find("div", class_=lambda x: x and "article-body" in x),
            lambda s: s.find("div", class_=lambda x: x and "article" in x),
            lambda s: s.find("div", class_=lambda x: x and "content" in x),
        ]
        for cand in candidates:
            article = cand(soup)
            if article:
                break

    # If still not found, return the visible text as a last resort
    if not article:
        return soup.get_text(separator="\n", strip=True)

    # Normalize and return the article text
    return article.get_text(separator="\n", strip=True)
