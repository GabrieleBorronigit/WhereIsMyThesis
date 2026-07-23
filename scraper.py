#!/usr/bin/env python3
"""
Quantum thesis proposals scraper.

Fetches department/lab pages listed in sources.yaml, extracts clean text,
asks Claude Haiku to structure it into thesis proposals / researcher contacts,
validates the output and stores it in SQLite + proposals.json.

Usage:
    python scraper.py                 # scrape all sources (skips fresh ones)
    python scraper.py --force         # ignore the 7-day freshness window
    python scraper.py --url <URL>     # scrape a single URL (added to DB if new)

Env:
    ANTHROPIC_API_KEY must be set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    import anthropic
except ImportError:
    print("Missing dependency: pip install anthropic", file=sys.stderr)
    raise

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

MODEL = "claude-haiku-4-5-20251001"

CATEGORIES = [
    "Quantum Algorithms",
    "Quantum Hardware",
    "Quantum Error Correction",
    "Quantum ML",
    "Quantum Communication",
    "Quantum Sensing",
    "Condensed Matter",
    "HEP",
    "Quantum Optics",
]

DEFAULT_SETTINGS = {
    "rescrape_after_days": 7,
    "min_text_length": 500,
    "max_text_chars": 60000,
    "api_min_interval_s": 1.5,
}

HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
HTTP_BACKOFF_S = 3  # base for exponential backoff
API_RETRIES = 4
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 quantum-thesis-scraper/1.0"
)

# Tags whose content is boilerplate, not proposal text.
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe", "svg"]

log = logging.getLogger("scraper")


# --------------------------------------------------------------------------- #
# Pydantic model (LLM output validation)
# --------------------------------------------------------------------------- #

class Proposal(BaseModel):
    title: Optional[str] = None
    supervisor: Optional[str] = None
    email: Optional[str] = None
    institution: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None
    category: list[str] = Field(default_factory=list)
    is_explicit_proposal: bool = False
    source_url: str

    @field_validator("category", mode="before")
    @classmethod
    def normalize_categories(cls, v):
        # The LLM sometimes returns a string instead of a list, or categories
        # with slightly different casing; normalize and drop unknown values.
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        canon = {c.lower(): c for c in CATEGORIES}
        out = []
        for item in v:
            match = canon.get(str(item).strip().lower())
            if match and match not in out:
                out.append(match)
        return out

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v):
        if not v:
            return None
        v = str(v).strip().replace("(at)", "@").replace("[at]", "@").replace(" at ", "@")
        v = re.sub(r"\s+", "", v)
        # Reject obviously invalid emails rather than failing the whole record.
        return v if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v) else None

    @field_validator("title", "supervisor", "institution", "country", "description", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #

def fetch_requests(url: str) -> Optional[str]:
    """Fetch raw HTML with requests, with retries on network errors."""
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_err = e
            wait = HTTP_BACKOFF_S * (2 ** (attempt - 1))
            log.warning("requests attempt %d/%d failed for %s: %s (retry in %ds)",
                        attempt, HTTP_RETRIES, url, e, wait)
            if attempt < HTTP_RETRIES:
                time.sleep(wait)
    log.error("All requests attempts failed for %s: %s", url, last_err)
    return None


def fetch_playwright(url: str) -> Optional[str]:
    """Fallback: render the page with headless Chromium (for JS-heavy pages)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright not installed; cannot render %s. "
                  "Run: pip install playwright && playwright install chromium", url)
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=USER_AGENT)
                page.goto(url, timeout=HTTP_TIMEOUT * 1000, wait_until="networkidle")
                return page.content()
            finally:
                browser.close()
    except Exception as e:  # playwright raises many error types; catch broadly
        log.error("Playwright failed for %s: %s", url, e)
        return None


def extract_text(html: str) -> str:
    """Strip boilerplate and return readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    # Keep mailto addresses visible to the LLM even when the link text is a name.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            a.append(f" <{href[len('mailto:'):].split('?')[0]}>")
    text = soup.get_text(separator="\n")
    # Collapse blank-line runs to keep the prompt compact.
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_page_text(url: str, min_text_length: int) -> tuple[Optional[str], bool]:
    """Return (clean_text, used_playwright). None on unrecoverable failure."""
    html = fetch_requests(url)
    text = extract_text(html) if html else ""
    if len(text) >= min_text_length:
        return text, False

    log.info("Text too short (%d chars) for %s — trying Playwright fallback", len(text), url)
    html = fetch_playwright(url)
    if html:
        text = extract_text(html)
        if len(text) >= min_text_length:
            return text, True
    # Return whatever we have (possibly short) if any fetch succeeded at all.
    return (text, True) if text else (None, False)


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #

EXTRACTION_PROMPT = """You are extracting master's thesis proposals in physics \
(quantum computing and related areas) from the text of a university/lab web page.

Return a JSON array. Each element is one thesis proposal, OR — if the page has no \
explicit proposals but lists researchers/supervisors — one researcher contact.

Each object must have exactly these fields:
- "title": string or null (proposal title; null for a plain researcher contact)
- "supervisor": string or null (person's name)
- "email": string or null (contact email if present in the text)
- "institution": string or null (university/lab name; infer from the page if clear)
- "country": string or null (infer from institution/URL if clear)
- "description": string or null (short summary of the proposal, or the researcher's research area)
- "category": array of strings, only from this list: {categories}
  (choose all that apply; empty array if none applies)
- "is_explicit_proposal": true if this is an actual advertised thesis project, \
false if it is just a researcher/contact entry

Rules:
- Extract ONLY what is supported by the text; never invent emails or names.
- If the page contains neither proposals nor researcher contacts, return [].
- Output ONLY the JSON array, no markdown fences, no commentary.

Page URL: {url}

Page text:
---
{text}
---"""


class Extractor:
    """Wraps the Anthropic client with rate limiting and retries."""

    def __init__(self, min_interval_s: float):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def extract(self, url: str, text: str, max_chars: int) -> list[Proposal]:
        if len(text) > max_chars:
            log.warning("Truncating text for %s: %d -> %d chars", url, len(text), max_chars)
            text = text[:max_chars]

        prompt = EXTRACTION_PROMPT.format(
            categories=", ".join(f'"{c}"' for c in CATEGORIES), url=url, text=text
        )

        raw = None
        for attempt in range(1, API_RETRIES + 1):
            self._throttle()
            try:
                resp = self.client.messages.create(
                    model=MODEL,
                    max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._last_call = time.monotonic()
                raw = "".join(b.text for b in resp.content if b.type == "text")
                break
            except anthropic.RateLimitError:
                wait = 10 * attempt
                log.warning("Rate limited (attempt %d/%d), sleeping %ds", attempt, API_RETRIES, wait)
                time.sleep(wait)
            except anthropic.APIError as e:
                self._last_call = time.monotonic()
                log.warning("API error (attempt %d/%d) for %s: %s", attempt, API_RETRIES, url, e)
                if attempt < API_RETRIES:
                    time.sleep(5 * attempt)
        if raw is None:
            log.error("LLM extraction failed for %s after %d attempts", url, API_RETRIES)
            return []

        items = self._parse_json_array(raw, url)
        proposals = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                log.warning("Skipping non-object item %d from %s", i, url)
                continue
            item["source_url"] = url  # enforced, never trusted from the LLM
            try:
                proposals.append(Proposal.model_validate(item))
            except ValidationError as e:
                log.warning("Validation failed for item %d from %s: %s", i, url, e)
        return proposals

    @staticmethod
    def _parse_json_array(raw: str, url: str) -> list:
        """Tolerant JSON parsing: strip fences, locate the outermost array."""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to the substring between the first '[' and last ']'.
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end <= start:
                log.error("No JSON array found in LLM output for %s", url)
                return []
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log.error("JSON parse error for %s: %s", url, e)
                return []
        if isinstance(data, dict):  # model wrapped the array in an object
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return data if isinstance(data, list) else []


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def open_db(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_path.read_text())
    return conn


def upsert_source(conn: sqlite3.Connection, url: str, name: Optional[str]) -> int:
    conn.execute(
        "INSERT INTO sources (url, name) VALUES (?, ?) "
        "ON CONFLICT(url) DO UPDATE SET name = COALESCE(excluded.name, sources.name)",
        (url, name),
    )
    return conn.execute("SELECT id FROM sources WHERE url = ?", (url,)).fetchone()["id"]


def is_fresh(conn: sqlite3.Connection, url: str, days: int) -> bool:
    row = conn.execute(
        "SELECT last_scraped_at, last_status FROM sources WHERE url = ?", (url,)
    ).fetchone()
    if not row or not row["last_scraped_at"] or row["last_status"] != "ok":
        return False
    last = datetime.fromisoformat(row["last_scraped_at"])
    return datetime.now(timezone.utc) - last < timedelta(days=days)


def save_proposals(conn: sqlite3.Connection, source_id: int, url: str,
                   proposals: list[Proposal], used_playwright: bool):
    now = datetime.now(timezone.utc).isoformat()
    with conn:  # single transaction: old rows are replaced atomically
        conn.execute("DELETE FROM proposals WHERE source_id = ?", (source_id,))
        for p in proposals:
            conn.execute(
                """INSERT OR REPLACE INTO proposals
                   (source_id, title, supervisor, email, institution, country,
                    description, category, is_explicit_proposal, source_url, extracted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source_id, p.title, p.supervisor, p.email, p.institution, p.country,
                 p.description, json.dumps(p.category), int(p.is_explicit_proposal),
                 p.source_url, now),
            )
        conn.execute(
            """UPDATE sources SET last_scraped_at = ?, last_status = 'ok',
               last_error = NULL, proposals_found = ?, used_playwright = ?
               WHERE id = ?""",
            (now, len(proposals), int(used_playwright), source_id),
        )


def mark_source_error(conn: sqlite3.Connection, source_id: int, error: str):
    with conn:
        conn.execute(
            "UPDATE sources SET last_status = 'error', last_error = ? WHERE id = ?",
            (error[:1000], source_id),
        )


def export_json(conn: sqlite3.Connection, path: Path):
    """Regenerate the aggregated flat file from the DB (source of truth)."""
    rows = conn.execute(
        """SELECT p.*, s.name AS source_name FROM proposals p
           JOIN sources s ON s.id = p.source_id
           ORDER BY p.source_url, p.title"""
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "title": r["title"],
            "supervisor": r["supervisor"],
            "email": r["email"],
            "institution": r["institution"],
            "country": r["country"],
            "description": r["description"],
            "category": json.loads(r["category"]),
            "is_explicit_proposal": bool(r["is_explicit_proposal"]),
            "source_url": r["source_url"],
            "source_name": r["source_name"],
            "extracted_at": r["extracted_at"],
        })
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info("Wrote %d entries to %s", len(out), path)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def scrape_source(conn: sqlite3.Connection, extractor: Extractor, url: str,
                  name: Optional[str], settings: dict, force: bool) -> bool:
    """Scrape one source. Returns True on success."""
    source_id = upsert_source(conn, url, name)
    conn.commit()

    if not force and is_fresh(conn, url, settings["rescrape_after_days"]):
        log.info("SKIP (fresh, <%dd): %s", settings["rescrape_after_days"], url)
        return True

    log.info("Scraping: %s", url)
    text, used_playwright = fetch_page_text(url, settings["min_text_length"])
    if text is None:
        mark_source_error(conn, source_id, "fetch failed")
        log.error("FAILED (fetch): %s", url)
        return False
    log.info("Fetched %d chars (playwright=%s): %s", len(text), used_playwright, url)

    try:
        proposals = extractor.extract(url, text, settings["max_text_chars"])
    except Exception as e:
        mark_source_error(conn, source_id, f"extraction: {e}")
        log.exception("FAILED (extraction): %s", url)
        return False

    save_proposals(conn, source_id, url, proposals, used_playwright)
    n_explicit = sum(p.is_explicit_proposal for p in proposals)
    log.info("OK: %s -> %d entries (%d explicit proposals, %d contacts)",
             url, len(proposals), n_explicit, len(proposals) - n_explicit)
    return True


def load_config(path: Path) -> tuple[list[dict], dict]:
    cfg = yaml.safe_load(path.read_text())
    settings = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    sources = cfg.get("sources") or []
    for s in sources:
        if not isinstance(s, dict) or "url" not in s:
            raise ValueError(f"Invalid source entry in {path}: {s!r}")
    return sources, settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum thesis proposals scraper")
    parser.add_argument("--config", default="sources.yaml", help="YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if scraped in the last N days")
    parser.add_argument("--url", help="Scrape only this URL (added to DB if new)")
    parser.add_argument("--db", default="proposals.db", help="SQLite database path")
    parser.add_argument("--schema", default="schema.sql", help="Schema file path")
    parser.add_argument("--output", default="proposals.json", help="Aggregated JSON output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("scraper.log", encoding="utf-8")],
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY is not set")
        return 1

    try:
        sources, settings = load_config(Path(args.config))
    except (OSError, yaml.YAMLError, ValueError) as e:
        log.error("Cannot load config %s: %s", args.config, e)
        return 1

    if args.url:
        # Reuse the name from config if the URL is already listed there.
        known = {s["url"]: s.get("name") for s in sources}
        sources = [{"url": args.url, "name": known.get(args.url)}]

    conn = open_db(Path(args.db), Path(args.schema))
    extractor = Extractor(settings["api_min_interval_s"])

    ok = failed = 0
    try:
        for s in sources:
            if scrape_source(conn, extractor, s["url"], s.get("name"), settings, args.force):
                ok += 1
            else:
                failed += 1
        export_json(conn, Path(args.output))
    finally:
        conn.close()

    log.info("Done: %d ok, %d failed (of %d sources)", ok, failed, len(sources))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
