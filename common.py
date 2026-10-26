"""
Shared code for the two scrapers (scraper.py and scraper_hybrid.py).

Everything that does not depend on which LLM is used lives here:
the Proposal model, page fetching, text cleaning, the extraction prompt,
JSON parsing of the model output, the SQLite storage and config loading.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, field_validator

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

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
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 quantum-thesis-scraper/1.0"
)

# Tags whose content is boilerplate, not proposal text.
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe", "svg"]

log = logging.getLogger("scraper")


class SiteNotFound(Exception):
    """Definitive 'wrong URL' signal: HTTP 404/410 or DNS resolution failure."""


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
    """Fetch raw HTML with requests, with retries on network errors.

    Raises SiteNotFound on 404/410 or DNS failure, since retrying won't help.
    """
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (404, 410):
                raise SiteNotFound(f"HTTP {status}")
            last_err = e
        except requests.ConnectionError as e:
            marker = repr(e)
            if any(s in marker for s in ("NameResolution", "getaddrinfo",
                                         "Name or service not known",
                                         "nodename nor servname")):
                raise SiteNotFound("DNS resolution failed")
            last_err = e
        except requests.RequestException as e:
            last_err = e
        wait = HTTP_BACKOFF_S * (2 ** (attempt - 1))
        log.warning("requests attempt %d/%d failed for %s: %s (retry in %ds)",
                    attempt, HTTP_RETRIES, url, last_err, wait)
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
# LLM prompt & output parsing
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


def build_prompt(url: str, text: str, max_chars: int) -> str:
    """Truncate the page text if needed and fill in the extraction prompt."""
    if len(text) > max_chars:
        log.warning("Truncating text for %s: %d -> %d chars", url, len(text), max_chars)
        text = text[:max_chars]
    return EXTRACTION_PROMPT.format(
        categories=", ".join(f'"{c}"' for c in CATEGORIES), url=url, text=text
    )


def parse_json_array(raw: str, url: str) -> list:
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


def to_proposals(raw: str, url: str) -> list[Proposal]:
    """Parse the raw model output and validate each item into a Proposal."""
    proposals = []
    for i, item in enumerate(parse_json_array(raw, url)):
        if not isinstance(item, dict):
            log.warning("Skipping non-object item %d from %s", i, url)
            continue
        item["source_url"] = url  # enforced, never trusted from the LLM
        try:
            proposals.append(Proposal.model_validate(item))
        except ValidationError as e:
            log.warning("Validation failed for item %d from %s: %s", i, url, e)
    return proposals


class Throttle:
    """Keeps at least min_interval_s seconds between two calls to the model."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def mark(self):
        self._last_call = time.monotonic()


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

def open_db(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_path.read_text())
    # Columns added later on; ALTER fails harmlessly if they already exist.
    for col_def in ("consecutive_failures INTEGER DEFAULT 0",
                    "scrape_count INTEGER DEFAULT 0",
                    "dead INTEGER DEFAULT 0",
                    "dead_reason TEXT"):
        try:
            conn.execute(f"ALTER TABLE sources ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
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
               last_error = NULL, proposals_found = ?, used_playwright = ?,
               consecutive_failures = 0, scrape_count = scrape_count + 1
               WHERE id = ?""",
            (now, len(proposals), int(used_playwright), source_id),
        )


def mark_source_error(conn: sqlite3.Connection, source_id: int, error: str) -> int:
    """Record a failed scrape. Returns the number of consecutive failures."""
    with conn:
        conn.execute(
            """UPDATE sources SET last_status = 'error', last_error = ?,
               consecutive_failures = consecutive_failures + 1 WHERE id = ?""",
            (error[:1000], source_id),
        )
    return conn.execute(
        "SELECT consecutive_failures FROM sources WHERE id = ?", (source_id,)
    ).fetchone()["consecutive_failures"]


def mark_dead(conn: sqlite3.Connection, source_id: int, reason: str):
    with conn:
        conn.execute(
            "UPDATE sources SET dead = 1, dead_reason = ? WHERE id = ?",
            (reason, source_id),
        )
    log.warning("Source marked DEAD (%s), will be pruned: id=%d", reason, source_id)


def prune_dead_sources(conn: sqlite3.Connection, config_path: Path) -> list[str]:
    """Remove dead sources from sources.yaml."""
    dead = {r["url"]: r["dead_reason"] for r in
            conn.execute("SELECT url, dead_reason FROM sources WHERE dead = 1")}
    if not dead:
        return []

    lines = config_path.read_text().splitlines(keepends=True)
    out, pruned, skip_indent = [], [], None
    url_re = re.compile(r"^(\s*)-\s+url:\s*(\S+)\s*$")
    for line in lines:
        if skip_indent is not None:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped and not stripped.startswith("#") and indent > skip_indent:
                continue
            skip_indent = None
        m = url_re.match(line)
        if m and m.group(2) in dead:
            pruned.append(m.group(2))
            skip_indent = len(m.group(1))
            continue
        out.append(line)

    if pruned:
        config_path.write_text("".join(out))
        with conn:
            conn.executemany("DELETE FROM sources WHERE url = ?",
                             [(u,) for u in pruned])
        for u in pruned:
            log.warning("PRUNED from %s: %s (%s)", config_path.name, u, dead[u])
    return pruned


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
# Config, CLI & logging
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> tuple[list[dict], dict]:
    cfg = yaml.safe_load(path.read_text())
    settings = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    sources = cfg.get("sources") or []
    for s in sources:
        if not isinstance(s, dict) or "url" not in s:
            raise ValueError(f"Invalid source entry in {path}: {s!r}")
    return sources, settings


def add_common_args(parser):
    """CLI options shared by both scrapers."""
    parser.add_argument("--config", default="sources.yaml", help="YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if scraped in the last N days")
    parser.add_argument("--url", help="Scrape only this URL (added to DB if new)")
    parser.add_argument("--db", default="proposals.db", help="SQLite database path")
    parser.add_argument("--schema", default="schema.sql", help="Schema file path")
    parser.add_argument("--output", default="proposals.json", help="Aggregated JSON output")
    parser.add_argument("-v", "--verbose", action="store_true")


def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("scraper.log", encoding="utf-8")],
    )


def select_sources(sources: list[dict], only_url: Optional[str]) -> list[dict]:
    """With --url, scrape just that page (reusing its name from the config if listed)."""
    if not only_url:
        return sources
    known = {s["url"]: s.get("name") for s in sources}
    return [{"url": only_url, "name": known.get(only_url)}]
