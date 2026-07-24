#!/usr/bin/env python3
"""
Hybrid quantum thesis proposals scraper: structured parsing + Ollama fallback.

Fetches department/lab pages, attempts structured extraction (regex + BeautifulSoup).
If structured extraction finds <3 proposals, falls back to Ollama LLM (requires local Ollama).

Usage:
    python scraper.py                 # scrape all sources (skips fresh ones)
    python scraper.py --force         # ignore the 7-day freshness window
    python scraper.py --url <URL>     # scrape a single URL (added to DB if new)

Setup:
    1. Install Ollama: https://ollama.ai
    2. Run a model: ollama run mistral (or another model)
    3. Verify it's running on localhost:11434
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

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")

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
    "api_min_interval_s": 1.0,
}

HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
HTTP_BACKOFF_S = 3
API_RETRIES = 2
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 quantum-thesis-scraper/1.0"
)

STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "iframe", "svg"]

log = logging.getLogger("scraper")
MAX_CONSECUTIVE_FAILURES = 4

# Minimum proposals from structured parsing before we skip Ollama fallback
STRUCTURED_CONFIDENCE_THRESHOLD = 3


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
    """Fallback: render the page with headless Chromium."""
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
    except Exception as e:
        log.error("Playwright failed for %s: %s", url, e)
        return None


def extract_text(html: str) -> str:
    """Strip boilerplate and return readable text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:"):
            a.append(f" <{href[len('mailto:'):].split('?')[0]}>")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_page_text(url: str, min_text_length: int) -> tuple[Optional[str], bool]:
    """Return (clean_text, used_playwright)."""
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
    return (text, True) if text else (None, False)


# --------------------------------------------------------------------------- #
# Structured extraction (regex + pattern matching)
# --------------------------------------------------------------------------- #

def extract_structured(text: str, url: str) -> list[Proposal]:
    """
    Extract proposals using regex and pattern matching.
    Looks for: emails, names (capitalized), potential titles, descriptions.
    """
    proposals = []

    # Extract emails (pattern: word@domain.ext)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = set(re.findall(email_pattern, text))

    # Extract potential supervisor names (Capitalized Name or Name Surname)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    names = set(re.findall(name_pattern, text))
    names = {n for n in names if len(n.split()) <= 2 and n not in
             ["The", "Our", "This", "For", "With", "From", "And", "Or"]}

    # Extract sections that look like thesis titles/descriptions
    # Lines in caps or lines ending with ':'
    lines = text.split('\n')
    section_titles = []
    for line in lines:
        stripped = line.strip()
        if (2 < len(stripped) < 200 and
            (stripped.isupper() or stripped.endswith(':')) and
            any(c.isalpha() for c in stripped)):
            section_titles.append(stripped)

    # If we found any structure, create basic proposals
    if emails or names or section_titles:
        # Match emails with nearby names
        for email in list(emails)[:5]:  # Limit to first 5 emails
            # Find name closest to email
            supervisor = next(iter(names), None) if names else None
            if supervisor and supervisor not in email:
                proposals.append(Proposal(
                    title=section_titles[0] if section_titles else None,
                    supervisor=supervisor,
                    email=email,
                    institution=None,
                    country=None,
                    description=None,
                    category=[],
                    is_explicit_proposal=False,
                    source_url=url
                ))

        # Also create entries for names without emails
        for name in list(names - {p.supervisor for p in proposals if p.supervisor})[:3]:
            proposals.append(Proposal(
                title=section_titles[0] if section_titles else None,
                supervisor=name,
                email=None,
                institution=None,
                country=None,
                description=None,
                category=[],
                is_explicit_proposal=False,
                source_url=url
            ))

    log.info("Structured extraction: found %d emails, %d names, %d section titles -> %d proposals",
             len(emails), len(names), len(section_titles), len(proposals))
    return proposals


# --------------------------------------------------------------------------- #
# Ollama extraction
# --------------------------------------------------------------------------- #

OLLAMA_EXTRACTION_PROMPT = """You are extracting master's thesis proposals in physics (quantum computing and related areas) from a university/lab web page.

Return a JSON array. Each element is one thesis proposal, OR — if the page has no explicit proposals but lists researchers/supervisors — one researcher contact.

Each object must have exactly these fields:
- "title": string or null (proposal title; null for a plain researcher contact)
- "supervisor": string or null (person's name)
- "email": string or null (contact email if present in the text)
- "institution": string or null (university/lab name; infer from the page if clear)
- "country": string or null (infer from institution/URL if clear)
- "description": string or null (short summary of the proposal, or the researcher's research area)
- "category": array of strings, only from this list: {categories}
- "is_explicit_proposal": true if this is an actual advertised thesis project, false if it is just a researcher/contact entry

Rules:
- Extract ONLY what is supported by the text; never invent emails or names.
- If the page contains neither proposals nor researcher contacts, return [].
- Output ONLY the JSON array, no markdown fences, no commentary.

Page URL: {url}

Page text:
---
{text}
---"""


class OllamaExtractor:
    """Wraps Ollama API (local) with error handling."""

    def __init__(self, url: str, model: str, min_interval_s: float):
        self.url = url
        self.model = model
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            available = self.model in models
            if not available:
                log.warning("Ollama model '%s' not found. Available: %s. "
                           "Run: ollama run %s", self.model, models, self.model)
            return available
        except Exception as e:
            log.warning("Ollama not available at %s: %s. "
                       "Run: ollama serve", self.url, e)
            return False

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """One API call to Ollama."""
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip() or None
        except requests.RequestException as e:
            log.error("Ollama request failed: %s", e)
            return None

    def extract(self, url: str, text: str, max_chars: int) -> list[Proposal]:
        """Extract proposals using Ollama."""
        if not self._available:
            log.warning("Ollama not available, skipping extraction for %s", url)
            return []

        if len(text) > max_chars:
            log.warning("Truncating text for %s: %d -> %d chars", url, len(text), max_chars)
            text = text[:max_chars]

        prompt = OLLAMA_EXTRACTION_PROMPT.format(
            categories=", ".join(f'"{c}"' for c in CATEGORIES), url=url, text=text
        )

        raw = None
        for attempt in range(1, API_RETRIES + 1):
            self._throttle()
            try:
                raw = self._call_ollama(prompt)
                self._last_call = time.monotonic()
                if raw is not None:
                    break
                log.warning("Empty Ollama response (attempt %d/%d) for %s",
                           attempt, API_RETRIES, url)
            except Exception as e:
                self._last_call = time.monotonic()
                log.warning("Ollama extraction error (attempt %d/%d) for %s: %s",
                           attempt, API_RETRIES, url, e)
                if attempt < API_RETRIES:
                    time.sleep(2 * attempt)

        if raw is None:
            log.error("Ollama extraction failed for %s after %d attempts", url, API_RETRIES)
            return []

        items = self._parse_json_array(raw, url)
        proposals = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                log.warning("Skipping non-object item %d from %s", i, url)
                continue
            item["source_url"] = url
            try:
                proposals.append(Proposal.model_validate(item))
            except ValidationError as e:
                log.warning("Validation failed for item %d from %s: %s", i, url, e)
        return proposals

    @staticmethod
    def _parse_json_array(raw: str, url: str) -> list:
        """Tolerant JSON parsing."""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end <= start:
                log.error("No JSON array found in Ollama output for %s", url)
                return []
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as e:
                log.error("JSON parse error for %s: %s", url, e)
                return []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
        return data if isinstance(data, list) else []


# --------------------------------------------------------------------------- #
# Storage (same as original)
# --------------------------------------------------------------------------- #

def open_db(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_path.read_text())
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
    with conn:
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
    """Record a failed scrape."""
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
    """Regenerate the aggregated flat file from the DB."""
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
# Main pipeline (hybrid: structured first, then Ollama fallback)
# --------------------------------------------------------------------------- #

def scrape_source(conn: sqlite3.Connection, ollama: OllamaExtractor, url: str,
                  name: Optional[str], settings: dict, force: bool) -> bool:
    """Scrape one source: structured parsing + Ollama fallback."""
    source_id = upsert_source(conn, url, name)
    conn.commit()

    row = conn.execute(
        "SELECT dead, scrape_count FROM sources WHERE id = ?", (source_id,)
    ).fetchone()
    if row["dead"]:
        log.info("SKIP (dead): %s", url)
        return False
    first_scrape = row["scrape_count"] == 0

    if not force and is_fresh(conn, url, settings["rescrape_after_days"]):
        log.info("SKIP (fresh, <%dd): %s", settings["rescrape_after_days"], url)
        return True

    log.info("Scraping: %s", url)
    try:
        text, used_playwright = fetch_page_text(url, settings["min_text_length"])
    except SiteNotFound as e:
        mark_source_error(conn, source_id, str(e))
        mark_dead(conn, source_id, f"site not found ({e})")
        log.error("FAILED (not found): %s", url)
        return False

    if text is None:
        failures = mark_source_error(conn, source_id, "fetch failed")
        if failures >= MAX_CONSECUTIVE_FAILURES:
            mark_dead(conn, source_id, f"{failures} consecutive errors")
        log.error("FAILED (fetch, %d consecutive): %s", failures, url)
        return False
    log.info("Fetched %d chars (playwright=%s): %s", len(text), used_playwright, url)

    # Try structured extraction first
    proposals = extract_structured(text, url)

    # Fallback to Ollama if we didn't find enough
    if len(proposals) < STRUCTURED_CONFIDENCE_THRESHOLD:
        log.info("Structured extraction found %d proposals, trying Ollama fallback...", len(proposals))
        ollama_proposals = ollama.extract(url, text, settings["max_text_chars"])
        proposals.extend(ollama_proposals)
        log.info("Ollama added %d proposals (total now: %d)", len(ollama_proposals), len(proposals))
    else:
        log.info("Structured extraction found %d proposals, skipping Ollama", len(proposals))

    if not proposals:
        failures = mark_source_error(conn, source_id, "no proposals extracted")
        if failures >= MAX_CONSECUTIVE_FAILURES:
            mark_dead(conn, source_id, f"{failures} consecutive errors")
        log.warning("FAILED (no proposals): %s", url)
        return False

    save_proposals(conn, source_id, url, proposals, used_playwright)
    n_explicit = sum(p.is_explicit_proposal for p in proposals)
    log.info("OK: %s -> %d entries (%d explicit proposals, %d contacts)",
             url, len(proposals), n_explicit, len(proposals) - n_explicit)

    if first_scrape and not proposals:
        mark_dead(conn, source_id, "0 entries on first scrape")
        return False
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
    parser = argparse.ArgumentParser(description="Hybrid quantum thesis proposals scraper")
    parser.add_argument("--config", default="sources.yaml", help="YAML config file")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if scraped in the last N days")
    parser.add_argument("--url", help="Scrape only this URL (added to DB if new)")
    parser.add_argument("--db", default="proposals.db", help="SQLite database path")
    parser.add_argument("--schema", default="schema.sql", help="Schema file path")
    parser.add_argument("--output", default="proposals.json", help="Aggregated JSON output")
    parser.add_argument("--no-prune", action="store_true",
                        help="Do not auto-remove dead sources from the YAML config")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama server URL")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("scraper.log", encoding="utf-8")],
    )

    try:
        sources, settings = load_config(Path(args.config))
    except (OSError, yaml.YAMLError, ValueError) as e:
        log.error("Cannot load config %s: %s", args.config, e)
        return 1

    if args.url:
        known = {s["url"]: s.get("name") for s in sources}
        sources = [{"url": args.url, "name": known.get(args.url)}]

    conn = open_db(Path(args.db), Path(args.schema))
    ollama = OllamaExtractor(args.ollama_url, args.ollama_model, settings["api_min_interval_s"])

    ok = failed = 0
    try:
        for s in sources:
            if scrape_source(conn, ollama, s["url"], s.get("name"), settings, args.force):
                ok += 1
            else:
                failed += 1
        pruned = [] if args.no_prune else prune_dead_sources(conn, Path(args.config))
        export_json(conn, Path(args.output))
    finally:
        conn.close()

    if pruned:
        log.info("Pruned %d dead source(s) from %s: %s",
                 len(pruned), args.config, ", ".join(pruned))
    log.info("Done: %d ok, %d failed (of %d sources)", ok, failed, len(sources))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
