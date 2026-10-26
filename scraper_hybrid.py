#!/usr/bin/env python3
"""
Hybrid quantum thesis proposals scraper: structured parsing + Ollama fallback.

Fetches department/lab pages, attempts structured extraction (regex + BeautifulSoup).
If structured extraction finds <3 proposals, falls back to a local LLM via Ollama.
Sources that keep failing (or return 404 / DNS errors) are marked dead and
removed from sources.yaml at the end of the run.

Usage:
    python scraper_hybrid.py                 # scrape all sources (skips fresh ones)
    python scraper_hybrid.py --force         # ignore the 7-day freshness window
    python scraper_hybrid.py --url <URL>     # scrape a single URL (added to DB if new)
    python scraper_hybrid.py --no-prune      # keep dead sources in sources.yaml

Setup:
    1. Install Ollama: https://ollama.ai
    2. Run a model: ollama run mistral (or another model)
    3. Verify it's running on localhost:11434
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import yaml

from common import (
    Proposal, SiteNotFound, Throttle, add_common_args, build_prompt, export_json,
    fetch_page_text, is_fresh, load_config, log, mark_dead, mark_source_error,
    open_db, prune_dead_sources, save_proposals, select_sources, setup_logging,
    to_proposals, upsert_source,
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
API_RETRIES = 2

MAX_CONSECUTIVE_FAILURES = 4

# Minimum proposals from structured parsing before we skip Ollama fallback
STRUCTURED_CONFIDENCE_THRESHOLD = 3


# --------------------------------------------------------------------------- #
# Structured extraction (regex + pattern matching)
# --------------------------------------------------------------------------- #

def extract_structured(text: str, url: str) -> list[Proposal]:
    """
    Extract proposals using regex and pattern matching.
    Looks for: emails, names (capitalized), potential titles, descriptions.

    TODO: this is way too naive, any capitalized word ends up as a "supervisor".
    It needs a proper rewrite before the results can be trusted.
    """
    proposals = []

    # Extract emails (pattern: word@domain.ext)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    emails = set(re.findall(email_pattern, text))

    # Extract potential supervisor names (Capitalized Name or Name Surname)
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    names = set(re.findall(name_pattern, text))
    names = {n for n in names if len(n.split()) <= 2 and n not in
             ["The", "Our", "This", "For", "With", "From", "And", "Or"]}

    # Lines in caps or ending with ':' look like section titles
    section_titles = []
    for line in text.split('\n'):
        stripped = line.strip()
        if (2 < len(stripped) < 200 and
                (stripped.isupper() or stripped.endswith(':')) and
                any(c.isalpha() for c in stripped)):
            section_titles.append(stripped)
    title = section_titles[0] if section_titles else None

    if emails or names or section_titles:
        # Match emails with nearby names
        for email in list(emails)[:5]:  # Limit to first 5 emails
            supervisor = next(iter(names), None)
            if supervisor and supervisor not in email:
                proposals.append(Proposal(title=title, supervisor=supervisor,
                                          email=email, source_url=url))

        # Also create entries for names without emails
        used = {p.supervisor for p in proposals if p.supervisor}
        for name in list(names - used)[:3]:
            proposals.append(Proposal(title=title, supervisor=name, source_url=url))

    log.info("Structured extraction: found %d emails, %d names, %d section titles -> %d proposals",
             len(emails), len(names), len(section_titles), len(proposals))
    return proposals


# --------------------------------------------------------------------------- #
# Ollama extraction
# --------------------------------------------------------------------------- #

class OllamaExtractor:
    """Wraps the local Ollama API with error handling."""

    def __init__(self, url: str, model: str, min_interval_s: float):
        self.url = url
        self.model = model
        self.throttle = Throttle(min_interval_s)
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name", "").split(":")[0] for m in resp.json().get("models", [])]
            if self.model not in models:
                log.warning("Ollama model '%s' not found. Available: %s. "
                            "Run: ollama run %s", self.model, models, self.model)
                return False
            return True
        except Exception as e:
            log.warning("Ollama not available at %s: %s. Run: ollama serve", self.url, e)
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """One API call to Ollama."""
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or None
        except requests.RequestException as e:
            log.error("Ollama request failed: %s", e)
            return None

    def extract(self, url: str, text: str, max_chars: int) -> list[Proposal]:
        """Extract proposals using Ollama."""
        if not self._available:
            log.warning("Ollama not available, skipping extraction for %s", url)
            return []

        prompt = build_prompt(url, text, max_chars)

        raw = None
        for attempt in range(1, API_RETRIES + 1):
            self.throttle.wait()
            raw = self._call_ollama(prompt)
            self.throttle.mark()
            if raw is not None:
                break
            log.warning("Empty Ollama response (attempt %d/%d) for %s", attempt, API_RETRIES, url)
            if attempt < API_RETRIES:
                time.sleep(2 * attempt)

        if raw is None:
            log.error("Ollama extraction failed for %s after %d attempts", url, API_RETRIES)
            return []

        return to_proposals(raw, url)


# --------------------------------------------------------------------------- #
# Main pipeline (hybrid: structured first, then Ollama fallback)
# --------------------------------------------------------------------------- #

def record_failure(conn: sqlite3.Connection, source_id: int, error: str) -> int:
    """Mark a failed scrape and kill the source after too many failures in a row."""
    failures = mark_source_error(conn, source_id, error)
    if failures >= MAX_CONSECUTIVE_FAILURES:
        mark_dead(conn, source_id, f"{failures} consecutive errors")
    return failures


def scrape_source(conn: sqlite3.Connection, ollama: OllamaExtractor, url: str,
                  name: Optional[str], settings: dict, force: bool) -> bool:
    """Scrape one source: structured parsing + Ollama fallback."""
    source_id = upsert_source(conn, url, name)
    conn.commit()

    row = conn.execute("SELECT dead FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row["dead"]:
        log.info("SKIP (dead): %s", url)
        return False

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
        failures = record_failure(conn, source_id, "fetch failed")
        log.error("FAILED (fetch, %d consecutive): %s", failures, url)
        return False
    log.info("Fetched %d chars (playwright=%s): %s", len(text), used_playwright, url)

    # Try structured extraction first, fall back to Ollama if it finds too little
    proposals = extract_structured(text, url)
    if len(proposals) < STRUCTURED_CONFIDENCE_THRESHOLD:
        log.info("Structured extraction found %d proposals, trying Ollama fallback...", len(proposals))
        ollama_proposals = ollama.extract(url, text, settings["max_text_chars"])
        proposals.extend(ollama_proposals)
        log.info("Ollama added %d proposals (total now: %d)", len(ollama_proposals), len(proposals))
    else:
        log.info("Structured extraction found %d proposals, skipping Ollama", len(proposals))

    if not proposals:
        record_failure(conn, source_id, "no proposals extracted")
        log.warning("FAILED (no proposals): %s", url)
        return False

    save_proposals(conn, source_id, url, proposals, used_playwright)
    n_explicit = sum(p.is_explicit_proposal for p in proposals)
    log.info("OK: %s -> %d entries (%d explicit proposals, %d contacts)",
             url, len(proposals), n_explicit, len(proposals) - n_explicit)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid quantum thesis proposals scraper")
    add_common_args(parser)
    parser.add_argument("--no-prune", action="store_true",
                        help="Do not auto-remove dead sources from the YAML config")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama server URL")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="Ollama model to use")
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        sources, settings = load_config(Path(args.config))
    except (OSError, yaml.YAMLError, ValueError) as e:
        log.error("Cannot load config %s: %s", args.config, e)
        return 1
    sources = select_sources(sources, args.url)

    conn = open_db(Path(args.db), Path(args.schema))
    ollama = OllamaExtractor(args.ollama_url, args.ollama_model, settings["api_min_interval_s"])

    ok = failed = 0
    pruned = []
    try:
        for s in sources:
            if scrape_source(conn, ollama, s["url"], s.get("name"), settings, args.force):
                ok += 1
            else:
                failed += 1
        if not args.no_prune:
            pruned = prune_dead_sources(conn, Path(args.config))
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
