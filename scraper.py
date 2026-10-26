#!/usr/bin/env python3
"""
Quantum thesis proposals scraper (Claude API version).

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
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from common import (
    Proposal, SiteNotFound, Throttle, add_common_args, build_prompt, export_json,
    fetch_page_text, is_fresh, load_config, log, mark_source_error, open_db,
    save_proposals, select_sources, setup_logging, to_proposals, upsert_source,
)

try:
    import anthropic
except ImportError:
    print("Missing dependency: pip install anthropic", file=sys.stderr)
    raise

MODEL = "claude-haiku-4-5-20251001"
API_RETRIES = 4


class Extractor:
    """Wraps the Anthropic client with rate limiting and retries."""

    def __init__(self, min_interval_s: float):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.throttle = Throttle(min_interval_s)

    def extract(self, url: str, text: str, max_chars: int) -> list[Proposal]:
        prompt = build_prompt(url, text, max_chars)

        raw = None
        for attempt in range(1, API_RETRIES + 1):
            self.throttle.wait()
            try:
                resp = self.client.messages.create(
                    model=MODEL,
                    max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.throttle.mark()
                raw = "".join(b.text for b in resp.content if b.type == "text")
                break
            except anthropic.RateLimitError:
                wait = 10 * attempt
                log.warning("Rate limited (attempt %d/%d), sleeping %ds", attempt, API_RETRIES, wait)
                time.sleep(wait)
            except anthropic.APIError as e:
                self.throttle.mark()
                log.warning("API error (attempt %d/%d) for %s: %s", attempt, API_RETRIES, url, e)
                if attempt < API_RETRIES:
                    time.sleep(5 * attempt)
        if raw is None:
            log.error("LLM extraction failed for %s after %d attempts", url, API_RETRIES)
            return []

        return to_proposals(raw, url)


def scrape_source(conn: sqlite3.Connection, extractor: Extractor, url: str,
                  name: Optional[str], settings: dict, force: bool) -> bool:
    """Scrape one source. Returns True on success."""
    source_id = upsert_source(conn, url, name)
    conn.commit()

    if not force and is_fresh(conn, url, settings["rescrape_after_days"]):
        log.info("SKIP (fresh, <%dd): %s", settings["rescrape_after_days"], url)
        return True

    log.info("Scraping: %s", url)
    try:
        text, used_playwright = fetch_page_text(url, settings["min_text_length"])
    except SiteNotFound as e:
        mark_source_error(conn, source_id, str(e))
        log.error("FAILED (not found): %s", url)
        return False
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum thesis proposals scraper")
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY is not set")
        return 1

    try:
        sources, settings = load_config(Path(args.config))
    except (OSError, yaml.YAMLError, ValueError) as e:
        log.error("Cannot load config %s: %s", args.config, e)
        return 1
    sources = select_sources(sources, args.url)

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
