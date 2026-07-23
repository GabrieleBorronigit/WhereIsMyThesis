-- SQLite schema for the quantum thesis proposals aggregator.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT    NOT NULL UNIQUE,
    name            TEXT,
    last_scraped_at TEXT,               -- ISO 8601 UTC; NULL = never scraped
    last_status     TEXT,               -- 'ok' | 'error'
    last_error      TEXT,
    proposals_found INTEGER DEFAULT 0,
    used_playwright INTEGER DEFAULT 0   -- 1 if the JS fallback was needed last run
);

CREATE TABLE IF NOT EXISTS proposals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id            INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    title                TEXT,
    supervisor           TEXT,
    email                TEXT,
    institution          TEXT,
    country              TEXT,
    description          TEXT,
    category             TEXT    NOT NULL DEFAULT '[]',  -- JSON array of category strings
    is_explicit_proposal INTEGER NOT NULL DEFAULT 0,
    source_url           TEXT    NOT NULL,
    extracted_at         TEXT    NOT NULL,               -- ISO 8601 UTC
    -- Dedup key: same title+supervisor from the same page is the same entry
    UNIQUE (source_url, title, supervisor)
);

CREATE INDEX IF NOT EXISTS idx_proposals_source_id ON proposals(source_id);
CREATE INDEX IF NOT EXISTS idx_proposals_email     ON proposals(email);
