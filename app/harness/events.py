"""Session event store — SQLite-backed click/dwell/save log.

The table lives inside the same data/listings.db for zero extra infra.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Literal

from app.config import get_settings

EventType = Literal["click", "dwell", "save"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS session_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    listing_id  TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    dwell_ms    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_session_events_session_ts
    ON session_events(session_id, ts DESC);
"""


def ensure_events_table(db_path: Path | None = None) -> None:
    path = db_path or get_settings().db_path
    con = sqlite3.connect(path)
    try:
        con.executescript(_CREATE_TABLE)
        con.commit()
    finally:
        con.close()


def log_event(
    *,
    session_id: str,
    event_type: EventType,
    listing_id: str,
    dwell_ms: int | None = None,
    db_path: Path | None = None,
) -> None:
    path = db_path or get_settings().db_path
    con = sqlite3.connect(path)
    try:
        con.execute(
            "INSERT INTO session_events (session_id, event_type, listing_id, ts, dwell_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, listing_id, int(time.time() * 1000), dwell_ms),
        )
        con.commit()
    finally:
        con.close()


def get_recent_clicks(
    session_id: str,
    *,
    n: int = 10,
    db_path: Path | None = None,
) -> list[str]:
    """Return listing_ids of the last `n` clicked/saved listings in the session."""
    path = db_path or get_settings().db_path
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT listing_id
            FROM session_events
            WHERE session_id = ? AND event_type IN ('click', 'save')
            ORDER BY ts DESC
            LIMIT ?
            """,
            (session_id, n),
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def get_recent_events(
    session_id: str,
    *,
    n: int = 20,
    db_path: Path | None = None,
) -> list[dict]:
    path = db_path or get_settings().db_path
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT event_type, listing_id, ts, dwell_ms
            FROM session_events
            WHERE session_id = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (session_id, n),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]
