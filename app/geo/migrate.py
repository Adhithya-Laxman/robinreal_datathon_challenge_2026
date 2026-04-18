"""DB migration: add geo enrichment columns to listings table (idempotent)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

_NEW_COLUMNS: list[tuple[str, str]] = [
    ("geo_transit_m", "INTEGER"),
    ("geo_supermarket_m", "INTEGER"),
    ("geo_school_m", "INTEGER"),
    ("geo_university_m", "INTEGER"),
]


def add_geo_columns(db_path: Path) -> None:
    """Add geo enrichment columns and indexes. Safe to call multiple times."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    existing = {row[1] for row in cur.execute("PRAGMA table_info(listings)").fetchall()}
    for col_name, col_type in _NEW_COLUMNS:
        if col_name not in existing:
            cur.execute(f"ALTER TABLE listings ADD COLUMN {col_name} {col_type}")
            print(f"  [migrate] Added column: {col_name} {col_type}")
        else:
            print(f"  [migrate] Column already exists: {col_name}")
    for col_name, _ in _NEW_COLUMNS:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_listings_{col_name} "
            f"ON listings({col_name})"
        )
    conn.commit()
    conn.close()
