#!/usr/bin/env python3
"""Migration: infer missing room counts from listing titles for existing DB rows.

Data lives on S3 — this patches the bootstrapped SQLite DB in-place so you
don't need to re-import.

Usage:
    uv run python scripts/backfill_rooms.py
    uv run python scripts/backfill_rooms.py --db /path/to/listings.db
    uv run python scripts/backfill_rooms.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.db import get_connection
from app.participant.listing_row_parser import infer_rooms_from_title

logger = logging.getLogger("backfill_rooms")

BATCH = 500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing to DB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    db_path = args.db or get_settings().db_path
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return 2

    with get_connection(db_path) as con:
        rows = con.execute(
            "SELECT listing_id, title FROM listings WHERE rooms IS NULL AND title IS NOT NULL"
        ).fetchall()

    logger.info("%d listings with null rooms", len(rows))

    updates: list[tuple[float, str]] = []
    for listing_id, title in rows:
        inferred = infer_rooms_from_title(title)
        if inferred is not None:
            updates.append((inferred, listing_id))

    logger.info("%d rooms inferred from title, %d remain null",
                len(updates), len(rows) - len(updates))

    if args.dry_run:
        for rooms, lid in updates[:5]:
            title = dict(rows)[lid]
            logger.info("  listing_id=%-10s rooms=%.1f  title=%s", lid, rooms, title[:80])
        logger.info("Dry run — no changes written.")
        return 0

    start = time.perf_counter()
    with get_connection(db_path) as con:
        for i in range(0, len(updates), BATCH):
            con.executemany(
                "UPDATE listings SET rooms = ? WHERE listing_id = ?",
                updates[i:i + BATCH],
            )
            con.commit()
            logger.info("Updated %d / %d", min(i + BATCH, len(updates)), len(updates))

    logger.info("Done in %.1fs", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
