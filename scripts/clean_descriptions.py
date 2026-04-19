#!/usr/bin/env python3
"""Migration: strip HTML tags and markdown bold/italic markers from existing
listing descriptions in the DB.

The raw data lives on S3 — this script patches the already-bootstrapped
SQLite DB in-place so you don't need to re-import from S3.

Usage:
    uv run python scripts/clean_descriptions.py
    uv run python scripts/clean_descriptions.py --db /path/to/listings.db
    uv run python scripts/clean_descriptions.py --dry-run
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
from app.participant.listing_row_parser import _strip_markup

logger = logging.getLogger("clean_descriptions")

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
            "SELECT listing_id, description FROM listings WHERE description IS NOT NULL"
        ).fetchall()

    logger.info("Loaded %d listings with descriptions", len(rows))
    updates: list[tuple[str, str]] = []
    unchanged = 0

    for listing_id, desc in rows:
        cleaned = _strip_markup(desc)
        if cleaned != desc:
            updates.append((cleaned, listing_id))
        else:
            unchanged += 1

    logger.info("%d need cleaning, %d already clean", len(updates), unchanged)

    if args.dry_run:
        if updates:
            lid, _ = updates[0][1], updates[0][0]
            sample_orig = dict(rows)[lid]
            logger.info("Sample (listing_id=%s):\n  BEFORE: %s\n  AFTER:  %s",
                        lid, sample_orig[:200], updates[0][0][:200])
        logger.info("Dry run — no changes written.")
        return 0

    start = time.perf_counter()
    with get_connection(db_path) as con:
        for i in range(0, len(updates), BATCH):
            con.executemany(
                "UPDATE listings SET description = ? WHERE listing_id = ?",
                updates[i:i + BATCH],
            )
            con.commit()
            logger.info("Updated %d / %d", min(i + BATCH, len(updates)), len(updates))

    logger.info("Done in %.1fs. Rebuild BM25 index to pick up clean text:\n"
                "  uv run python scripts/build_bm25_index.py", time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
