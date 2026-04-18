#!/usr/bin/env python3
"""Offline: build the BM25 index over all listings.

Usage:

    docker compose exec api python scripts/build_bm25_index.py

Or from the host:

    uv run python scripts/build_bm25_index.py

Writes the pickled index next to `listings.db` (same directory), by default
`/data/bm25.pkl` in the Dockerized setup. Safe to re-run.
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
from app.participant.bm25_index import build_index, default_index_path, save_index

logger = logging.getLogger("build_bm25_index")


def _load_corpus(db_path: Path) -> tuple[list[str], list[str]]:
    with get_connection(db_path) as con:
        rows = con.execute(
            """
            SELECT listing_id, title, description, city, canton,
                   object_category, object_type
            FROM listings
            """
        ).fetchall()

    listing_ids: list[str] = []
    documents: list[str] = []
    for row in rows:
        parts: list[str] = []
        for key in ("title", "city", "canton", "object_category", "object_type", "description"):
            val = row[key]
            if val:
                parts.append(str(val))
        listing_ids.append(str(row["listing_id"]))
        documents.append("\n".join(parts))
    return listing_ids, documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the pickled index (defaults to <db_dir>/bm25.pkl).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    db_path = settings.db_path
    if not db_path.exists():
        logger.error("Listings DB not found at %s. Start the api service first.", db_path)
        return 2

    output = args.output or default_index_path()
    logger.info("Loading corpus from %s...", db_path)
    started = time.perf_counter()
    listing_ids, documents = _load_corpus(db_path)
    logger.info("Loaded %d listings in %.2fs", len(listing_ids), time.perf_counter() - started)

    if not listing_ids:
        logger.error("No listings found — nothing to index.")
        return 2

    logger.info("Tokenizing + building BM25...")
    started = time.perf_counter()
    index = build_index(listing_ids, documents)
    logger.info(
        "Built BM25 in %.2fs (avg doc len = %.1f tokens)",
        time.perf_counter() - started,
        index.avg_doc_len,
    )

    save_index(index, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
