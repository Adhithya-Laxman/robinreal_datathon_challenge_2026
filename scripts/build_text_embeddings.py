#!/usr/bin/env python3
"""Offline: compute Cohere multilingual embeddings for every listing.

Usage (inside the api container, so DB + env are wired up):

    docker compose exec api python scripts/build_text_embeddings.py

Or from a host shell if /data is bind-mounted:

    uv run python scripts/build_text_embeddings.py

Idempotent — only embeds listings that don't already have a vector for the
current Bedrock embedding model. Safe to re-run after adding new CSVs.

Writes into the `listing_embeddings` table inside `listings.db`.
Commits every `--batch-size` listings so interrupted runs don't lose work.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterator

# Make the repo importable when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.db import get_connection
from app.participant.embeddings import (
    active_model_id,
    embed_documents,
    ensure_vector_table,
    listing_ids_missing_embedding,
    upsert_embeddings,
)

logger = logging.getLogger("build_text_embeddings")


def _listing_documents(
    db_path: Path,
    listing_ids: list[str],
) -> Iterator[tuple[str, str]]:
    """Yield (listing_id, document_text) for the requested listings.

    Document format is `Title\\n\\n<city, canton>\\n\\nDescription`. Cohere
    truncates at the end, which is fine — title and location go first.
    """
    if not listing_ids:
        return
    placeholders = ",".join("?" * len(listing_ids))
    with get_connection(db_path) as con:
        rows = con.execute(
            f"""
            SELECT listing_id, title, description, city, canton,
                   object_category, rooms, price
            FROM listings
            WHERE listing_id IN ({placeholders})
            """,
            listing_ids,
        ).fetchall()

    for row in rows:
        parts: list[str] = []
        if row["title"]:
            parts.append(str(row["title"]).strip())

        loc_bits: list[str] = []
        if row["city"]:
            loc_bits.append(str(row["city"]))
        if row["canton"]:
            loc_bits.append(str(row["canton"]))
        if loc_bits:
            parts.append(", ".join(loc_bits))

        meta_bits: list[str] = []
        if row["rooms"] is not None:
            meta_bits.append(f"{row['rooms']} rooms")
        if row["price"] is not None:
            meta_bits.append(f"CHF {row['price']}/mo")
        if row["object_category"]:
            meta_bits.append(str(row["object_category"]))
        if meta_bits:
            parts.append(" · ".join(meta_bits))

        if row["description"]:
            parts.append(str(row["description"]).strip())

        document = "\n\n".join(p for p in parts if p)
        yield str(row["listing_id"]), document


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Listings per Bedrock batch call (Cohere v3 max is 96).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of listings to embed (for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even listings that already have a vector for this model.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    db_path = settings.db_path
    model_id = active_model_id()

    if not db_path.exists():
        logger.error("Listings DB not found at %s. Start the api service first.", db_path)
        return 2

    ensure_vector_table(db_path)

    if args.force:
        with get_connection(db_path) as con:
            rows = con.execute("SELECT listing_id FROM listings").fetchall()
        pending = [str(r["listing_id"]) for r in rows]
    else:
        pending = listing_ids_missing_embedding(db_path, model_id=model_id)

    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        logger.info("Nothing to do — all listings already embedded for model=%s.", model_id)
        return 0

    logger.info(
        "Embedding %d listings with model=%s (batch_size=%d)...",
        len(pending),
        model_id,
        args.batch_size,
    )

    total = len(pending)
    done = 0
    started = time.perf_counter()
    for chunk_ids in _batched(pending, args.batch_size):
        docs = list(_listing_documents(db_path, chunk_ids))
        if not docs:
            continue
        ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        vectors = embed_documents(texts)
        upsert_embeddings(
            db_path,
            listing_ids=ids,
            vectors=vectors,
            model_id=model_id,
        )
        done += len(ids)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate if rate else 0.0
        logger.info(
            "  %d / %d done (%.1f/s, eta %.0fs)",
            done,
            total,
            rate,
            eta,
        )

    logger.info("Done in %.1fs.", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
