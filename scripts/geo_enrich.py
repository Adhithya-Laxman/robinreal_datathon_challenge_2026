#!/usr/bin/env python3
"""
Offline geo-enrichment script — run ONCE before serving.

What it does:
  1. Adds geo_transit_m / geo_supermarket_m / geo_school_m / geo_university_m
     columns to the listings table (idempotent migration).
  2. Reverse-geocodes the ~11k SRED listings that have lat/lng but no city/canton.
     Uses the offline `reverse_geocoder` package (GeoNames data, no API calls).
  3. Downloads Swiss POI data from Overpass API and caches it to data/geo/.
     (Skipped on subsequent runs — reads from cache instead.)
  4. Computes nearest-POI distances for every listing that has lat/lng.

Runtime:
  - Step 2 (reverse geocode 11k rows): ~5 seconds
  - Step 3 (Overpass fetch, 4 query types): ~2-8 minutes, then instant from cache
  - Step 4 (POI distances for 23k rows): ~30-60 seconds

Usage:
    uv run python scripts/geo_enrich.py
    uv run python scripts/geo_enrich.py --db data/listings.db --geo-cache data/geo
    uv run python scripts/geo_enrich.py --skip-reverse-geocode   # only POI
    uv run python scripts/geo_enrich.py --skip-poi               # only reverse geocode
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Make app/ importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.geo.migrate import add_geo_columns
from app.geo.poi import build_poi_trees, compute_poi_distances, fetch_swiss_pois
from app.geo.reverse_geocode import reverse_geocode_batch


def _parse_args() -> argparse.Namespace:
    # Defaults come from app settings so the script works in both
    # `docker compose exec` (DB at /data/listings.db via volume) and local
    # dev (DB at data/listings.db). Override with --db / --geo-cache.
    settings = get_settings()
    default_db = settings.db_path
    default_cache = Path(default_db).parent / "geo"

    p = argparse.ArgumentParser(description="Geo-enrich the listings SQLite database.")
    p.add_argument("--db", default=str(default_db),
                   help=f"Path to SQLite DB (default: {default_db})")
    p.add_argument("--geo-cache", default=str(default_cache),
                   help=f"Overpass JSON cache dir (default: {default_cache})")
    p.add_argument("--skip-reverse-geocode", action="store_true",
                   help="Skip the city/canton backfill step")
    p.add_argument("--skip-poi", action="store_true",
                   help="Skip POI distance computation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2: reverse geocode
# ---------------------------------------------------------------------------

def _reverse_geocode_step(conn: sqlite3.Connection) -> None:
    print("\n=== Step 2: Reverse-geocoding listings with missing city ===")
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT listing_id, latitude, longitude "
        "FROM listings "
        "WHERE city IS NULL AND latitude IS NOT NULL AND longitude IS NOT NULL"
    ).fetchall()
    print(f"  {len(rows)} listings missing city (all SRED)")
    if not rows:
        print("  Nothing to do.")
        return

    coords = [(r[1], r[2]) for r in rows]
    print(f"  Running offline reverse geocoder on {len(coords)} coordinates...")
    geo = reverse_geocode_batch(coords)

    updates = [
        (g["city"], g["canton"], row[0])
        for g, row in zip(geo, rows)
        if g["city"] is not None
    ]
    no_result = len(rows) - len(updates)
    print(f"  Updating {len(updates)} rows (no result for {no_result})...")
    cur.executemany(
        "UPDATE listings SET city = ?, canton = ? WHERE listing_id = ?",
        updates,
    )
    conn.commit()
    print(f"  Done. city/canton backfilled for {len(updates)} SRED listings.")


# ---------------------------------------------------------------------------
# Step 3-4: POI distances
# ---------------------------------------------------------------------------

def _poi_step(conn: sqlite3.Connection, cache_dir: Path) -> None:
    print("\n=== Steps 3-4: POI distances ===")
    cur = conn.cursor()

    # Only process rows that are still missing at least one POI distance.
    rows = cur.execute(
        "SELECT listing_id, latitude, longitude "
        "FROM listings "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
        "  AND (geo_transit_m IS NULL OR geo_supermarket_m IS NULL "
        "       OR geo_school_m IS NULL OR geo_university_m IS NULL)"
    ).fetchall()
    print(f"  {len(rows)} listings need POI distances")
    if not rows:
        print("  Nothing to do.")
        return

    print("  Fetching/loading Swiss POIs (Overpass — cached after first run)...")
    pois = fetch_swiss_pois(cache_dir)
    trees = build_poi_trees(pois)
    print(f"  BallTrees ready: {', '.join(trees)}")

    print(f"  Computing nearest POI for {len(rows)} listings...")
    batch: list[tuple] = []
    for i, (listing_id, lat, lng) in enumerate(rows):
        d = compute_poi_distances(lat, lng, trees)
        batch.append((
            d.get("transit"),
            d.get("supermarket"),
            d.get("school"),
            d.get("university"),
            listing_id,
        ))
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(rows)} ...")

    print(f"  Writing {len(batch)} records to DB...")
    cur.executemany(
        """UPDATE listings
           SET geo_transit_m     = ?,
               geo_supermarket_m = ?,
               geo_school_m      = ?,
               geo_university_m  = ?
           WHERE listing_id = ?""",
        batch,
    )
    conn.commit()
    print("  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    db_path = Path(args.db)
    cache_dir = Path(args.geo_cache)

    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        print("Bootstrap the DB first by starting the app once.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Step 1: DB migration (add geo columns) ===")
    add_geo_columns(db_path)

    conn = sqlite3.connect(db_path)
    try:
        if not args.skip_reverse_geocode:
            _reverse_geocode_step(conn)
        if not args.skip_poi:
            _poi_step(conn, cache_dir)
    finally:
        conn.close()

    print("\nGeo enrichment complete.")


if __name__ == "__main__":
    main()
