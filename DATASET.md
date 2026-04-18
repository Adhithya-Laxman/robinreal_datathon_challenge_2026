# Dataset Guide

Practical reference for the Datathon 2026 listing data. Start here before you write any retrieval / ranking code.

## TL;DR

- **22,819 listings** from 3 sources: COMPARIS (10,917) · SRED (11,105) · ROBINREAL (797).
- Mostly rentals (21,785 RENT, 1 SALE, 1,033 unspecified).
- **Text metadata** lives in CSVs under `raw_data/` and is loaded into `/data/listings.db` (SQLite) on first API boot.
- **Photos** live in two places: the full S3-mirrored bundle under `downloads/prod/<source>/images/platform_id=.../*.jpg` (not committed), and SRED one-per-listing montages under `raw_data/sred_images/<listing_id>.jpeg`.
- **Join key** everywhere: `(scrape_source, platform_id)`.

## On-disk layout

```
repo/
├── raw_data/                                     <-- organizer bundle (gitignored)
│   ├── robinreal_data_withimages-*.csv              797 ROBINREAL rows   (3.5 MB)
│   ├── structured_data_withimages-*.csv           4,160 COMPARIS rows    ( 47 MB)
│   ├── structured_data_withoutimages-*.csv        6,757 COMPARIS rows    ( 68 MB)
│   ├── sred_data_withmontageimages_latlong.csv   11,105 SRED rows        ( 34 MB)
│   └── sred_images/                              11,106 montage JPEGs   (136 MB)
│       └── <listing_id>.jpeg                      one montage per listing
│
├── downloads/prod/                               <-- from S3 via scripts/download_images.sh (gitignored)
│   ├── robinreal/images/
│   │   └── platform_id=<hex>/                     ~7 images per listing
│   │       ├── 0-ff0000000033993988.jpg           numeric prefix = display order
│   │       └── ...                                5,385 files total  (792 MB)
│   └── comparis/images/
│       └── platform_id=<int>/                     ~7 images per listing
│           ├── 2737371effef0334__getdirect        quirky ext — still a JPEG
│           └── ...                                25,076 files total  (3.4 GB)
│
└── /data/listings.db                             <-- SQLite built on boot from raw_data/*.csv
                                                  (lives in the listings_data Docker volume)
```

Quick visual at any time:

```bash
tree -L 2 raw_data/ downloads/
# per-listing file count inside one folder:
ls downloads/prod/comparis/images/platform_id=37090183/ | wc -l
```

## The 3 sources, compared

| Source     | Rows    | City data | Price | Rooms | lat/lng | Images on S3 | Notes |
|------------|--------:|:---------:|:-----:|:-----:|:-------:|:------------:|---|
| COMPARIS   | 10,917  | **100%**  | 94%   | 73%   | ~100%   | 4,160 (of 10,917) | Richest metadata. Scraped from comparis.ch. `platform_id` is a short integer. |
| SRED       | 11,105  | **0%**    | 100%  | 100%  | ~100%   | local montages only | **No city / postal / canton / street** — lat/lng is the only geo signal. Every listing has exactly 1 pre-computed montage image in `raw_data/sred_images/`. |
| ROBINREAL  |   797   | **100%**  | 97%   | 100%  | ~100%   | 797 (all) | Small, clean. `platform_id` is a hex string. |

Nullability across the full DB (from `listings` table):

| Column            | Missing |
|-------------------|--------:|
| `description`     |   0.2%  |
| `price`           |   2.9%  |
| `latitude`/`longitude` | 7.2% |
| `rooms`           |  13.1%  |
| `area`            |  18.6%  |
| `city`            |  48.7%  *(all from SRED)* |
| `postal_code`     |  48.8%  *(all from SRED)* |
| `street`          |  57.6%  |
| `canton`          |  65.2%  |
| `available_from`  |  69.9%  |

> **Practical consequence:** naive `city == "Zurich"` filters silently drop **every single SRED listing**. If you want SRED to participate in city-constrained queries, either (a) reverse-geocode lat/lng → city/postal code once at import time, or (b) always apply city filters as `city == 'Zurich' OR (city IS NULL AND point_within_zurich_bbox)`.

## CSV schema (all 4 share the same 52 columns)

Primary identifiers and geo:

| Column            | Type   | Notes |
|-------------------|--------|---|
| `id`              | string | = `platform_id` in practice; used as `listing_id`. |
| `platform_id`     | string | **Join key** to image paths on S3/local. |
| `scrape_source`   | string | `COMPARIS` / `ROBINREAL` / `SRED`. Lowercased in image paths. |
| `platform_url`    | string | Canonical source-site URL. |
| `geo_lat`,`geo_lng` | float | `'` and `,` are sometimes used as separators — parser handles it. |
| `object_street`, `object_zip`, `object_city`, `object_state` | strings | Empty for SRED. |
| `location_address` | JSON blob | Nested {PostalCode, City, Street, StreetNumber, canton, Country}. Used as fallback when flat columns are empty. |

Listing facts:

| Column             | Notes |
|--------------------|---|
| `title`            | Human-readable headline. |
| `object_description`, `remarks` | Long-form German/French/Italian text — main signal for soft ranking. |
| `price`, `rent_net`, `rent_extra`, `rent_gross`, `price_type` | Price parser prefers `rent_gross` → `price` → `rent_net + rent_extra`. |
| `number_of_rooms`, `area` | Floats. |
| `available_from`   | Date; frequently missing. |
| `object_category`, `object_type`, `object_type_text` | E.g. `Wohnung`, `Gewerbeobjekt`, `Parkplatz`, `Haus`. Often null (53%). |
| `offer_type`       | `RENT` / `SALE`. |

Distance features (integers, meters; mostly COMPARIS-only):

| Column | Meaning |
|---|---|
| `distance_public_transport` | Walking distance to nearest transit stop |
| `distance_shop`             | Nearest supermarket |
| `distance_kindergarten`     | Nearest kindergarten |
| `distance_school_1`, `distance_school_2` | Primary / secondary school |

Boolean-ish feature flags (`true`/`false`/empty):

`prop_balcony`, `prop_elevator`, `prop_parking`, `prop_garage`, `prop_fireplace`, `prop_child_friendly`, `animal_allowed`, `maybe_temporary`, `is_new_building`.

Additional ground-truth features come from parsing `orig_data.Features` and `orig_data.MainData` — see `app/participant/listing_row_parser.py`.

Image references:

| Column   | Shape |
|----------|---|
| `images` | JSON: `{"images": [{"url": "...", "filename": "..."}]}` — for SRED this is a local path like `/raw-data-images/<id>.jpeg`; for others it's a CDN URL on `assets-comparis.b-cdn.net` or `assets.comparis.ch`. |

## SQLite schema (after harness import)

The harness flattens the 52 raw columns into a more usable `listings` table. Full DDL in `app/harness/csv_import.py`. Highlights:

```sql
CREATE TABLE listings (
    listing_id                     TEXT PRIMARY KEY,
    platform_id                    TEXT,
    scrape_source                  TEXT,
    title                          TEXT NOT NULL,
    description                    TEXT,
    street, city, postal_code, canton  TEXT,
    price                          INTEGER,   -- CHF
    rooms, area, latitude, longitude  REAL,
    available_from                 TEXT,      -- ISO date
    distance_public_transport      INTEGER,
    distance_shop                  INTEGER,
    distance_kindergarten          INTEGER,
    distance_school_1              INTEGER,
    distance_school_2              INTEGER,
    -- derived boolean flags (1 / 0 / NULL)
    feature_balcony                INTEGER,
    feature_elevator               INTEGER,
    feature_parking                INTEGER,
    feature_garage                 INTEGER,
    feature_fireplace              INTEGER,
    feature_child_friendly         INTEGER,
    feature_pets_allowed           INTEGER,
    feature_temporary              INTEGER,
    feature_new_build              INTEGER,
    feature_wheelchair_accessible  INTEGER,
    feature_private_laundry        INTEGER,
    feature_minergie_certified     INTEGER,
    features_json                  TEXT NOT NULL,   -- list of enabled feature names
    offer_type                     TEXT,    -- RENT / SALE
    object_category, object_type   TEXT,
    original_url                   TEXT,
    images_json                    TEXT,    -- parsed images blob
    location_address_json          TEXT,
    orig_data_json                 TEXT,
    raw_json                       TEXT NOT NULL
);
-- Indexes on: city, postal_code, canton, price, rooms, latitude, longitude
```

## Linking images → listings

Image paths are self-describing. The convention is:

```
downloads/prod/<scrape_source_lowercased>/images/platform_id=<platform_id>/<idx>-<image_id>.jpg
```

Example round-trip:

```python
import sqlite3
from pathlib import Path

con = sqlite3.connect("/data/listings.db")   # or "data/listings.db" on host
con.row_factory = sqlite3.Row

row = con.execute("""
    SELECT listing_id, platform_id, scrape_source, title, city, price
    FROM listings WHERE city = 'Zurich' AND price < 2800 AND rooms >= 3
    LIMIT 1
""").fetchone()

src = row["scrape_source"].lower()
pid = row["platform_id"]

# SRED montages live in a different place (one image per listing)
if src == "sred":
    images = [Path("raw_data/sred_images") / f"{row['listing_id']}.jpeg"]
else:
    folder = Path(f"downloads/prod/{src}/images/platform_id={pid}")
    images = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []

print(row["title"], "→", len(images), "images")
```

Image count per listing: **~6–7 images median**, up to 28 max on comparis and 13 max on robinreal.

### Quirk: `__getdirect` files

797 files under comparis have no extension, just a `__getdirect` suffix (`file` reports them as JPEG data). `app/core/s3.py` filters by `(.jpg|.jpeg|.png|.webp)`, so those get silently skipped when generating S3 URLs. If you iterate the local filesystem for embeddings, **include them**:

```python
is_image = p.is_file() and (p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}
                             or p.name.endswith("__getdirect"))
```

## Quick-look commands

```bash
# Summary numbers
docker compose exec -T api python -c "
import sqlite3
c = sqlite3.connect('/data/listings.db')
for r in c.execute('SELECT scrape_source, COUNT(*) FROM listings GROUP BY scrape_source'):
    print(r)
"

# Peek at one COMPARIS row
sqlite3 data/listings.db "SELECT listing_id,title,city,price,rooms FROM listings WHERE scrape_source='COMPARIS' LIMIT 3;"

# Cities with most listings
sqlite3 data/listings.db "
SELECT city, COUNT(*) AS n FROM listings
WHERE city IS NOT NULL GROUP BY city ORDER BY n DESC LIMIT 10;
"

# Price / room distribution by source
sqlite3 data/listings.db "
SELECT scrape_source,
       AVG(price) AS avg_price,
       MIN(price) AS min_price,
       MAX(price) AS max_price,
       AVG(rooms) AS avg_rooms
FROM listings WHERE price IS NOT NULL
GROUP BY scrape_source;
"

# Feature coverage
sqlite3 data/listings.db "
SELECT
  SUM(feature_balcony=1) AS balcony,
  SUM(feature_elevator=1) AS elevator,
  SUM(feature_parking=1) AS parking,
  SUM(feature_pets_allowed=1) AS pets,
  SUM(feature_new_build=1) AS new_build,
  COUNT(*) AS total
FROM listings;
"

# How many images per listing on average?
for src in robinreal comparis; do
  echo "-- $src --"
  find downloads/prod/$src/images -maxdepth 1 -mindepth 1 -type d \
    | while read d; do ls -1 "$d" | wc -l; done \
    | awk '{s+=$1; c++} END{printf "avg=%.1f over %d listings (total %d)\n", s/c, c, s}'
done
```

## Known pitfalls & things to handle

1. **SRED has no city / postal / canton / street.** 48.7% of DB rows have `city IS NULL` — all of them are SRED. Either reverse-geocode lat/lng (Nominatim works offline via Docker) or encode city filters to also accept SRED listings whose coordinates fall in the city's bbox/polygon.
2. **`available_from` is 70% missing.** Date filters will have low recall. Prefer to treat availability as a soft signal unless the user explicitly hard-requires a move-in date.
3. **`canton` is 65% missing** in raw data. It's uppercase when present (e.g. `ZH`, `VD`, `GE`).
4. **Multilingual text.** Descriptions are German, French, Italian (Switzerland). If you're using sentence embeddings, pick a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`, `bge-m3`, or similar).
5. **Price semantics.** `price` is already CHF, usually monthly rent gross. But `rent_net + rent_extra` sometimes differs from `rent_gross`. Trust the harness parser (`_derive_price` in `listing_row_parser.py`) unless you have a reason not to.
6. **Image URLs in `images_json`** point at CDNs (`assets.comparis.ch`, `assets-comparis.b-cdn.net`) — often re-encoded / watermarked versions. The files under `downloads/prod/` are the *original* uploads. Use local files for embedding work.
7. **Bootstrap is skipped if DB file exists.** After editing `listing_row_parser.py` or adding columns, rebuild:
   ```bash
   docker compose down
   docker volume rm robinreal_datathon_challenge_2026_listings_data
   docker compose up -d api
   ```
8. **Evaluation set is not in S3.** The challenge mentions queries + hard-criteria matches + a soft ranking over 20 candidates — that'll be delivered separately by the organizers.

## Data provenance

- `comparis` = Comparis.ch (Swiss real-estate aggregator). CSV `structured_data_*` + S3 `prod/comparis/images/`.
- `robinreal` = Robinreal.ai (product we're building this for). CSV `robinreal_data_*` + S3 `prod/robinreal/images/`.
- `sred` = SRED dataset with pre-rendered photo montages (one composite image per listing). CSV `sred_data_withmontageimages_latlong.csv` + local `raw_data/sred_images/`.
