# `app/geo` — Geo Enrichment Module

Owned by: **Mattia**
Run once before serving: `uv run python scripts/geo_enrich.py`

---

## What was enriched

The raw database had two geo problems:

| Problem | Affected rows | Fix |
|---|---|---|
| `city` / `canton` = NULL | 11,105 SRED listings (48.7% of DB) | Reverse-geocoded from lat/lng → city name + canton abbreviation |
| No POI distances | All 22,819 listings | Computed nearest transit / supermarket / school / university in meters |

After running the script:
- **11,024 SRED listings** have `city` and `canton` filled in (81 remain NULL — coordinates outside CH borders)
- **21,182 listings** have all 4 POI distance columns filled in (the rest have no lat/lng at all)

---

## New DB columns

Four `INTEGER` columns were added to the `listings` table:

| Column | Meaning |
|---|---|
| `geo_transit_m` | Distance in meters to nearest bus stop / tram stop / train station |
| `geo_supermarket_m` | Distance in meters to nearest supermarket or convenience store |
| `geo_school_m` | Distance in meters to nearest primary or secondary school |
| `geo_university_m` | Distance in meters to nearest university (any Swiss university) |

All values are straight-line (great-circle) distances. NULL means the listing had no coordinates.

Typical Swiss ranges for sanity-checking:
- `geo_transit_m`: 30–800 m in cities, up to 3 km in rural areas
- `geo_supermarket_m`: 50–1500 m
- `geo_school_m`: 100–2000 m
- `geo_university_m`: 200 m (city centre) – 80 km (remote areas)

---

## Python API for ranking (`app/participant/ranking.py`)

Import from `app.geo`:

```python
from app.geo import haversine_m, nearest_university, UNIVERSITY_ANCHORS
```

### `haversine_m(lat1, lng1, lat2, lng2) -> float`

Great-circle distance in **meters** between two WGS84 points.
Use this at query time to score a listing's distance to a user-specified anchor.

```python
# "I want something close to ETH Zurich"
eth_lat, eth_lng = UNIVERSITY_ANCHORS["ETH Zurich"]
dist_m = haversine_m(listing["latitude"], listing["longitude"], eth_lat, eth_lng)

# Normalise to [0, 1] score — closer = higher score
# Example: full score within 500 m, zero score beyond 5 km
s_geo = max(0.0, 1.0 - dist_m / 5000)
```

---

### `UNIVERSITY_ANCHORS: dict[str, tuple[float, float]]`

Hardcoded lat/lng for 13 Swiss universities. Use these as named anchors when the
query understanding step emits something like `anchors: [{"text": "ETH Zurich", ...}]`.

```python
UNIVERSITY_ANCHORS = {
    "ETH Zurich":            (47.3763, 8.5483),
    "EPFL Lausanne":         (46.5191, 6.5668),
    "University of Zurich":  (47.3743, 8.5490),
    "University of Bern":    (46.9499, 7.4380),
    "University of Basel":   (47.5596, 7.5806),
    "University of Geneva":  (46.2044, 6.1432),
    "ETH Hönggerberg":       (47.4065, 8.5072),
    "HSG St. Gallen":        (47.4310, 9.3800),
    "ZHAW Winterthur":       (47.4900, 8.7227),
    "Université de Lausanne":(46.5226, 6.5818),
    "University of Fribourg":(46.8065, 7.1571),
    "University of Neuchâtel":(47.0020, 6.9414),
    "USI Lugano":            (46.0037, 8.9511),
}
```

---

### `nearest_university(lat, lng) -> (name, dist_m)`

Returns the name and distance to the closest university anchor.
Useful for building explanations ("3 min walk from ETH Zurich").

```python
name, dist_m = nearest_university(listing["latitude"], listing["longitude"])
# → ("ETH Zurich", 420.0)
```

---

## Using precomputed POI columns in scoring

The `geo_transit_m` etc. columns are already in every listing dict returned by
`search_listings()`. Use them directly — no extra DB query needed.

```python
def score_geo(listing: dict, soft_prefs: dict) -> float:
    """Example s_geo signal — adapt weights to your query understanding output."""
    scores = []

    # 1. Transit proximity (always useful)
    transit_m = listing.get("geo_transit_m")
    if transit_m is not None:
        scores.append(max(0.0, 1.0 - transit_m / 1000))

    # 2. University anchor from query (if present)
    for anchor in soft_prefs.get("anchors", []):
        anchor_lat = anchor.get("lat")
        anchor_lng = anchor.get("lng")
        if anchor_lat and listing.get("latitude"):
            d = haversine_m(listing["latitude"], listing["longitude"],
                            anchor_lat, anchor_lng)
            scores.append(max(0.0, 1.0 - d / 3000))

    # 3. Family signal: school proximity
    if soft_prefs.get("family_friendly"):
        school_m = listing.get("geo_school_m")
        if school_m is not None:
            scores.append(max(0.0, 1.0 - school_m / 1500))

    return sum(scores) / len(scores) if scores else 0.5
```

> **Note:** The listing dicts from `search_listings()` currently don't expose the new
> `geo_*` columns — you may need to add them to the SELECT in `app/core/hard_filters.py`
> or read them in a second pass. Check with whoever owns that file.

---

## Re-running the script

The script is fully **idempotent** — safe to run multiple times.

```bash
# Full run (skips already-computed rows automatically)
uv run python scripts/geo_enrich.py

# Only POI distances (e.g. after DB was rebuilt from scratch)
uv run python scripts/geo_enrich.py --skip-reverse-geocode

# Only reverse geocode (city/canton)
uv run python scripts/geo_enrich.py --skip-poi

# Custom paths
uv run python scripts/geo_enrich.py --db data/listings.db --geo-cache data/geo
```

Overpass API responses are cached to `data/geo/overpass_*.json` — once those files
exist, re-runs need no internet connection for the POI step.

---

## File overview

```
app/geo/
  __init__.py          haversine_m(), nearest_university(), UNIVERSITY_ANCHORS
  reverse_geocode.py   reverse_geocode_batch() — offline GeoNames lookup
  poi.py               fetch_swiss_pois(), build_poi_trees(), compute_poi_distances()
  migrate.py           add_geo_columns() — idempotent ALTER TABLE

scripts/
  geo_enrich.py        CLI runner — call this offline before serving
```
