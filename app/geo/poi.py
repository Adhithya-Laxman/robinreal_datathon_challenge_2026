"""
POI (Points of Interest) distance computation for Swiss listings.

Workflow:
  1. fetch_swiss_pois()  — download from Overpass API, cache to JSON
  2. build_poi_trees()   — build sklearn BallTree (haversine) per POI type
  3. compute_poi_distances() — nearest distance in meters per listing

POI types and their DB columns:
  transit      -> geo_transit_m     (bus/tram/train stops)
  supermarket  -> geo_supermarket_m (grocery stores)
  school       -> geo_school_m      (primary & secondary schools)
  university   -> geo_university_m  (universities / ETH / EPFL)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000.0

# Switzerland bounding box [south, west, north, east] — used in every query to avoid
# slow area-lookup joins. Overpass bbox format is (south,west,north,east).
_CH_BBOX = "45.8,5.9,47.9,10.5"

# Multiple public Overpass endpoints — tried in order on timeout/5xx.
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Queries use [bbox:...] global filter — much faster than area["ISO3166-1:alpha2"="CH"].
# Transit uses highway=bus_stop + railway stations only (not stop_position — too many nodes).
_QUERIES: dict[str, str] = {
    "transit": f"""
[out:json][timeout:120][bbox:{_CH_BBOX}];
(
  node["highway"="bus_stop"];
  node["railway"="station"];
  node["railway"="halt"];
  node["amenity"="bus_station"];
  node["railway"="tram_stop"];
);
out body;
""",
    "supermarket": f"""
[out:json][timeout:120][bbox:{_CH_BBOX}];
(
  node["shop"="supermarket"];
  node["shop"="convenience"];
  node["shop"="grocery"];
  way["shop"="supermarket"];
);
out center;
""",
    "school": f"""
[out:json][timeout:120][bbox:{_CH_BBOX}];
(
  node["amenity"="school"];
  way["amenity"="school"];
);
out center;
""",
    "university": f"""
[out:json][timeout:120][bbox:{_CH_BBOX}];
(
  node["amenity"="university"];
  way["amenity"="university"];
  relation["amenity"="university"];
);
out center;
""",
}


def _extract_coords(elements: list[dict]) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for el in elements:
        if el.get("type") == "node":
            if "lat" in el and "lon" in el:
                coords.append((el["lat"], el["lon"]))
        elif "center" in el:
            c = el["center"]
            if "lat" in c and "lon" in c:
                coords.append((c["lat"], c["lon"]))
    return coords


def _fetch_overpass(poi_type: str) -> list[dict]:
    """
    POST the Overpass query for poi_type, trying each endpoint in turn.
    Retries up to 2 times per endpoint with exponential back-off on 5xx/timeout.
    Raises RuntimeError if all endpoints fail.
    """
    query = _QUERIES[poi_type]
    last_err: Exception | None = None
    for endpoint in _OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                print(
                    f"  [poi] Fetching {poi_type} from {endpoint.split('/')[2]}"
                    f" (attempt {attempt + 1})..."
                )
                with httpx.Client(timeout=240.0) as client:
                    resp = client.post(endpoint, data={"data": query})
                    resp.raise_for_status()
                    data = resp.json()
                print(f"  [poi] Got {len(data['elements'])} {poi_type} elements")
                return data["elements"]
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
                last_err = exc
                wait = 10 * (2 ** attempt)
                print(f"  [poi] {exc.__class__.__name__} — waiting {wait}s before retry")
                time.sleep(wait)
    raise RuntimeError(
        f"All Overpass endpoints failed for '{poi_type}': {last_err}"
    )


def fetch_swiss_pois(
    cache_dir: Path,
    poi_types: list[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Fetch Swiss POIs from Overpass API (cached after first call).

    Returns dict mapping poi_type -> list[(lat, lng)].
    Cached JSON files are written to cache_dir/overpass_<type>.json.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    types = poi_types or list(_QUERIES.keys())
    result: dict[str, list[tuple[float, float]]] = {}

    for poi_type in types:
        cache_path = cache_dir / f"overpass_{poi_type}.json"
        if cache_path.exists():
            print(f"  [poi] Loading cached {poi_type} POIs from {cache_path.name}")
            elements = json.loads(cache_path.read_text())["elements"]
        else:
            elements = _fetch_overpass(poi_type)
            cache_path.write_text(json.dumps({"elements": elements}))

        coords = _extract_coords(elements)
        result[poi_type] = coords
        print(f"  [poi] {len(coords)} {poi_type} POIs ready")

    return result


def build_poi_trees(
    pois: dict[str, list[tuple[float, float]]],
) -> dict[str, BallTree]:
    """Build a haversine BallTree for each POI type."""
    trees: dict[str, BallTree] = {}
    for poi_type, coords in pois.items():
        if not coords:
            print(f"  [poi] WARNING: no {poi_type} POIs — skipping tree")
            continue
        arr = np.radians(np.array(coords, dtype=float))  # (N, 2) in radians
        trees[poi_type] = BallTree(arr, metric="haversine")
    return trees


def compute_poi_distances(
    lat: float,
    lng: float,
    trees: dict[str, BallTree],
) -> dict[str, int]:
    """Return nearest distance in meters for each POI type."""
    point = np.radians([[lat, lng]])
    distances: dict[str, int] = {}
    for poi_type, tree in trees.items():
        dist_rad, _ = tree.query(point, k=1)
        distances[poi_type] = int(dist_rad[0][0] * EARTH_RADIUS_M)
    return distances


def load_poi_elements(cache_dir: Path, poi_type: str) -> list[dict]:
    """Load raw Overpass elements for poi_type from the JSON cache."""
    cache_path = cache_dir / f"overpass_{poi_type}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"POI cache not found: {cache_path}")
    return json.loads(cache_path.read_text())["elements"]


def find_nearest_pois(
    lat: float,
    lng: float,
    elements: list[dict],
    k: int = 5,
    max_radius_m: float = 2000.0,
) -> list[dict]:
    """
    Return up to k nearest POIs from elements within max_radius_m metres.
    Each result dict has: latitude, longitude, distance_m, name, type.
    """
    coords: list[tuple[float, float]] = []
    valid: list[dict] = []
    for el in elements:
        if el.get("type") == "node":
            if "lat" in el and "lon" in el:
                coords.append((el["lat"], el["lon"]))
                valid.append(el)
        elif "center" in el:
            c = el["center"]
            if "lat" in c and "lon" in c:
                coords.append((c["lat"], c["lon"]))
                valid.append(el)

    if not coords:
        return []

    arr = np.radians(np.array(coords, dtype=float))
    tree = BallTree(arr, metric="haversine")
    point = np.radians([[lat, lng]])
    actual_k = min(k, len(coords))
    dists_rad, indices = tree.query(point, k=actual_k)

    results = []
    for dist_rad, idx in zip(dists_rad[0], indices[0]):
        dist_m = int(dist_rad * EARTH_RADIUS_M)
        if dist_m > max_radius_m:
            break
        el = valid[idx]
        poi_lat, poi_lng = coords[idx]
        tags = el.get("tags", {})
        results.append({
            "latitude": poi_lat,
            "longitude": poi_lng,
            "distance_m": dist_m,
            "name": tags.get("name"),
            "type": (
                tags.get("amenity")
                or tags.get("highway")
                or tags.get("railway")
                or tags.get("shop")
            ),
        })
    return results
