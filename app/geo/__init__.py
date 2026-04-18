"""
Geo utilities for the listings pipeline.

Public API used by ranking (app/participant/ranking.py):
    haversine_m(lat1, lng1, lat2, lng2) -> float   # meters
    nearest_university(lat, lng) -> (name, dist_m)
    UNIVERSITY_ANCHORS: dict[str, (lat, lng)]
"""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0

# Hardcoded Swiss university anchors (WGS84).
# Used at query-time by ranking to score proximity to named institutions.
UNIVERSITY_ANCHORS: dict[str, tuple[float, float]] = {
    "ETH Zurich": (47.3763, 8.5483),
    "EPFL Lausanne": (46.5191, 6.5668),
    "University of Zurich": (47.3743, 8.5490),
    "University of Bern": (46.9499, 7.4380),
    "University of Basel": (47.5596, 7.5806),
    "University of Geneva": (46.2044, 6.1432),
    "ETH Hönggerberg": (47.4065, 8.5072),
    "HSG St. Gallen": (47.4310, 9.3800),
    "ZHAW Winterthur": (47.4900, 8.7227),
    "Université de Lausanne": (46.5226, 6.5818),
    "University of Fribourg": (46.8065, 7.1571),
    "University of Neuchâtel": (47.0020, 6.9414),
    "USI Lugano": (46.0037, 8.9511),
}


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def nearest_university(lat: float, lng: float) -> tuple[str, float]:
    """Return (name, distance_m) of the nearest university anchor."""
    best_name, best_dist = "", float("inf")
    for name, (ulat, ulng) in UNIVERSITY_ANCHORS.items():
        d = haversine_m(lat, lng, ulat, ulng)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, best_dist
