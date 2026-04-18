"""
Offline reverse-geocoding for Swiss listings with missing city/canton.

Uses the `reverse_geocoder` package (GeoNames offline dataset).
First call downloads ~50 MB of GeoNames data and caches it locally.
Subsequent calls are instant (in-memory KD-tree).

GeoNames returns sub-place names for large cities (e.g. "Zuerich (Kreis 1) / Lindenhof").
_normalize_city() strips district qualifiers and maps to canonical Swiss city names so
that the result matches what COMPARIS/ROBINREAL rows store in the `city` column.
"""
from __future__ import annotations

import re

import reverse_geocoder as rg

# Canonical Swiss city names keyed by lowercase GeoNames name (after district stripping).
_CITY_CANON: dict[str, str] = {
    "zuerich": "Zürich",
    "zurich": "Zürich",
    "geneve": "Genève",
    "genf": "Genève",
    "bern": "Bern",
    "berne": "Bern",
    "basel": "Basel",
    "lausanne": "Lausanne",
    "winterthur": "Winterthur",
    "lucerne": "Luzern",
    "luzern": "Luzern",
    "lugano": "Lugano",
    "st. gallen": "St. Gallen",
    "st gallen": "St. Gallen",
    "biel": "Biel/Bienne",
    "bienne": "Biel/Bienne",
    "thun": "Thun",
    "schaffhausen": "Schaffhausen",
    "fribourg": "Fribourg",
    "freiburg": "Freiburg im Breisgau",
    "chur": "Chur",
    "neuchatel": "Neuchâtel",
    "neuchâtel": "Neuchâtel",
    "sion": "Sion",
    "aarau": "Aarau",
    "zug": "Zug",
    "solothurn": "Solothurn",
    "frauenfeld": "Frauenfeld",
    "liestal": "Liestal",
    "herisau": "Herisau",
    "appenzell": "Appenzell",
    "glarus": "Glarus",
    "altdorf": "Altdorf",
    "sarnen": "Sarnen",
    "stans": "Stans",
    "schwyz": "Schwyz",
    "bellinzona": "Bellinzona",
    "delémont": "Delémont",
    "porrentruy": "Porrentruy",
}

# Patterns that indicate GeoNames returned a sub-city district entry.
_STRIP_RE = re.compile(
    r"\s*\(Kreis\s*\d+\)"  # "(Kreis 1)"
    r"|\s*/.*$",           # " / Lindenhof"
    re.IGNORECASE,
)

# Maps GeoNames admin1 names (English + local variants) to Swiss canton abbreviations.
_CANTON_MAP: dict[str, str] = {
    "Aargau": "AG",
    "Appenzell Ausserrhoden": "AR",
    "Appenzell Inner-Rhoden": "AI",
    "Appenzell Innerrhoden": "AI",
    "Basel-Landschaft": "BL",
    "Basel-City": "BS",
    "Basel-Stadt": "BS",
    "Bern": "BE",
    "Berne": "BE",
    "Fribourg": "FR",
    "Geneva": "GE",
    "Genève": "GE",
    "Genf": "GE",
    "Glarus": "GL",
    "Graubünden": "GR",
    "Graubuenden": "GR",
    "Grischun": "GR",
    "Jura": "JU",
    "Lucerne": "LU",
    "Luzern": "LU",
    "Neuchâtel": "NE",
    "Neuchatel": "NE",
    "Nidwalden": "NW",
    "Obwalden": "OW",
    "Schaffhausen": "SH",
    "Schwyz": "SZ",
    "Solothurn": "SO",
    "St. Gallen": "SG",
    "Sankt Gallen": "SG",
    "Thurgau": "TG",
    "Ticino": "TI",
    "Uri": "UR",
    "Valais": "VS",
    "Wallis": "VS",
    "Vaud": "VD",
    "Zug": "ZG",
    "Zurich": "ZH",
    "Zürich": "ZH",
}


def _normalize_city(raw: str) -> str:
    """
    Strip GeoNames district qualifiers and return the canonical Swiss city name.

    "Zuerich (Kreis 1) / Lindenhof" -> "Zürich"
    "Innere Stadt"                   -> "Innere Stadt"  (no match, returned as-is)
    """
    stripped = _STRIP_RE.sub("", raw).strip()
    return _CITY_CANON.get(stripped.lower(), stripped)


def reverse_geocode_batch(
    coords: list[tuple[float, float]],
) -> list[dict[str, str | None]]:
    """
    Reverse-geocode a batch of (lat, lng) tuples offline.

    Returns a list of {"city": str | None, "canton": str | None}.
    Non-Swiss coordinates yield {"city": None, "canton": None}.
    """
    results = rg.search(coords, mode=1, verbose=False)
    output: list[dict[str, str | None]] = []
    for r in results:
        if r.get("cc") != "CH":
            output.append({"city": None, "canton": None})
            continue
        raw_name: str = r.get("name") or ""
        city: str | None = _normalize_city(raw_name) if raw_name else None
        canton: str | None = _CANTON_MAP.get(r.get("admin1", ""))
        output.append({"city": city, "canton": canton})
    return output
