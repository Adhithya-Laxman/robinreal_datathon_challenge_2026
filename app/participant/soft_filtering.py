from __future__ import annotations

import re
from typing import Any


# ── Listing-type pre-filters ───────────────────────────────────────────────

# Exchange / Tauschwohnung: owner wants to swap apartments, not rent out.
_EXCHANGE_RE = re.compile(
    r"\btausch(?:wohnung)?\b"
    r"|\bzum\s+tausch\b"
    r"|\[tauschwohnung\]"
    r"|\bwohnung(?:en)?\s+tauschen\b"
    r"|\bswap\s+apartment\b",
    re.IGNORECASE,
)

# Wanted-ad signals: the listing is posted by someone SEEKING housing, not offering it.
# Matched only against the title to avoid false positives in descriptions.
_WANTED_TITLE_RE = re.compile(
    # French "cherche appartement / sous-location / reprise de bail"
    r"(?:^|\b)cherche\s+(?:sous-?location|appartement|logement|reprise|bail|studio|maison)"
    # German "Wohnung gesucht" (apartment wanted — not "Nachmieter gesucht" which is an offer)
    r"|\bwohnung\s+gesucht\b"
    # Italian wanted ads in title
    r"|\bcerco\s+(?:appartamento|monolocale|stanza|camera)\b",
    re.IGNORECASE,
)

# Coliving: intentional shared-living product with shared common areas.
_COLIVING_RE = re.compile(r"\bco-?living\b", re.IGNORECASE)

# Temporary / fixed-term lease markers in listing title or description.
_TEMPORARY_LISTING_RE = re.compile(
    r"\bbefristet\b"
    r"|\btemporäre?\b"
    r"|\btemporaire\b"
    r"|\btemporaneo\b"
    r"|\bfixed.term\b"
    r"|\bkurzfristig\b"
    r"|\bbis\s+\d{2}\.\d{2}\.\d{4}\b",   # "bis 30.06.2026"
    re.IGNORECASE,
)

# User signals that they explicitly want temporary housing.
_WANTS_TEMPORARY_RE = re.compile(
    r"\b\d+\s*(?:monat(?:e|en)?|month[s]?|mois|mes[ei])\b"
    r"|\bvorübergehend\b"
    r"|\btemporär\b"
    r"|\btemporairement\b"
    r"|\btemporaneamente\b"
    r"|\bkurzfristig\b"
    r"|\bauf\s+zeit\b"
    r"|\bfor\s+a\s+(?:few|couple|short)\b",
    re.IGNORECASE,
)

# Explicit single-room WG (shared flat room) ads.
_WG_ROOM_RE = re.compile(
    r"\bwg[-\s]?zimmer\b"
    r"|\bzimmer\s+in\s+(?:einer\s+)?wg\b"
    r"|\bcamera\s+in\s+appartamento\s+condiviso\b"
    r"|\broom\s+in\s+shared\s+(?:flat|apartment)\b"
    r"|\bchambre\s+en\s+colocation\b",
    re.IGNORECASE,
)


def _text(listing: dict[str, Any]) -> str:
    return f"{listing.get('title', '')} {listing.get('description', '') or ''}"


def _is_exchange_listing(listing: dict[str, Any]) -> bool:
    return bool(_EXCHANGE_RE.search(_text(listing)))


def _is_wanted_ad(listing: dict[str, Any]) -> bool:
    return bool(_WANTED_TITLE_RE.search(listing.get("title", "")))


def _is_coliving(listing: dict[str, Any]) -> bool:
    return bool(_COLIVING_RE.search(_text(listing)))


def _is_wg_room_ad(listing: dict[str, Any]) -> bool:
    return bool(_WG_ROOM_RE.search(_text(listing)))


def _is_temporary_listing(listing: dict[str, Any]) -> bool:
    return bool(_TEMPORARY_LISTING_RE.search(_text(listing)))


def _user_wants_temporary(soft_facts: dict[str, Any]) -> bool:
    """True when the query explicitly requests short-term / temporary housing."""
    raw = soft_facts.get("raw_query", "")
    return bool(_WANTS_TEMPORARY_RE.search(raw))


def _filter_by_tenancy_type(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    if _user_wants_temporary(soft_facts):
        return candidates
    filtered = [c for c in candidates if not _is_temporary_listing(c)]
    return filtered if filtered else candidates


def _user_wants_shared(soft_facts: dict[str, Any]) -> bool:
    """True when the query suggests shared housing (WG / coliving) is acceptable."""
    soft = soft_facts.get("soft", {})
    if soft.get("student_friendly"):
        return True
    raw = soft_facts.get("raw_query", "").lower()
    return any(kw in raw for kw in ("wg", "coliving", "co-living", "shared flat", "colocation"))


def _filter_by_listing_type(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    wants_shared = _user_wants_shared(soft_facts)
    filtered = [
        c for c in candidates
        if not _is_exchange_listing(c)
        and not _is_wanted_ad(c)
        and (wants_shared or not _is_coliving(c))
        and (wants_shared or not _is_wg_room_ad(c))
    ]
    # Safety: never collapse to empty — return originals if all were rejected.
    return filtered if filtered else candidates


# ── Size (m²) constraint enforcement ──────────────────────────────────────

# Fraction of stated min_area below which a listing is excluded even after
# hard-filter relaxation.  0.80 = 80 %, so a 39 m² flat is rejected when
# the user stated ≥ 70 m² (39 / 70 = 56 % < 80 %).
# Tightened: 97% catches "92 m² returned for 95 m² minimum" (96.8% > 80% was passing through).
_AREA_THRESHOLD = 0.97


def _filter_by_area(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    understanding = soft_facts.get("understanding")
    if understanding is None:
        return candidates
    min_area: float | None = getattr(understanding.hard, "min_area", None)
    if not min_area:
        return candidates
    cutoff = min_area * _AREA_THRESHOLD
    filtered = [c for c in candidates if (c.get("area") or 0) >= cutoff]
    return filtered if filtered else candidates


# ── Public entry point ─────────────────────────────────────────────────────


def filter_soft_facts(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = _filter_by_listing_type(candidates, soft_facts)
    candidates = _filter_by_tenancy_type(candidates, soft_facts)
    candidates = _filter_by_area(candidates, soft_facts)
    return candidates
