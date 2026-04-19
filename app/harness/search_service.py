from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.hard_filters import HardFilterParams, search_listings
from app.models.schemas import HardFilters, ListingsResponse
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


_MIN_RESULTS = 5

# Fields we will NEVER relax. If the user/LLM asked for a specific location,
# a smaller result set from the correct place beats a bigger result set
# from the wrong place. Showing "Dübendorf" for a "Zürich" query is worse
# than showing only 2 listings.
_INVIOLABLE_FIELDS: frozenset[str] = frozenset({
    "city",
    "canton",
    "postal_code",
    "latitude",
    "longitude",
    "radius_km",
    "offer_type",         # rent vs buy — dropping this would be absurd
    "object_category",    # Haus vs Wohnung — same reasoning
})


def filter_hard_facts(
    db_path: Path,
    hard_facts: HardFilters,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply hard filters, relaxing only non-location / non-category fields.

    Returns
    -------
    results : list[dict]
        Matching listing rows (possibly fewer than `_MIN_RESULTS`).
    relaxed : list[str]
        Names of fields that were relaxed in order to reach `_MIN_RESULTS`.
        Empty list if the strict filter already produced enough results.
        Never contains any field from `_INVIOLABLE_FIELDS`.
    """
    results = search_listings(db_path, to_hard_filter_params(hard_facts))
    if len(results) >= _MIN_RESULTS:
        return results, []

    relaxed_fields: list[str] = []
    relaxed = hard_facts.model_copy()

    # Ordered from "most comfortable to drop" to "last resort".
    # LOCATION fields (city, canton, postal_code, lat/lng, radius_km),
    # offer_type, and object_category are intentionally absent — they
    # are in _INVIOLABLE_FIELDS.
    relaxations: list[tuple[str, Any]] = [
        ("max_area", None),
        ("min_area", None),
        ("features", None),
        ("max_price", lambda v: int(v * 1.2) if v else None),
        ("min_price", lambda v: int(v * 0.8) if v else None),
        ("max_rooms", lambda v: v + 0.5 if v else None),
        ("min_rooms", lambda v: max(0, v - 0.5) if v else None),
        ("max_floor", None),
        ("min_floor", None),
    ]

    for field, transform in relaxations:
        if field in _INVIOLABLE_FIELDS:   # belt + suspenders
            continue
        current = getattr(relaxed, field, None)
        if current is None:
            continue
        new_value = transform(current) if callable(transform) else transform
        setattr(relaxed, field, new_value)
        relaxed_fields.append(field)
        results = search_listings(db_path, to_hard_filter_params(relaxed))
        if len(results) >= _MIN_RESULTS:
            return results, relaxed_fields

    return results, relaxed_fields


def query_from_text(
    *,
    db_path: Path,
    query: str,
    limit: int,
    offset: int,
) -> ListingsResponse:
    hard_facts = extract_hard_facts(query)
    hard_facts.limit = limit
    hard_facts.offset = offset
    soft_facts = extract_soft_facts(query)
    candidates, _relaxed = filter_hard_facts(db_path, hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(candidates, soft_facts),
        meta={},
    )


def query_from_filters(
    *,
    db_path: Path,
    hard_facts: HardFilters | None,
) -> ListingsResponse:
    structured_hard_facts = hard_facts or HardFilters()
    soft_facts = extract_soft_facts("")
    candidates, _relaxed = filter_hard_facts(db_path, structured_hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(candidates, soft_facts),
        meta={},
    )


def to_hard_filter_params(hard_facts: HardFilters) -> HardFilterParams:
    return HardFilterParams(
        city=hard_facts.city,
        postal_code=hard_facts.postal_code,
        canton=hard_facts.canton,
        min_price=hard_facts.min_price,
        max_price=hard_facts.max_price,
        min_rooms=hard_facts.min_rooms,
        max_rooms=hard_facts.max_rooms,
        min_area=hard_facts.min_area,
        max_area=hard_facts.max_area,
        min_floor=hard_facts.min_floor,
        max_floor=hard_facts.max_floor,
        latitude=hard_facts.latitude,
        longitude=hard_facts.longitude,
        radius_km=hard_facts.radius_km,
        features=hard_facts.features,
        offer_type=hard_facts.offer_type,
        object_category=hard_facts.object_category,
        include_exchange=getattr(hard_facts, "include_exchange", False),
        include_sublet=getattr(hard_facts, "include_sublet", False),
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )
