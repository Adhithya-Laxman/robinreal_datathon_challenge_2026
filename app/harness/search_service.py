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


def filter_hard_facts(db_path: Path, hard_facts: HardFilters) -> list[dict[str, Any]]:
    results = search_listings(db_path, to_hard_filter_params(hard_facts))
    if len(results) >= _MIN_RESULTS:
        return results

    # Relax constraints one by one from weakest to strongest until we get enough results
    relaxed = hard_facts.model_copy()
    relaxations = [
        ("max_area", None),
        ("min_area", None),
        ("features", None),
        ("radius_km", lambda v: v * 2 if v else None),
        ("max_price", lambda v: int(v * 1.2) if v else None),
        ("min_price", lambda v: int(v * 0.8) if v else None),
        ("max_rooms", lambda v: v + 0.5 if v else None),
        ("min_rooms", lambda v: max(0, v - 0.5) if v else None),
        ("max_area", None),
        ("city", None),
        ("canton", None),
    ]

    for field, transform in relaxations:
        current = getattr(relaxed, field)
        if current is None:
            continue
        new_value = transform(current) if transform else None
        setattr(relaxed, field, new_value)
        results = search_listings(db_path, to_hard_filter_params(relaxed))
        if len(results) >= _MIN_RESULTS:
            return results

    return results


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
    candidates = filter_hard_facts(db_path, hard_facts)
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
    candidates = filter_hard_facts(db_path, structured_hard_facts)
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
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )
