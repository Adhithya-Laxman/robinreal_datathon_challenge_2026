from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query, Request

from app.config import get_settings
from app.geo.poi import find_nearest_pois, load_poi_elements
from app.harness.search_service import query_from_filters, query_from_text
from app.models.schemas import (
    HealthResponse,
    ListingData,
    ListingsQueryRequest,
    ListingsResponse,
    ListingsSearchRequest,
    RankedListingResult,
)
from app.participant.unified_ranker import unified_search

router = APIRouter()

_GEO_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "geo"
_VALID_POI_TYPES = {"transit", "supermarket", "school", "university"}


@lru_cache(maxsize=None)
def _get_poi_elements(poi_type: str) -> list[dict]:
    return load_poi_elements(_GEO_CACHE_DIR, poi_type)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/listings", response_model=ListingsResponse)
def listings(body: ListingsQueryRequest, request: Request) -> ListingsResponse:
    session_id = getattr(request.state, "session_id", None)
    resp = unified_search(
        body.query,
        top_k=body.limit,
        use_vlm=True,
        session_id=session_id,
    )
    results = []
    for r in resp.results[body.offset:]:
        row = r.row
        top_signals = sorted(r.signals.items(), key=lambda t: -t[1] * r.weights.get(t[0], 0))[:3]
        reason = ", ".join(f"{k}={v:.2f}" for k, v in top_signals) or "unified score"
        results.append(RankedListingResult(
            listing_id=r.listing_id,
            score=r.score,
            reason=reason,
            signals={k: round(v, 4) for k, v in r.signals.items() if v > 0},
            listing=ListingData(
                id=r.listing_id,
                title=row.get("title") or "",
                description=row.get("description"),
                street=row.get("street"),
                city=row.get("city"),
                postal_code=row.get("postal_code"),
                canton=row.get("canton"),
                latitude=row.get("latitude"),
                longitude=row.get("longitude"),
                price_chf=row.get("price"),
                rooms=row.get("rooms"),
                living_area_sqm=row.get("area"),
                available_from=row.get("available_from"),
                image_urls=row.get("image_urls") or [],
                hero_image_url=row.get("hero_image_url"),
                original_listing_url=row.get("original_url"),
                features=row.get("features") or [],
                offer_type=row.get("offer_type"),
                object_category=row.get("object_category"),
                object_type=row.get("object_type"),
                geo_transit_m=row.get("geo_transit_m"),
                geo_supermarket_m=row.get("geo_supermarket_m"),
                geo_school_m=row.get("geo_school_m"),
                geo_university_m=row.get("geo_university_m"),
            ),
        ))
    return ListingsResponse(
        listings=results,
        meta={
            "pipeline": "unified",
            "feedback_applied": resp.feedback_applied,
        },
    )


@router.get("/poi/nearby", response_model=dict)
def poi_nearby(
    lat: float = Query(..., description="Latitude of the query location"),
    lng: float = Query(..., description="Longitude of the query location"),
    poi_type: str = Query("transit", description="One of: transit, supermarket, school, university"),
    k: int = Query(5, ge=1, le=20, description="Number of POIs to return"),
    max_radius_m: float = Query(2000.0, ge=0, le=10000, description="Maximum search radius in metres"),
) -> dict:
    if poi_type not in _VALID_POI_TYPES:
        return {"error": f"Unknown poi_type '{poi_type}'. Valid: {sorted(_VALID_POI_TYPES)}", "pois": []}
    elements = _get_poi_elements(poi_type)
    pois = find_nearest_pois(lat, lng, elements, k=k, max_radius_m=max_radius_m)
    return {
        "poi_type": poi_type,
        "queried_location": {"latitude": lat, "longitude": lng},
        "pois": pois,
    }


@router.post("/listings/search/filter", response_model=ListingsResponse)
def listings_search(request: ListingsSearchRequest) -> ListingsResponse:
    settings = get_settings()
    return query_from_filters(
        db_path=settings.db_path,
        hard_facts=request.hard_filters,
    )
