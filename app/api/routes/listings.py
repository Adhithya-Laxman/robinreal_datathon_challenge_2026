from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
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


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/listings", response_model=ListingsResponse)
def listings(request: ListingsQueryRequest) -> ListingsResponse:
    resp = unified_search(request.query, top_k=request.limit, use_vlm=True)
    results = []
    for r in resp.results[request.offset:]:
        row = r.row
        top_signals = sorted(r.signals.items(), key=lambda t: -t[1] * r.weights.get(t[0], 0))[:3]
        reason = ", ".join(f"{k}={v:.2f}" for k, v in top_signals) or "unified score"
        results.append(RankedListingResult(
            listing_id=r.listing_id,
            score=r.score,
            reason=reason,
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
            ),
        ))
    return ListingsResponse(listings=results, meta={"pipeline": "unified"})


@router.post("/listings/search/filter", response_model=ListingsResponse)
def listings_search(request: ListingsSearchRequest) -> ListingsResponse:
    settings = get_settings()
    return query_from_filters(
        db_path=settings.db_path,
        hard_facts=request.hard_filters,
    )
