from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class ListingsApiClient:
    base_url: str

    async def search_listings(
        self,
        *,
        query: str,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            response = await client.post(
                "/listings",
                json={"query": query, "limit": limit, "offset": offset},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("listings"), list):
                raise ValueError("Listings API returned an invalid listings wrapper payload.")
            return payload

    async def get_nearby_pois(
        self,
        *,
        lat: float,
        lng: float,
        poi_type: str = "transit",
        k: int = 5,
        max_radius_m: float = 2000.0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.get(
                "/poi/nearby",
                params={"lat": lat, "lng": lng, "poi_type": poi_type, "k": k, "max_radius_m": max_radius_m},
            )
            response.raise_for_status()
            return response.json()


def get_listings_api_client() -> ListingsApiClient:
    return ListingsApiClient(
        base_url=os.getenv("APPS_SDK_LISTINGS_API_BASE_URL", "http://localhost:8000")
    )
