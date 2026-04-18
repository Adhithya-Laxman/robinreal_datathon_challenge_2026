from __future__ import annotations

import json
import os

import anthropic

from app.models.schemas import HardFilters

_SYSTEM_PROMPT = """You are a real estate search assistant for Swiss rental listings.
Extract structured search filters from the user's natural language query.
The dataset contains rentals in Switzerland, mostly in German-speaking cantons.
All listings are RENT type.

Available features: balcony, elevator, parking, garage, fireplace, child_friendly,
pets_allowed, temporary, new_build, wheelchair_accessible, private_laundry, minergie_certified

Canton codes: ZH, BE, BL, AG, SG, BS, SO, TI, FR, LU, GE, VD, VS, GR, TG, SH, AR, AI, GL, NW, OW, UR, SZ, ZG, NE, JU

Object categories: Wohnung, Haus, Parkplatz, Gewerbeobjekt

Return ONLY a JSON object with these optional fields (omit fields not mentioned):
- city: list of city names (e.g. ["Zürich", "Bern"])
- canton: single canton code (e.g. "ZH")
- postal_code: list of postal codes
- min_price: minimum monthly rent in CHF
- max_price: maximum monthly rent in CHF
- min_rooms: minimum number of rooms
- max_rooms: maximum number of rooms
- features: list of required features from the available list
- object_category: list of property types
- sort_by: one of price_asc, price_desc, rooms_asc, rooms_desc
"""


def extract_hard_facts(query: str) -> HardFilters:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return HardFilters()

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Extract search filters from this query: {query}",
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return HardFilters()

    return HardFilters(
        city=data.get("city"),
        canton=data.get("canton"),
        postal_code=data.get("postal_code"),
        min_price=data.get("min_price"),
        max_price=data.get("max_price"),
        min_rooms=data.get("min_rooms"),
        max_rooms=data.get("max_rooms"),
        features=data.get("features"),
        object_category=data.get("object_category"),
        sort_by=data.get("sort_by"),
    )
