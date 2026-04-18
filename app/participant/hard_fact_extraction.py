from __future__ import annotations

import json
import os

import anthropic

from app.models.schemas import HardFilters

_FEATURE_MAP: dict[str, str] = {
    # balcony
    "balcony": "balcony", "balkon": "balcony", "balcon": "balcony", "terrasse": "balcony",
    "balcone": "balcony", "terrazza": "balcony",
    # elevator
    "elevator": "elevator", "aufzug": "elevator", "lift": "elevator", "ascenseur": "elevator",
    "ascensore": "elevator",
    # parking
    "parking": "parking", "parkplatz": "parking", "aussenparkplatz": "parking",
    "parking extérieur": "parking", "parcheggio": "parking", "posto auto": "parking",
    # garage
    "garage": "garage", "tiefgarage": "garage", "garage souterrain": "garage",
    "garage sotterraneo": "garage", "autorimessa": "garage",
    # fireplace
    "fireplace": "fireplace", "kamin": "fireplace", "cheminée": "fireplace",
    "camino": "fireplace", "caminetto": "fireplace",
    # child_friendly
    "child_friendly": "child_friendly", "kinderfreundlich": "child_friendly",
    "familienfreundlich": "child_friendly", "adapté aux enfants": "child_friendly",
    "adatto ai bambini": "child_friendly", "per famiglie": "child_friendly",
    # pets_allowed
    "pets_allowed": "pets_allowed", "haustiere erlaubt": "pets_allowed",
    "tiere erlaubt": "pets_allowed", "animaux admis": "pets_allowed",
    "animali ammessi": "pets_allowed", "animali consentiti": "pets_allowed",
    # new_build
    "new_build": "new_build", "neubau": "new_build", "erstbezug": "new_build",
    "construction neuve": "new_build", "nuova costruzione": "new_build", "prima occupazione": "new_build",
    # wheelchair_accessible
    "wheelchair_accessible": "wheelchair_accessible", "rollstuhlgängig": "wheelchair_accessible",
    "barrierefrei": "wheelchair_accessible", "accès handicapés": "wheelchair_accessible",
    "accessibile in sedia a rotelle": "wheelchair_accessible", "senza barriere": "wheelchair_accessible",
    # private_laundry
    "private_laundry": "private_laundry", "waschmaschine": "private_laundry",
    "tumbler": "private_laundry", "machine à laver privée": "private_laundry",
    "lavatrice privata": "private_laundry", "lavatrice": "private_laundry",
    # minergie_certified
    "minergie_certified": "minergie_certified", "minergie": "minergie_certified",
    "certifié minergie": "minergie_certified", "certificato minergie": "minergie_certified",
    # temporary
    "temporary": "temporary", "temporär": "temporary", "befristet": "temporary",
    "temporaire": "temporary", "temporaneo": "temporary", "a termine": "temporary",
    # furnished
    "furnished": "furnished", "möbliert": "furnished", "meublé": "furnished", "arredato": "furnished",
    # unfurnished
    "unfurnished": "unfurnished", "unmöbliert": "unfurnished", "non meublé": "unfurnished",
    "non arredato": "unfurnished",
}


def _normalize_features(features: list[str] | None) -> list[str] | None:
    if not features:
        return features
    normalized = []
    for f in features:
        key = _FEATURE_MAP.get(f.lower())
        if key and key not in normalized:
            normalized.append(key)
    return normalized or None


_SYSTEM_PROMPT = """You are a real estate search assistant for Swiss rental listings.
Extract structured search filters from the user's natural language query.
The query may be in English, German, French, Italian, or Swiss German — handle all equally.
The dataset contains rentals in Switzerland, mostly in German-speaking cantons.
All listings are RENT type.

CITY NAME MAPPING — always output the German name as stored in the database:
- Zurich / Zürich / Genf / Genève / Geneva → Zürich / Genf
- Berne / Bern → Bern
- Basle / Basel / Bâle → Basel
- Lausanne → Lausanne
- Lucerne / Luzern → Luzern
- St. Gallen / Saint-Gall → St. Gallen

Canton codes: ZH, BE, BL, AG, SG, BS, SO, TI, FR, LU, GE, VD, VS, GR, TG, SH, AR, AI, GL, NW, OW, UR, SZ, ZG, NE, JU
Object categories: Wohnung, Haus, Parkplatz, Gewerbeobjekt

Return ONLY a JSON object with these optional fields (omit fields not mentioned):
- city: list of city names in German (e.g. ["Zürich", "Bern"])
- canton: single canton code (e.g. "ZH")
- postal_code: list of postal codes
- min_price: minimum monthly rent in CHF
- max_price: maximum monthly rent in CHF
- min_rooms: minimum number of rooms
- max_rooms: maximum number of rooms
- features: list of required features in any language (e.g. "balcony", "Balkon", "Aufzug")
- object_category: list of property types
- min_area: minimum living area in sqm
- max_area: maximum living area in sqm
- min_floor: minimum floor number (0 = ground floor)
- max_floor: maximum floor number
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
        features=_normalize_features(data.get("features")),
        object_category=data.get("object_category"),
        min_area=data.get("min_area"),
        max_area=data.get("max_area"),
        min_floor=data.get("min_floor"),
        max_floor=data.get("max_floor"),
        sort_by=data.get("sort_by"),
    )
