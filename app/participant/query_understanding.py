"""Query understanding via Claude Sonnet 4 on Bedrock with tool use.

Single LLM call maps a natural-language query into a typed
`QueryUnderstanding` containing both hard filters and soft preferences.

Why one call instead of two:
  * Hard and soft extractions benefit from global context (e.g., "not too
    expensive" modifies price; the model sees the whole sentence either way).
  * Halves latency and spend.
  * The harness splits them via `extract_hard_facts` / `extract_soft_facts`,
    but both call into here and share a per-query cache.

If Bedrock is not configured (or the call fails), we fall back to a
rule-based extractor. The fallback is intentionally conservative — it
should never hallucinate constraints that weren't in the query.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.models.schemas import HardFilters
from app.participant.bedrock_client import bedrock_available, get_bedrock_client
from app.participant.schemas import (
    FEATURE_NAMES,
    Anchor,
    QueryUnderstanding,
    SoftPreferences,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You extract structured filters from a Swiss real-estate \
search query for a ranked-listings system.

Hard criteria are things that MUST hold for a listing to be considered at \
all (city, max price, min rooms, required features). Soft preferences are \
things that should influence ranking but must never filter listings out \
(e.g. "bright", "modern", "quiet", "close to ETH", "ideally with parking").

Rules:
- Extract only what is explicitly or strongly implied in the query.
- NEVER invent constraints. If the user didn't mention rooms, leave it unset.
- "not too expensive" / "affordable" / "cheap" are SOFT price_intent, not a \
  numeric max_price.
- "close to X", "near X", "by X" produce an anchor, NOT lat/lng/radius_km.
- "must have balcony", "with balcony" => HARD feature. \
  "ideally with parking", "nice to have a balcony" => SOFT boolean pref.
- Use the canonical feature names only (balcony, elevator, parking, garage, \
  fireplace, child_friendly, pets_allowed, temporary, new_build, \
  wheelchair_accessible, private_laundry, minergie_certified).
- Set `language` to the query's primary language.
- Provide `interpretation` in English as one short sentence the user would \
  recognize as "yes, that's what I meant".

Always respond by calling the `record_query_understanding` tool exactly once."""


def _tool_schema() -> dict[str, Any]:
    """Anthropic-on-Bedrock tool schema for the single-shot extractor."""
    return {
        "name": "record_query_understanding",
        "description": (
            "Record the structured interpretation of the user's real-estate "
            "search query. Always call this exactly once per query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["de", "fr", "it", "en", "unknown"],
                },
                "interpretation": {
                    "type": "string",
                    "description": "One-sentence English paraphrase.",
                },
                "hard": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "city": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Swiss city names, e.g. ['Zürich'].",
                        },
                        "postal_code": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "canton": {
                            "type": "string",
                            "description": "Uppercase 2-letter canton, e.g. 'ZH'.",
                        },
                        "min_price": {"type": "integer", "minimum": 0},
                        "max_price": {"type": "integer", "minimum": 0},
                        "min_rooms": {"type": "number", "minimum": 0},
                        "max_rooms": {"type": "number", "minimum": 0},
                        "features": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(FEATURE_NAMES)},
                            "description": "HARD required features only.",
                        },
                        "offer_type": {
                            "type": "string",
                            "enum": ["RENT", "SALE"],
                        },
                    },
                },
                "soft": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "descriptors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Adjectival hints, verbatim from the query, "
                                "translated to English if helpful. "
                                'e.g. ["bright", "modern kitchen", "quiet"].'
                            ),
                        },
                        "bright": {"type": "boolean"},
                        "modern": {"type": "boolean"},
                        "quiet": {"type": "boolean"},
                        "family_friendly": {"type": "boolean"},
                        "pet_friendly": {"type": "boolean"},
                        "student_friendly": {"type": "boolean"},
                        "near_public_transport": {"type": "boolean"},
                        "near_schools": {"type": "boolean"},
                        "nice_views": {"type": "boolean"},
                        "balcony_or_terrace": {"type": "boolean"},
                        "parking": {"type": "boolean"},
                        "new_build": {"type": "boolean"},
                        "furnished": {"type": "boolean"},
                        "spacious": {"type": "boolean"},
                        "anchors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "max_distance_km": {
                                        "type": "number",
                                        "minimum": 0,
                                    },
                                },
                                "required": ["text"],
                            },
                        },
                        "price_intent": {
                            "type": "string",
                            "enum": ["cheap", "mid", "premium"],
                        },
                        "availability_hint": {"type": "string"},
                        "negatives": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "weights": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": (
                                "Per-signal importance in [0,1]. Keys are any of: "
                                "bm25, dense_text, clip_image, brightness, "
                                "modernity, geo_anchor, transit, feature_match, "
                                "price_band, freshness."
                            ),
                        },
                    },
                },
            },
            "required": ["language", "hard", "soft"],
        },
    }


# ---- Public API ----------------------------------------------------------


def understand(query: str) -> QueryUnderstanding:
    """Cached, thread-safe query understanding. Safe to call concurrently."""
    return _cached_understand(query.strip())


def invalidate_cache() -> None:
    _cached_understand.cache_clear()


# ---- Internals -----------------------------------------------------------


_call_lock = threading.Lock()


@lru_cache(maxsize=1024)
def _cached_understand(query: str) -> QueryUnderstanding:
    if not query:
        return QueryUnderstanding(
            raw_query=query,
            hard=HardFilters(),
            soft=SoftPreferences(),
        )

    settings = get_settings()

    if bedrock_available(settings):
        try:
            with _call_lock:
                return _understand_via_bedrock(query)
        except Exception as exc:
            logger.warning(
                "Bedrock query understanding failed, trying fallbacks. (%s)",
                exc,
                exc_info=False,
            )

    if settings.anthropic_api_key:
        try:
            with _call_lock:
                return _understand_via_anthropic(query)
        except Exception as exc:
            logger.warning(
                "Anthropic-direct query understanding failed, using heuristic. (%s)",
                exc,
                exc_info=False,
            )

    return _heuristic_understand(query)


def _understand_via_anthropic(query: str) -> QueryUnderstanding:
    """Call Anthropic's direct API as a Bedrock fallback.

    Uses the same tool-use schema. Requires `ANTHROPIC_API_KEY` in env.
    """
    import anthropic  # lazy import — only needed on fallback path.

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool = _tool_schema()

    response = client.messages.create(
        model=settings.anthropic_model_id,
        max_tokens=1024,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": query}],
    )

    tool_input: dict[str, Any] | None = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
            tool_input = dict(block.input or {})
            break
    if tool_input is None:
        raise RuntimeError(
            "Anthropic direct: no tool_use block in response"
        )

    return _assemble_understanding(query=query, tool_input=tool_input, used_llm=True)


def _understand_via_bedrock(query: str) -> QueryUnderstanding:
    settings = get_settings()
    client = get_bedrock_client()

    tool = _tool_schema()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": SYSTEM_PROMPT,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": [{"role": "user", "content": query}],
    }

    response = client.invoke_model(
        modelId=settings.bedrock_query_understanding_model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read())

    tool_input: dict[str, Any] | None = None
    for block in payload.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == tool["name"]:
            tool_input = block.get("input") or {}
            break
    if tool_input is None:
        raise RuntimeError(
            "Claude did not return a tool_use block. Payload=%r" % payload
        )

    return _assemble_understanding(query=query, tool_input=tool_input, used_llm=True)


def _assemble_understanding(
    *,
    query: str,
    tool_input: dict[str, Any],
    used_llm: bool,
) -> QueryUnderstanding:
    hard_raw = dict(tool_input.get("hard") or {})
    soft_raw = dict(tool_input.get("soft") or {})

    hard = HardFilters(**{k: v for k, v in hard_raw.items() if v not in (None, [], "")})

    anchors_raw = soft_raw.pop("anchors", []) or []
    anchors = [Anchor(**a) for a in anchors_raw if a.get("text")]

    soft = SoftPreferences(
        **{k: v for k, v in soft_raw.items() if v is not None},
        anchors=anchors,
    )

    return QueryUnderstanding(
        raw_query=query,
        language=tool_input.get("language", "unknown"),
        hard=hard,
        soft=soft,
        used_llm=used_llm,
        interpretation=tool_input.get("interpretation"),
    )


# ---- Heuristic fallback --------------------------------------------------


_CITY_HINTS = (
    "zürich",
    "zurich",
    "winterthur",
    "basel",
    "bern",
    "genève",
    "geneva",
    "lausanne",
    "luzern",
    "lucerne",
    "st. gallen",
    "st gallen",
    "lugano",
    "fribourg",
)
_CANTON_ALIAS = {
    "zurich": "ZH",
    "zürich": "ZH",
    "bern": "BE",
    "basel": "BS",
    "geneva": "GE",
    "genève": "GE",
    "lausanne": "VD",
    "luzern": "LU",
    "lucerne": "LU",
    "winterthur": "ZH",
}

_FEATURE_KEYWORDS = {
    "balcony": ("balcony", "balkon", "balcon"),
    "parking": ("parking", "parkplatz", "parcheggio", "stationnement"),
    "garage": ("garage", "tiefgarage"),
    "elevator": ("elevator", "lift", "aufzug", "ascenseur"),
    "fireplace": ("fireplace", "cheminée", "cheminee", "kamin"),
    "pets_allowed": ("pets allowed", "pet-friendly", "haustiere"),
    "new_build": ("new build", "neubau", "newly built"),
    "minergie_certified": ("minergie",),
    "wheelchair_accessible": ("wheelchair", "barrierefrei"),
    "private_laundry": ("in-unit laundry", "private laundry", "waschmaschine"),
    "child_friendly": ("kinderfreundlich", "family-friendly", "familienfreundlich"),
}
_SOFT_KEYWORDS = {
    "bright": ("bright", "hell", "lumineux", "lichtdurchflutet", "luminoso"),
    "modern": ("modern", "modernisiert", "moderne"),
    "quiet": ("quiet", "ruhig", "calme", "silenzioso", "tranquil"),
    "family_friendly": ("family", "familie", "famille", "kids", "kinder"),
    "pet_friendly": ("pet", "dog", "cat", "hund", "katze", "animaux"),
    "student_friendly": ("student", "studentin", "wg"),
    "near_public_transport": (
        "public transport",
        "transit",
        "tram",
        "bus",
        "bahnhof",
        "station",
        "öv",
        "transports",
    ),
    "near_schools": ("school", "schule", "école", "scuola"),
    "nice_views": ("view", "aussicht", "vue", "vista"),
    "balcony_or_terrace": ("terrace", "terrasse", "balcony", "balkon"),
    "parking": ("parking", "parkplatz"),
    "new_build": ("new build", "neubau"),
    "furnished": ("furnished", "möbliert", "meublé"),
    "spacious": ("spacious", "geräumig", "spacieux"),
}
_PRICE_INTENT_KEYWORDS = {
    "cheap": ("cheap", "affordable", "budget", "günstig", "bon marché"),
    "premium": ("premium", "luxury", "luxurious", "luxus", "high-end"),
}
_NEAR_PATTERN = re.compile(
    r"(?:close to|near|nearby|by|beside|next to|in der nähe von|nahe|proche de)\s+([A-Z][\w\u00C0-\u024F\s\-\.']{2,40})",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(
    r"(?:under|below|less than|bis|max(?:imum)?|<=?|up to)\s*(?:chf\s*)?([0-9]{3,5})",
    re.IGNORECASE,
)
_ROOMS_PATTERN = re.compile(
    r"(\d(?:[.,]\d)?)\s*[- ]?\s*(?:room|rooms|zimmer|zi\.?|pièces|locali)",
    re.IGNORECASE,
)


def _heuristic_understand(query: str) -> QueryUnderstanding:
    q = query.lower()

    hard = HardFilters()
    soft = SoftPreferences()

    for city in _CITY_HINTS:
        if city in q:
            canonical = city.title().replace("St ", "St. ")
            hard.city = [canonical]
            canton = _CANTON_ALIAS.get(city)
            if canton:
                hard.canton = canton
            break

    if (m := _PRICE_PATTERN.search(query)):
        try:
            hard.max_price = int(m.group(1))
        except ValueError:
            pass

    if (m := _ROOMS_PATTERN.search(query)):
        try:
            hard.min_rooms = float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    hard_features: list[str] = []
    for feat, keywords in _FEATURE_KEYWORDS.items():
        for kw in keywords:
            idx = q.find(kw)
            if idx < 0:
                continue
            window = q[max(0, idx - 20) : idx]
            if any(soft_trigger in window for soft_trigger in ("ideally", "maybe", "if possible", "nice to have", "hätte gern")):
                continue
            hard_features.append(feat)
            break
    if hard_features:
        hard.features = sorted(set(hard_features))

    descriptors: list[str] = []
    for field, keywords in _SOFT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            setattr(soft, field, True)
            descriptors.append(field.replace("_", " "))

    for intent, keywords in _PRICE_INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            soft.price_intent = intent  # type: ignore[assignment]
            break

    if descriptors:
        soft.descriptors = descriptors

    anchor_matches = _NEAR_PATTERN.findall(query)
    if anchor_matches:
        soft.anchors = [Anchor(text=a.strip(" ,.")) for a in anchor_matches]

    # Rough weight hints so the ranker still has something to work with.
    weights: dict[str, float] = {
        "bm25": 0.4,
        "dense_text": 0.5,
        "feature_match": 0.6,
    }
    if soft.anchors:
        weights["geo_anchor"] = 0.8
    if soft.bright:
        weights["brightness"] = 0.7
    if soft.modern:
        weights["modernity"] = 0.6
    if soft.near_public_transport:
        weights["transit"] = 0.6
    if soft.price_intent:
        weights["price_band"] = 0.6
    soft.weights = weights

    return QueryUnderstanding(
        raw_query=query,
        hard=hard,
        soft=soft,
        used_llm=False,
        interpretation=None,
    )
