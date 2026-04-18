"""Participant-side Pydantic contracts.

These extend the harness' `HardFilters` with structured representations of
the *soft* query intent so that downstream modules (soft filtering, ranking,
reranker, explanation writer) can all code against typed interfaces.

Every module under `app/participant/` that consumes the query understanding
should import from here instead of passing raw dicts around.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import HardFilters


# Canonical names used everywhere (hard filter column suffixes, soft boolean
# preferences, the Claude tool-use schema). Keep in sync with the
# `feature_*` columns created in app/harness/csv_import.py.
FEATURE_NAMES: tuple[str, ...] = (
    "balcony",
    "elevator",
    "parking",
    "garage",
    "fireplace",
    "child_friendly",
    "pets_allowed",
    "temporary",
    "new_build",
    "wheelchair_accessible",
    "private_laundry",
    "minergie_certified",
)


class Anchor(BaseModel):
    """A named location the user references for proximity scoring.

    Examples: "ETH Zurich", "Bahnhof Bern", "near Seebach", "Kreis 4".
    Latitude / longitude are filled in by the geospatial enrichment step;
    the query understanding step only extracts `text`.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Natural-language name of the anchor.")
    latitude: float | None = None
    longitude: float | None = None
    max_distance_km: float | None = Field(
        default=None,
        ge=0,
        description="Optional hard cap on distance to this anchor.",
    )


class SoftPreferences(BaseModel):
    """Soft signals extracted from the query.

    Every field is optional. `weights` is a dict of signal-name -> importance
    in [0, 1] inferred by the query-understanding LLM; the ranker combines
    the signals linearly using these weights as a starting point.
    """

    model_config = ConfigDict(extra="forbid")

    # Free-text soft hints the LLM pulled out, verbatim. Used for semantic
    # retrieval (embedding + BM25).
    descriptors: list[str] = Field(
        default_factory=list,
        description='Adjectival hints like ["bright", "modern kitchen", "quiet"].',
    )

    # Discrete boolean preferences. True = user wants it, False = user explicitly
    # does NOT want it, None = not mentioned.
    bright: bool | None = None
    modern: bool | None = None
    quiet: bool | None = None
    family_friendly: bool | None = None
    pet_friendly: bool | None = None
    student_friendly: bool | None = None
    near_public_transport: bool | None = None
    near_schools: bool | None = None
    nice_views: bool | None = None
    balcony_or_terrace: bool | None = None
    parking: bool | None = None
    new_build: bool | None = None
    furnished: bool | None = None
    spacious: bool | None = None

    # Places the user mentioned for proximity (e.g., "close to ETH Zurich").
    anchors: list[Anchor] = Field(default_factory=list)

    # Soft price intent, when the user said something like "not too expensive"
    # or "affordable" instead of a number. Ranker interprets relative to the
    # city's price distribution.
    price_intent: Literal["cheap", "mid", "premium"] | None = None

    # Soft availability intent: "June move-in", "ASAP", "summer".
    availability_hint: str | None = None

    # Neighborhood / district / locality names the user mentioned INSIDE a
    # city (e.g. "Bümpliz" for Bern, "Ouchy" for Lausanne, "Oerlikon" for
    # Zürich, "Eaux-Vives" for Geneva, "Kleinbasel" for Basel). These are
    # stored separately from `hard.city` because excluding purely on a
    # district keyword would destroy recall for cities where descriptions
    # don't mention the district at all. The ranker uses them as a
    # *strong preference* (soft re-weight + BM25 boost) and only as a
    # hard filter when at least `min_keep` candidates contain the hint.
    locality_hints: list[str] = Field(
        default_factory=list,
        description=(
            'Sub-city areas named by the user, e.g. ["Bümpliz"] or '
            '["Oerlikon", "Schwamendingen"]. Never the main city itself.'
        ),
    )

    # Per-signal weight hint in [0, 1]. Keys are any of:
    #   "bm25", "dense_text", "clip_image", "brightness", "modernity",
    #   "geo_anchor", "transit", "feature_match", "price_band", "freshness"
    # Missing keys default to a baseline weight in the ranker.
    weights: dict[str, float] = Field(default_factory=dict)

    # Negative signals — things the user explicitly rejected.
    # Example: "no ground floor", "not temporary", "no student housing".
    negatives: list[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    """Combined output of the query understanding step.

    Produced by a single Claude Sonnet 4 tool-use call, consumed by:
    - hard filter (via `.hard`)
    - soft filtering / ranking (via `.soft`)
    - explanation generation (via `.raw_query` + `.soft.descriptors`)
    """

    model_config = ConfigDict(extra="forbid")

    raw_query: str
    language: Literal["de", "fr", "it", "en", "unknown"] = "unknown"
    hard: HardFilters
    soft: SoftPreferences
    # Populated when the LLM call succeeded; `False` means the heuristic
    # fallback was used and downstream modules may want to be more cautious.
    used_llm: bool = False
    # Optional natural-language restatement of what the system understood.
    # Useful for clarification UIs and for jury explainability.
    interpretation: str | None = None


class RankingSignals(BaseModel):
    """Per-candidate numeric signals in [0, 1], computed by soft_filtering.

    Matches the `weights` keys in SoftPreferences. Every signal defaults to
    0.0 when absent so the ranker can always compute a weighted sum.
    """

    model_config = ConfigDict(extra="forbid")

    bm25: float = 0.0
    dense_text: float = 0.0
    clip_image: float = 0.0
    brightness: float = 0.0
    modernity: float = 0.0
    geo_anchor: float = 0.0
    transit: float = 0.0
    feature_match: float = 0.0
    price_band: float = 0.0
    freshness: float = 0.0
    # Free-form bag for signals not yet standardized.
    extra: dict[str, float] = Field(default_factory=dict)


class ScoredCandidate(BaseModel):
    """Output of soft_filtering, input to ranking."""

    model_config = ConfigDict(extra="allow")

    listing_id: str
    signals: RankingSignals
    # The raw DB row (as returned by the hard filter) carried through for
    # downstream use (image paths, explanations, map pins).
    listing: dict[str, Any]
