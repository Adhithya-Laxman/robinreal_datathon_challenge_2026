"""Unified search + ranking pipeline.

End-to-end flow (one function, one call):

    query -> query understanding (hard + soft facts, via Claude)
          -> hard filter SQL + relaxation (keeps >= 5 candidates)
          -> dense text score      (e5 multilingual via fastembed)
          -> bm25 lexical score    (rank-bm25, multilingual stemming)
          -> vlm image score       (SigLIP2 on listing photos) [optional]
          -> geo proximity scores  (transit / schools / POIs)
          -> price-band score      (from soft price_intent)
          -> weighted fusion       (uses soft.weights or defaults)
          -> top-k results with per-signal breakdown

Design notes:
  * Signals are only activated when the query intent actually asks for them.
    E.g. `geo_transit` weight is 0 unless `soft.near_public_transport` is True.
    This avoids geo-enrichment quietly dominating unrelated queries.
  * Every stage is defensive: missing VLM shards, missing geo columns, empty
    BM25 index, 0 embeddings - any of these degrade to a 0 signal instead of
    crashing. The pipeline always returns *something* if hard filter did.
  * All per-signal scores are normalized to [0, 1] before fusion so weights
    are directly interpretable as importance.
  * VLM is opt-in via `use_vlm=True` because loading SigLIP2 text tower costs
    ~10 s + ~2 GB download on first call.

Public API:

    from app.participant.unified_ranker import unified_search
    results = unified_search("helle 3-Zimmer Wohnung in Zürich", top_k=10)
    for r in results:
        print(r.listing_id, r.score, r.signals)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.harness.search_service import filter_hard_facts
from app.models.schemas import HardFilters
from app.participant.bm25_index import index_exists, load_index
from app.participant.embeddings import search_by_query_text
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.query_understanding import understand
from app.participant.schemas import QueryUnderstanding, SoftPreferences

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Defaults + constants                                                         #
# --------------------------------------------------------------------------- #

_VLM_SHARDS_DIR = Path(__file__).resolve().parents[2] / "features_vlm" / "siglip2"

# Signals we may compute. Weights from `soft.weights` override per-key.
# The value is a (default_weight, always_on) tuple:
#   always_on = True  -> weight applies regardless of query intent
#   always_on = False -> weight is zeroed unless the relevant soft signal fires
_SIGNAL_DEFAULTS: dict[str, tuple[float, bool]] = {
    "dense":       (0.35, True),
    "bm25":        (0.20, True),
    "vlm":         (0.20, False),  # gated on visual soft descriptors
    "geo_transit": (0.10, False),  # gated on near_public_transport
    "geo_school":  (0.05, False),  # gated on near_schools / family_friendly
    "geo_anchor":  (0.15, False),  # gated on soft.anchors (ETH, station, ...)
    "price_band":  (0.10, False),  # gated on price_intent
}

# Distance scale (meters) for exp(-d / tau) geo-proximity scoring.
_TAU_TRANSIT = 500.0
_TAU_SCHOOL = 1000.0
_TAU_ANCHOR = 800.0   # ETH/EPFL/Uni proximity — "within walking distance"

# The LLM (see `app/participant/schemas.py::SoftPreferences.weights`) emits
# weights under its own naming convention. We translate them into the ranker's
# internal signal names so the LLM's per-query prioritization is actually
# honored. Multiple LLM keys can map to the same signal — we take the max.
_LLM_WEIGHT_KEY_MAP: dict[str, str] = {
    "dense_text":    "dense",
    "bm25":          "bm25",
    # Visual signals — all these land on the single "vlm" signal.
    "clip_image":    "vlm",
    "brightness":    "vlm",
    "modernity":     "vlm",
    # Geo / proximity signals.
    "transit":       "geo_transit",
    # `geo_anchor` is proximity to a named landmark (ETH, station, lake...).
    # We score it off `geo_university_m` because the enrichment step populates
    # that column specifically for ETH/EPFL/Uni anchors which dominate queries.
    "geo_anchor":    "geo_anchor",
    # Price intent.
    "price_band":    "price_band",
    # Not yet implemented signals are ignored (but logged).
    # "feature_match", "freshness"
}

# Words in descriptors that suggest the VLM can help.
_VLM_TRIGGER_WORDS = {
    "bright", "hell", "clair", "luminoso", "luminosa",
    "modern", "moderne", "moderno",
    "dark", "dunkel", "sombre", "scuro",
    "cozy", "gemütlich", "confortable",
    "spacious", "geräumig", "spacieux", "spazioso",
    "view", "aussicht", "vue", "vista",
    "kitchen", "küche", "cuisine", "cucina",
    "bathroom", "bad", "salle de bain", "bagno",
    "garden", "garten", "jardin", "giardino",
    "balcony", "balkon", "balcon", "balcone",
    "terrace", "terrasse", "terrazza",
}


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class StageStatus:
    """Per-stage record of what happened during a pipeline run.

    `state` is one of:
        "ok"       - stage ran, produced results
        "skipped"  - stage was disabled by config / gating
        "empty"    - stage ran but produced no usable signal
        "missing"  - stage disabled because a prerequisite was absent
                     (shards, DB column, BM25 index, etc.)
        "error"   - stage raised; `detail` has the exception message
    """
    name: str
    state: str
    detail: str = ""
    scored: int = 0   # how many candidates the stage produced a score for


@dataclass
class UnifiedResult:
    """One ranked listing with a full breakdown of how we got its score."""

    listing_id: str
    score: float
    signals: dict[str, float]           # per-signal [0, 1] score
    weights: dict[str, float]           # weights applied (for explainability)
    row: dict[str, Any] = field(default_factory=dict)  # raw DB row

    def to_dict(self, rank: int | None = None) -> dict[str, Any]:
        """JSON-serializable view with ALL fields useful for analysis / UI.

        Groups the raw row into semantic buckets so the output is easy to scan:
          * identity     - ids, urls
          * location     - address + coordinates + POI distances
          * property     - price, rooms, area, features
          * images       - list of image URLs
          * scoring      - per-signal scores, weights, weighted contributions
        """
        r = self.row
        contributions = {k: self.signals.get(k, 0.0) * self.weights.get(k, 0.0)
                         for k in self.signals}
        return {
            "rank": rank,
            "listing_id": self.listing_id,
            "final_score": round(self.score, 6),

            "identity": {
                "listing_id": r.get("listing_id"),
                "original_url": r.get("original_url"),
                "object_category": r.get("object_category"),
                "object_type": r.get("object_type"),
                "offer_type": r.get("offer_type"),
                "available_from": r.get("available_from"),
                "title": r.get("title"),
                "description": r.get("description"),
            },

            "location": {
                "street": r.get("street"),
                "postal_code": r.get("postal_code"),
                "city": r.get("city"),
                "canton": r.get("canton"),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                # CSV-provided distances (not always populated).
                "csv_distances_m": {
                    "public_transport": r.get("distance_public_transport"),
                    "shop": r.get("distance_shop"),
                    "kindergarten": r.get("distance_kindergarten"),
                    "school_1": r.get("distance_school_1"),
                    "school_2": r.get("distance_school_2"),
                },
                # Enriched distances from scripts/geo_enrich.py (optional).
                "enriched_distances_m": {
                    "transit": r.get("geo_transit_m"),
                    "supermarket": r.get("geo_supermarket_m"),
                    "school": r.get("geo_school_m"),
                    "university": r.get("geo_university_m"),
                },
            },

            "property": {
                "price_chf": r.get("price"),
                "rooms": r.get("rooms"),
                "area_sqm": r.get("area"),
                "features": r.get("features") or [],
            },

            "images": {
                "urls": r.get("image_urls") or [],
                "hero_image_url": r.get("hero_image_url"),
            },

            "scoring": {
                "signals_normalized": {k: round(v, 6) for k, v in self.signals.items()},
                "weights_used": {k: round(v, 6) for k, v in self.weights.items()},
                "weighted_contributions": {k: round(v, 6) for k, v in contributions.items()},
                "top_signals": sorted(
                    [(k, v) for k, v in contributions.items() if v > 0],
                    key=lambda t: -t[1],
                )[:3],
            },
        }


@dataclass
class UnifiedResponse:
    """Full pipeline output, including the understanding step."""

    query: str
    understanding: QueryUnderstanding
    hard: HardFilters
    candidates_before_rerank: int
    weights_used: dict[str, float]
    results: list[UnifiedResult]
    stages: list[StageStatus] = field(default_factory=list)
    llm_weight_hints: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Comprehensive JSON-serializable dump of the entire pipeline run."""
        hard_dump = self.hard.model_dump(exclude_none=True)
        for k in ("limit", "offset"):
            hard_dump.pop(k, None)
        soft_dump = self.understanding.soft.model_dump(exclude_defaults=True, exclude_none=True)
        return {
            "query": self.query,
            "understanding": {
                "language": self.understanding.language,
                "interpretation": self.understanding.interpretation,
                "used_llm": self.understanding.used_llm,
                "hard_filters": hard_dump,
                "soft_preferences": soft_dump,
            },
            "pipeline": {
                "candidates_before_rerank": self.candidates_before_rerank,
                "weights_used": {k: round(v, 6) for k, v in self.weights_used.items()},
                "active_signals": sorted(
                    [k for k, v in self.weights_used.items() if v > 0.0]
                ),
                "llm_weight_hints": self.llm_weight_hints,
                "stages": [
                    {
                        "name": s.name,
                        "state": s.state,
                        "detail": s.detail,
                        "scored": s.scored,
                    }
                    for s in self.stages
                ],
            },
            "results": [r.to_dict(rank=i) for i, r in enumerate(self.results, 1)],
        }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def unified_search(
    query: str,
    *,
    top_k: int = 10,
    db_path: Path | None = None,
    use_vlm: bool = False,
    vlm_shards_dir: Path | None = None,
    override_weights: dict[str, float] | None = None,
) -> UnifiedResponse:
    """Run the full hybrid pipeline for `query` and return top-k ranked listings.

    Parameters
    ----------
    query:
        Natural-language query in any of de / fr / it / en.
    top_k:
        Number of listings to return after weighted fusion.
    db_path:
        SQLite DB with listings + embeddings + (optional) geo columns. Defaults
        to `get_settings().db_path`.
    use_vlm:
        If True AND SigLIP2 shards exist AND query has visual descriptors,
        score image similarity and include it in fusion.
    vlm_shards_dir:
        Override for the shards location.
    override_weights:
        Dict partially overriding the default weights. Useful for A/B tuning.
    """
    db_path = db_path or get_settings().db_path
    stages: list[StageStatus] = []

    # --- Stage 1: query understanding --------------------------------------
    hard = extract_hard_facts(query)                 # Haiku, strict JSON
    qu = understand(query)                           # Sonnet, full QU (soft)
    soft: SoftPreferences = qu.soft
    stages.append(StageStatus(
        name="query_understanding",
        state="ok" if qu.used_llm else "empty",
        detail=f"lang={qu.language} used_llm={qu.used_llm} "
               f"descriptors={len(soft.descriptors)} "
               f"anchors={len(soft.anchors)}",
    ))

    # --- Stage 2: hard filter (+ relaxation) -------------------------------
    # Location / offer_type / object_category are NEVER relaxed: showing a
    # listing from the wrong city is worse than showing fewer listings.
    # See app/harness/search_service._INVIOLABLE_FIELDS.
    hard.limit = 200                                 # wider pool for reranking
    hard.offset = 0
    candidates, relaxed_fields = filter_hard_facts(db_path, hard)
    relaxed_note = (f" relaxed={relaxed_fields}"
                    if relaxed_fields else " relaxed=none")
    stages.append(StageStatus(
        name="hard_filter",
        state="ok" if candidates else "empty",
        detail=(f"city={hard.city} max_price={hard.max_price} "
                f"rooms=[{hard.min_rooms}, {hard.max_rooms}]"
                f"{relaxed_note}"),
        scored=len(candidates),
    ))
    if not candidates:
        logger.warning("Hard filter returned no candidates for query=%r", query)
        return UnifiedResponse(
            query=query, understanding=qu, hard=hard,
            candidates_before_rerank=0, weights_used={}, results=[],
            stages=stages, llm_weight_hints=dict(soft.weights),
        )

    cand_ids = [str(c["listing_id"]) for c in candidates]
    rows_by_id = {str(c["listing_id"]): c for c in candidates}

    # Merge the enriched geo columns (populated by `scripts/geo_enrich.py`)
    # into the candidate rows. Hard filter SELECT does not return them,
    # so geo scoring + JSON output would otherwise always show null.
    geo_cols_found = _merge_geo_columns(rows_by_id, db_path)
    stages.append(StageStatus(
        name="geo_column_merge",
        state="ok" if geo_cols_found else "missing",
        detail=(f"merged {len(geo_cols_found)} columns: {geo_cols_found}"
                if geo_cols_found else
                "no geo_* columns in DB - run scripts/geo_enrich.py"),
    ))

    # --- Stage 3: text scoring (dense + bm25) ------------------------------
    dense_scores, dense_status = _score_dense(query, cand_ids)
    stages.append(dense_status)
    bm25_scores, bm25_status = _score_bm25(query, cand_ids)
    stages.append(bm25_status)

    # --- Stage 4: VLM (optional) -------------------------------------------
    vlm_scores: dict[str, float] = {}
    vlm_prompt = _build_vlm_prompt(query, soft)
    if not use_vlm:
        stages.append(StageStatus(name="vlm", state="skipped",
                                   detail="use_vlm=False"))
    elif not vlm_prompt:
        stages.append(StageStatus(
            name="vlm", state="skipped",
            detail="no visual descriptors found in query; VLM would be noise",
        ))
    else:
        shards_dir = vlm_shards_dir or _VLM_SHARDS_DIR
        # CRITICAL: shards are keyed on `platform_id` (e.g. "29387655") or
        # "sred_<id>", but our candidate IDs are `listing_id` (the DB PK).
        # Build the mapping here so VLM actually finds matches.
        shard_key_map = _build_shard_key_map(cand_ids, db_path)
        vlm_scores, vlm_status = _score_vlm(
            vlm_prompt, cand_ids, shards_dir, shard_key_map,
        )
        stages.append(vlm_status)

    # --- Stage 5: geo proximity --------------------------------------------
    geo_transit_scores = _score_geo_distance(rows_by_id, "geo_transit_m", _TAU_TRANSIT)
    stages.append(StageStatus(
        name="geo_transit",
        state=("ok" if geo_transit_scores else
               ("missing" if "geo_transit_m" not in geo_cols_found else "empty")),
        scored=len(geo_transit_scores),
        detail=(f"tau={_TAU_TRANSIT}m" if geo_transit_scores else
                "no geo_transit_m column or all NULL"),
    ))
    geo_school_scores = _score_geo_distance(rows_by_id, "geo_school_m", _TAU_SCHOOL)
    stages.append(StageStatus(
        name="geo_school",
        state=("ok" if geo_school_scores else
               ("missing" if "geo_school_m" not in geo_cols_found else "empty")),
        scored=len(geo_school_scores),
        detail=(f"tau={_TAU_SCHOOL}m" if geo_school_scores else
                "no geo_school_m column or all NULL"),
    ))
    # `geo_anchor` is semantically "proximity to the named landmark(s) in the
    # query". We use `geo_university_m` because the enrichment populated it
    # for ETH/EPFL/Uni anchors, which dominate in practice.
    geo_anchor_scores = _score_geo_distance(rows_by_id, "geo_university_m", _TAU_ANCHOR)
    anchor_names = [a.text for a in soft.anchors if a.text]
    stages.append(StageStatus(
        name="geo_anchor",
        state=("ok" if geo_anchor_scores else
               ("missing" if "geo_university_m" not in geo_cols_found else "empty")),
        scored=len(geo_anchor_scores),
        detail=(f"tau={_TAU_ANCHOR}m anchors={anchor_names or '-'}"
                if geo_anchor_scores else
                "no geo_university_m column or all NULL"),
    ))

    # --- Stage 6: price band -----------------------------------------------
    price_scores = _score_price_band(rows_by_id, soft.price_intent)
    stages.append(StageStatus(
        name="price_band",
        state=("ok" if price_scores else
               ("skipped" if not soft.price_intent else "empty")),
        scored=len(price_scores),
        detail=f"intent={soft.price_intent}",
    ))

    # --- Stage 7: resolve weights -------------------------------------------
    weights, weight_trace = _resolve_weights(
        soft=soft,
        vlm_active=bool(vlm_scores),
        override=override_weights or {},
    )
    stages.append(StageStatus(
        name="weight_resolution",
        state="ok",
        detail=weight_trace,
    ))

    # --- Stage 8: fuse + rank ------------------------------------------------
    per_signal: dict[str, dict[str, float]] = {
        "dense":       dense_scores,
        "bm25":        bm25_scores,
        "vlm":         vlm_scores,
        "geo_transit": geo_transit_scores,
        "geo_school":  geo_school_scores,
        "geo_anchor":  geo_anchor_scores,
        "price_band":  price_scores,
    }

    results: list[UnifiedResult] = []
    for lid in cand_ids:
        signals = {k: per_signal[k].get(lid, 0.0) for k in per_signal}
        score = sum(signals[k] * weights.get(k, 0.0) for k in signals)
        results.append(UnifiedResult(
            listing_id=lid,
            score=score,
            signals=signals,
            weights=weights,
            row=rows_by_id[lid],
        ))

    results.sort(key=lambda r: -r.score)
    return UnifiedResponse(
        query=query,
        understanding=qu,
        hard=hard,
        candidates_before_rerank=len(cand_ids),
        weights_used=weights,
        results=results[:top_k],
        stages=stages,
        llm_weight_hints=dict(soft.weights),
    )


# --------------------------------------------------------------------------- #
# Signal computation helpers                                                   #
# --------------------------------------------------------------------------- #


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max to [0, 1] so weights are directly comparable across signals."""
    if not scores:
        return scores
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _score_dense(
    query: str, cand_ids: list[str],
) -> tuple[dict[str, float], StageStatus]:
    try:
        hits = search_by_query_text(query, top_k=len(cand_ids), candidate_ids=cand_ids)
    except Exception as exc:  # noqa: BLE001 - defensive: embeddings are optional
        logger.warning("Dense scoring disabled: %s", exc)
        return {}, StageStatus(name="dense", state="error", detail=str(exc))
    raw = {lid: max(0.0, score) for lid, score in hits}
    scores = _normalize(raw)
    status = StageStatus(
        name="dense",
        state="ok" if scores else "empty",
        scored=len(scores),
        detail=f"e5-large; matched {len(scores)}/{len(cand_ids)} candidates",
    )
    return scores, status


def _score_bm25(
    query: str, cand_ids: list[str],
) -> tuple[dict[str, float], StageStatus]:
    if not index_exists():
        logger.warning("BM25 index not found; bm25 signal = 0")
        return {}, StageStatus(
            name="bm25", state="missing",
            detail="run scripts/build_bm25_index.py first",
        )
    try:
        idx = load_index()
        hits = idx.search(query, top_k=len(cand_ids), candidate_ids=cand_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 scoring disabled: %s", exc)
        return {}, StageStatus(name="bm25", state="error", detail=str(exc))
    raw = {lid: max(0.0, s) for lid, s in hits}
    scores = _normalize(raw)
    return scores, StageStatus(
        name="bm25",
        state="ok" if scores else "empty",
        scored=len(scores),
        detail=f"matched {len(scores)}/{len(cand_ids)} candidates",
    )


def _build_vlm_prompt(query: str, soft: SoftPreferences) -> str | None:
    """Pick a short English-ish prompt to feed the SigLIP2 text tower.

    Prefer concrete descriptors pulled by the LLM; fall back to the raw query
    only if at least one visual trigger word is present (else VLM is pointless).
    """
    desc_text = " ".join(soft.descriptors).strip()
    if desc_text:
        return desc_text
    lower = query.lower()
    if any(w in lower for w in _VLM_TRIGGER_WORDS):
        return query
    return None


def _build_shard_key_map(
    cand_ids: list[str],
    db_path: Path,
) -> dict[str, str]:
    """Map each `listing_id` to the key actually used inside the VLM shards.

    Shards use two conventions (see `src/vision/search.py::extract_listing_id`):
        platform_id=<n>      for comparis listings
        sred_<listing_id>    for SRED tiles

    We need BOTH possible keys because the data has listings with a
    platform_id AND (for SRED rows) not. Returns a dict
    `shard_key -> listing_id` so the VLM scorer can reverse-lookup fast.
    """
    if not cand_ids:
        return {}
    con = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" * len(cand_ids))
        rows = con.execute(
            f"SELECT listing_id, platform_id FROM listings "
            f"WHERE listing_id IN ({placeholders})",
            cand_ids,
        ).fetchall()
    finally:
        con.close()

    mapping: dict[str, str] = {}
    for listing_id, platform_id in rows:
        lid = str(listing_id)
        if platform_id:
            mapping[str(platform_id)] = lid
        # SRED rows: the shard extracts "sred_<listing_id>". We store both
        # shapes so the lookup is unambiguous whatever the shard contains.
        mapping[f"sred_{lid}"] = lid
        mapping[lid] = lid  # also allow direct match
    return mapping


def _score_vlm(
    prompt: str,
    cand_ids: list[str],
    shards_dir: Path,
    shard_key_map: dict[str, str] | None = None,
) -> tuple[dict[str, float], StageStatus]:
    """Return (normalized per-listing VLM scores, status).

    Loud on every failure mode so you can diagnose silent 0s.
    `shard_key_map` is a dict `shard_key -> listing_id` built from the DB
    (`_build_shard_key_map`). If omitted, we fall back to naive matching,
    which only works if listing_id == shard key (rarely true).
    """
    if not shards_dir.exists():
        msg = f"shards_dir does not exist: {shards_dir}"
        logger.warning("[vlm] %s", msg)
        return {}, StageStatus(name="vlm", state="missing", detail=msg)

    shard_files = list(shards_dir.glob("shard_*.npz"))
    if not shard_files:
        msg = f"no shard_*.npz under {shards_dir}"
        logger.warning("[vlm] %s", msg)
        return {}, StageStatus(name="vlm", state="missing", detail=msg)

    try:
        # Lazy import; heavy deps (torch, transformers) are only pulled when used.
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        src_path = repo_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        import torch

        from vision.search import encode_text, extract_listing_id, load_shards

        # GPU-memory-safe strategy: the dense text tower (e5-large, ~4 GB) is
        # already resident on GPU from `_score_dense`. SigLIP2-so400m needs
        # ~3 GB which overflows 8 GB cards. Since we only encode a SINGLE
        # short prompt with the SigLIP2 text tower, CPU is ~1 s — trivial.
        # All similarity math is numpy matmul on CPU anyway.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cpu")

        paths, embs, model_name = load_shards(shards_dir)
        logger.info("[vlm] loaded %d image embeddings, dim=%d, from %d shards; "
                    "encoding prompt=%r on CPU",
                    len(paths), embs.shape[1], len(shard_files), prompt)
        qvec = encode_text(
            prompt,
            model_name or "google/siglip2-so400m-patch14-384",
            device,
        )
        scores = embs @ qvec  # (N_images,)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[vlm] scoring failed during load/encode: %s", exc)
        return {}, StageStatus(
            name="vlm", state="error",
            detail=f"{type(exc).__name__}: {exc}",
        )

    # Shards store by `platform_id` or `sred_<lid>`; our candidates are
    # `listing_id`. Use the pre-built map (via DB join) for O(1) lookup.
    key_map = shard_key_map or {lid: lid for lid in cand_ids}
    per_listing_max: dict[str, float] = {}
    images_seen = 0
    images_matched = 0
    sample_ids_shard: list[str] = []
    for i, rel in enumerate(paths):
        images_seen += 1
        # extract_listing_id returns e.g. "platform_id=29387655" or "sred_1154156"
        raw_id = extract_listing_id(rel)
        shard_key = (raw_id.split("=", 1)[1]
                     if raw_id.startswith("platform_id=") else raw_id)
        if len(sample_ids_shard) < 5:
            sample_ids_shard.append(shard_key)
        lid = key_map.get(shard_key)
        if lid is None:
            continue
        images_matched += 1
        s = float(scores[i])
        if lid not in per_listing_max or s > per_listing_max[lid]:
            per_listing_max[lid] = s

    if not per_listing_max:
        sample_cand_keys = list(key_map.keys())[:5]
        msg = (f"shard IDs don't overlap with candidate-shard keys. "
               f"candidate keys sample: {sample_cand_keys} | "
               f"shard ID sample: {sample_ids_shard} | "
               f"scanned {images_seen} images across shards")
        logger.warning("[vlm] %s", msg)
        return {}, StageStatus(
            name="vlm", state="empty",
            detail=msg, scored=0,
        )

    scores_norm = _normalize(per_listing_max)
    detail = (f"matched {images_matched} imgs across {len(per_listing_max)} "
              f"listings out of {len(cand_ids)} candidates "
              f"(scanned {images_seen} shard images)")
    logger.info("[vlm] %s", detail)
    return scores_norm, StageStatus(
        name="vlm", state="ok",
        scored=len(scores_norm),
        detail=detail,
    )


def _score_geo_distance(
    rows: dict[str, dict[str, Any]],
    column: str,
    tau: float,
) -> dict[str, float]:
    """Raw score = exp(-d / tau); higher means closer. Normalized to [0, 1]."""
    raw: dict[str, float] = {}
    for lid, r in rows.items():
        d = r.get(column)
        if d is None:
            continue
        try:
            d_val = float(d)
        except (TypeError, ValueError):
            continue
        raw[lid] = math.exp(-d_val / tau)
    return _normalize(raw)


def _score_price_band(
    rows: dict[str, dict[str, Any]],
    intent: str | None,
) -> dict[str, float]:
    """Reward listings whose price matches the user's soft intent.

    cheap   -> rank ascending by price
    mid     -> closest to median
    premium -> descending by price
    """
    if intent not in ("cheap", "mid", "premium"):
        return {}
    prices = [(lid, float(r["price"])) for lid, r in rows.items()
              if r.get("price") is not None]
    if not prices:
        return {}

    if intent == "cheap":
        raw = {lid: -p for lid, p in prices}
    elif intent == "premium":
        raw = {lid: p for lid, p in prices}
    else:  # mid
        median = float(np.median([p for _, p in prices]))
        raw = {lid: -abs(p - median) for lid, p in prices}
    return _normalize(raw)


# --------------------------------------------------------------------------- #
# Weight resolution                                                            #
# --------------------------------------------------------------------------- #


def _resolve_weights(
    *,
    soft: SoftPreferences,
    vlm_active: bool,
    override: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Decide each signal's weight based on query intent.

    Pipeline:
      1. Seed weights from `_SIGNAL_DEFAULTS` (zero out gated-off signals).
      2. Remap the LLM-produced per-query `soft.weights` (which use its
         own vocabulary: brightness/modernity/clip_image/dense_text/...)
         onto our internal signal names via `_LLM_WEIGHT_KEY_MAP`.
         When multiple LLM keys map to the same signal (brightness+modernity
         -> vlm), we take the MAX so that a visually-focused query really does
         push the VLM signal up.
      3. Apply explicit `override` last (used by CLI `--weight` flags).
      4. L1-normalize so weights sum to 1.

    Returns the normalized weights and a human-readable trace string.
    """
    weights: dict[str, float] = {}
    gated_off: list[str] = []
    for key, (default, always_on) in _SIGNAL_DEFAULTS.items():
        if always_on:
            weights[key] = default
        elif _gate_active(key, soft, vlm_active):
            weights[key] = default
        else:
            weights[key] = 0.0
            gated_off.append(key)

    # --- Step 2: remap LLM weights into our signal names -------------------
    llm_mapped: dict[str, float] = {}
    ignored_llm: list[str] = []
    for llm_key, val in soft.weights.items():
        internal = _LLM_WEIGHT_KEY_MAP.get(llm_key)
        if internal is None or internal not in weights:
            ignored_llm.append(llm_key)
            continue
        # Respect signal gating: if the gate is off, don't let LLM revive it
        # (e.g. geo_transit needs `near_public_transport=True`).
        if weights[internal] == 0.0 and internal in gated_off:
            continue
        prev = llm_mapped.get(internal, 0.0)
        llm_mapped[internal] = max(prev, max(0.0, float(val)))

    # Special case: if LLM put weight on brightness/modernity/clip_image but
    # the VLM signal isn't actually active (shards missing / use_vlm=False),
    # redistribute that intent onto `dense` so we at least use the textual
    # descriptors instead of silently dropping the LLM's prioritization.
    if not vlm_active:
        visual_hint = max(
            (float(soft.weights.get(k, 0.0)) for k in
             ("brightness", "modernity", "clip_image")),
            default=0.0,
        )
        if visual_hint > 0:
            llm_mapped["dense"] = max(llm_mapped.get("dense", 0.0),
                                      weights.get("dense", 0.0) + 0.5 * visual_hint)

    for k, v in llm_mapped.items():
        weights[k] = v

    # --- Step 3: explicit CLI overrides take precedence --------------------
    for k, v in override.items():
        if k in weights:
            weights[k] = max(0.0, float(v))

    total = sum(weights.values())
    if total <= 0:
        weights = {k: d for k, (d, on) in _SIGNAL_DEFAULTS.items() if on}
        total = sum(weights.values()) or 1.0
    normed = {k: v / total for k, v in weights.items()}

    trace = (
        f"gated_off={gated_off or []} "
        f"llm_applied={ {k: round(v, 3) for k, v in llm_mapped.items()} } "
        f"llm_ignored={ignored_llm or []} "
        f"override={override or {}}"
    )
    return normed, trace


def _gate_active(signal: str, soft: SoftPreferences, vlm_active: bool) -> bool:
    if signal == "vlm":
        return vlm_active
    if signal == "geo_transit":
        return bool(soft.near_public_transport)
    if signal == "geo_school":
        return bool(soft.near_schools or soft.family_friendly)
    if signal == "geo_anchor":
        # Open the gate if the LLM extracted any named landmark (ETH, station,
        # lake...) OR explicitly assigned a geo_anchor weight.
        return bool(soft.anchors) or float(soft.weights.get("geo_anchor", 0.0)) > 0
    if signal == "price_band":
        return soft.price_intent is not None
    return True


# --------------------------------------------------------------------------- #
# Convenience for scripts                                                      #
# --------------------------------------------------------------------------- #


_GEO_COLUMNS = ("geo_transit_m", "geo_supermarket_m", "geo_school_m", "geo_university_m")


def _merge_geo_columns(
    rows_by_id: dict[str, dict[str, Any]],
    db_path: Path,
) -> list[str]:
    """Fetch the enriched geo_* columns for the candidate IDs and merge into rows.

    Returns the list of geo columns actually merged (may be empty if the DB
    has not been enriched yet via `scripts/geo_enrich.py`).
    """
    if not rows_by_id:
        return []
    con = sqlite3.connect(db_path)
    try:
        existing = {row[1] for row in con.execute(
            "PRAGMA table_info(listings)"
        ).fetchall()}
        available = [c for c in _GEO_COLUMNS if c in existing]
        if not available:
            return []
        ids = list(rows_by_id.keys())
        placeholders = ",".join("?" * len(ids))
        cols = ", ".join(["listing_id", *available])
        rows = con.execute(
            f"SELECT {cols} FROM listings WHERE listing_id IN ({placeholders})",
            ids,
        ).fetchall()
    finally:
        con.close()

    for row in rows:
        lid = str(row[0])
        target = rows_by_id.get(lid)
        if target is None:
            continue
        for i, col in enumerate(available, start=1):
            target[col] = row[i]
    return available


def fetch_display_rows(
    listing_ids: list[str],
    db_path: Path | None = None,
) -> dict[str, sqlite3.Row]:
    """Pull the small display subset of columns for the CLI formatter.

    Geo columns are only requested if they exist (schema is migrated by
    `scripts/geo_enrich.py`), so this works on a freshly bootstrapped DB too.
    """
    if not listing_ids:
        return {}
    db_path = db_path or get_settings().db_path
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    existing = {row[1] for row in con.execute("PRAGMA table_info(listings)").fetchall()}
    base = ["listing_id", "title", "city", "canton", "price", "rooms", "area"]
    optional = [c for c in ("geo_transit_m", "geo_school_m") if c in existing]
    columns = ", ".join(base + optional)

    placeholders = ",".join("?" * len(listing_ids))
    rows = con.execute(
        f"SELECT {columns} FROM listings WHERE listing_id IN ({placeholders})",
        listing_ids,
    ).fetchall()
    con.close()
    return {str(r["listing_id"]): r for r in rows}
