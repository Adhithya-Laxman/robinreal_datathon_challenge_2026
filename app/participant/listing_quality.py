"""Post-hard-filter listing quality checks.

The hard SQL filter catches structural mismatches (wrong city, wrong price,
wrong room count). This module catches a different failure mode: listings
that are *in the right city and price range* but shouldn't be shown at all
for a typical rental query because they are

  * apartment SWAPS ("Tauschwohnung — suche grössere"),
  * WANTED ads ("Cherche appartement", "Suche Wohnung"),
  * foreign-country listings mis-tagged with a Swiss city (e.g. "FR-Ferney-
    Voltaire" shown for Geneva), or
  * TEMPORARY / short-term leases when the user clearly wants a stable
    long-term home (family, relocating expat, etc.).

These all showed up as 0/5 or 1/5 in the LLM-judge eval even though every
numeric hard filter matched. The judge was right: the listing simply wasn't
an answer to the user's question.

Design:
  * Hard drops (always remove, regardless of query):
      - swap / wanted listings   (these are never offer listings)
      - non-Swiss listings       (the dataset is CH-only by spec)
  * Soft drops (remove only when the query implies long-term intent):
      - "Befristet" / "temporaire" / "until YYYY" listings
  * Everything is a pure string-pattern check on title + description + a
    couple of structured fields. No ML, no external calls. If we were
    wrong once in a hundred, we'd notice: erring on the side of dropping
    a weird listing is almost always correct here.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.participant.schemas import SoftPreferences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Detects "I'm offering to SWAP my apartment" listings. These match rental
# queries semantically because the lister describes their current place AND
# what they want — but they are not an actual offer.
#
# Rule of thumb: the word "Tausch" (swap) in DE, "échange" in FR, "scambio"
# in IT is a near-certain negative. We also catch "[Tauschwohnung]" which
# Comparis tags explicitly.
_SWAP_PATTERNS = re.compile(
    r"("
    r"\btauschwohnung\b"
    r"|\btausch\s*gegen\b"
    r"|\bzum\s+tausch\b"
    r"|\bwohnungstausch\b"
    r"|\bapartment\s+swap\b"
    r"|\b[ée]change\s+(?:d['a]|contre)\b"
    r"|\bappartement\s+[ée]change\b"
    r"|\bscambio\s+(?:di\s+)?appartament"
    r")",
    re.IGNORECASE,
)

# Detects "I'm LOOKING FOR an apartment" ads (seeker / wanted). These
# trigger dense + BM25 heavily because the seeker describes exactly what
# the user wants (bedrooms, location, etc.) — but it's a want, not an
# offer. Very high precision, very easy to catch by title prefix.
_WANTED_PATTERNS = re.compile(
    r"("
    # DE
    r"\bsuche\s+(?:wohnung|\d[\.,]?\d?\s*[-\s]?zimmer|appartement|studio)\b"
    r"|\bich\s+suche\b"
    r"|\bwir\s+suchen\b"
    # FR
    r"|\bcherche\s+(?:appartement|studio|\d\s*pièces|sous[\s-]?location)\b"
    r"|\ba\s+la\s+recherche\s+d['a]\s*(?:un\s+)?appartement"
    r"|\breprise\s+de\s+bail"
    # IT
    r"|\bcerco\s+(?:appartament|monolocale|bilocale|\d[\.,]?\d?\s*local)"
    # EN
    r"|\bwanted\b\s*(?::|-|apartment)"
    r"|\blooking\s+for\s+(?:an?\s+)?(?:apartment|flat|studio)"
    r")",
    re.IGNORECASE,
)

# Foreign-country listings. The dataset is CH-only, but a few listings
# with city='Genf' are actually in FR (e.g. Ferney-Voltaire) or DE (e.g.
# Weil am Rhein). Those have explicit country prefixes in the title or
# addresses with foreign postal codes (5-digit for DE, 5-digit for FR
# but *not* Swiss 4-digit).
_NON_SWISS_TITLE = re.compile(
    r"(?:"
    r"\bFR[- ](?:Ferney|Gaillard|Annemasse|Thonon|St[\s-]?Julien|Annecy)"
    r"|\bDE[- ](?:Weil|Lörrach|Freiburg|Konstanz|Singen|Waldshut)"
    r"|\bIT[- ](?:Como|Varese|Chiasso|Milano|Bergamo)"
    r"|\b(?:France|Germany|Italy)\b"
    r"|\bFerney[\s-]?Voltaire\b"
    r"|\bGaillard\b"
    r")",
    re.IGNORECASE,
)

# "Temporary" / "short-term" / "until DATE" indicators. We only want to
# drop these when the user implied a stable long-term need.
_TEMPORARY_PATTERNS = re.compile(
    r"("
    r"\bbefristet\b"
    r"|\btemporäre?\b"
    r"|\btemporar(?:y|ily)\b"
    r"|\btemporair(?:e|ement)\b"
    r"|\btemporan(?:e|ea|eamente)\b"
    r"|\bzwischenmiete\b"
    r"|\buntervermietung\b"
    r"|\bsous[\s-]?location\b"
    r"|\bsublet\b"
    r"|\babriss\b"
    r"|\bdemoli(?:r|tion|zione)"
    r"|\bbis\s+(?:zum\s+)?\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\bbis\s+\w+\s+\d{4}\b"
    r"|\bjusqu['\s]*(?:au|en)\s+\w+\s*\d{2,4}\b"
    r"|\bfino\s+al\s+\d"
    r")",
    re.IGNORECASE,
)

# Hints in the user's query that they want a stable long-term rental.
# When any of these fire we treat "Befristet" listings as drops, not just
# as a down-rank.
_LONG_TERM_QUERY_PATTERNS = re.compile(
    r"("
    r"\bfamil(?:y|ie|le|ia)"
    r"|\bkids?\b|\bchildren\b|\benfants?\b|\bkinder\b|\bbambin"
    r"|\brelocat(?:e|ing|ion)\b|\bumzug\b|\bziehen\s+nach\b"
    r"|\bstable\s+(?:housing|home|rental)"
    r"|\blong[\s-]?term\b"
    r"|\bpermanent\b|\bdauerhaft\b"
    r"|\brentner|\bretired?\b|\bretrait[ée]"
    r"|\bpension(?:ier|é)"
    r")",
    re.IGNORECASE,
)

# Hints that the user WANTS a temporary / furnished short stay. When these
# fire we do NOT drop temporary listings (that would remove the right
# answer — see eval Q4 where the user explicitly asked for ~6 months).
_SHORT_TERM_QUERY_PATTERNS = re.compile(
    r"("
    r"\btempor[äae]r"
    r"|\btemporair"
    r"|\bshort[\s-]?term\b"
    r"|\bfew\s+months\b|\bfor\s+\d+\s+months?\b|\bpour\s+\d+\s+mois\b"
    r"|\bbefristet\b|\bzwischenmiete\b|\buntervermietung\b|\bsous[\s-]?location\b"
    r"|\bsublet\b"
    r"|\bmöbl?iert\b|\bfurnished\b|\bmeubl[ée]\b|\barredat"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classifiers on one listing row
# ---------------------------------------------------------------------------


def _row_text(row: dict[str, Any]) -> str:
    """Concatenate the searchable text fields of a row, stripping None."""
    parts = [row.get("title") or "", row.get("description") or ""]
    return " ".join(p for p in parts if p)


def is_swap_or_wanted(row: dict[str, Any]) -> tuple[bool, str]:
    """Return (drop, reason) — True if this is a swap or wanted ad."""
    text = _row_text(row)
    if _SWAP_PATTERNS.search(text):
        return True, "swap_listing"
    if _WANTED_PATTERNS.search(text):
        return True, "wanted_listing"
    return False, ""


def is_non_swiss(row: dict[str, Any]) -> tuple[bool, str]:
    """Return (drop, reason) — True if this listing is actually abroad.

    We only flag on high-precision signals: explicit country prefixes in
    the title (FR-Ferney-Voltaire, DE-Weil am Rhein) or the city name
    itself being a well-known foreign border town. The dataset stores
    Swiss 4-digit postal codes but we don't check postal code here
    because Swiss + foreign codes can collide numerically.
    """
    title = row.get("title") or ""
    if _NON_SWISS_TITLE.search(title):
        return True, "non_swiss_title"
    return False, ""


def is_short_term(row: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_temporary, reason) — listing is a temporary/short-term lease."""
    text = _row_text(row)
    m = _TEMPORARY_PATTERNS.search(text)
    if m:
        # Report the actual matched phrase so the drop is explainable in logs.
        return True, f"temporary:{m.group(0).lower().strip()}"
    return False, ""


# ---------------------------------------------------------------------------
# Query-level intent
# ---------------------------------------------------------------------------


def query_wants_short_term(query: str, soft: SoftPreferences | None = None) -> bool:
    """True when the user explicitly asked for a temporary / furnished stay."""
    if soft is not None:
        if soft.furnished is True or soft.student_friendly is True:
            # "student" and "furnished" are not hard proof of short-term,
            # but combined with any short-term keyword we'd already hit
            # below. Keep this path as a weak positive.
            pass
        if soft.availability_hint:
            text = soft.availability_hint.lower()
            if _SHORT_TERM_QUERY_PATTERNS.search(text):
                return True
    return bool(_SHORT_TERM_QUERY_PATTERNS.search(query or ""))


def query_wants_long_term(query: str, soft: SoftPreferences | None = None) -> bool:
    """True when the user implied a stable long-term rental.

    Only ever True when `query_wants_short_term` is False; we never treat
    a query as simultaneously wanting both.
    """
    if query_wants_short_term(query, soft):
        return False
    return bool(_LONG_TERM_QUERY_PATTERNS.search(query or ""))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def apply_post_filter(
    candidates: list[dict[str, Any]],
    *,
    query: str,
    soft: SoftPreferences | None = None,
    min_keep: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop low-quality listings, return (kept, drop_counts).

    Drops:
      * swap / wanted ads                         — ALWAYS
      * non-Swiss listings                        — ALWAYS
      * Befristet / temporary                     — only when user implied
                                                    long-term; skipped
                                                    entirely when user
                                                    asked for short-term
    Safety: if post-filtering would leave fewer than `min_keep` candidates,
    we only apply the ALWAYS-drops (swap, wanted, non-swiss) and skip the
    short-term drop, so we never kill recall just to enforce a preference.

    `drop_counts` is a per-reason dict useful for logging/debug.
    """
    if not candidates:
        return candidates, {}

    want_short = query_wants_short_term(query, soft)
    want_long = query_wants_long_term(query, soft)

    drop_counts: dict[str, int] = {}
    # Stage A: hard drops (always-on).
    stage_a: list[dict[str, Any]] = []
    for row in candidates:
        swap, s_reason = is_swap_or_wanted(row)
        if swap:
            drop_counts[s_reason] = drop_counts.get(s_reason, 0) + 1
            continue
        foreign, f_reason = is_non_swiss(row)
        if foreign:
            drop_counts[f_reason] = drop_counts.get(f_reason, 0) + 1
            continue
        stage_a.append(row)

    # Stage B: short-term drops (only if user wants long-term).
    if not want_long or want_short:
        if want_short:
            drop_counts["_short_term_kept_by_intent"] = len([
                r for r in stage_a if is_short_term(r)[0]
            ])
        return stage_a, drop_counts

    stage_b: list[dict[str, Any]] = []
    short_term_rows: list[dict[str, Any]] = []
    for row in stage_a:
        temp, t_reason = is_short_term(row)
        if temp:
            short_term_rows.append(row)
            drop_counts[t_reason] = drop_counts.get(t_reason, 0) + 1
        else:
            stage_b.append(row)

    # Recall safety: if we've now dropped below `min_keep` and we had
    # enough before, put the short-term listings back (they are still
    # better than nothing, and the ranker will down-rank them).
    if len(stage_b) < min_keep and short_term_rows:
        # Recover shortest available slate, but log that we did.
        logger.info(
            "[post_filter] short-term drop would leave %d candidates "
            "(< min_keep=%d); restoring %d short-term listings",
            len(stage_b), min_keep, len(short_term_rows),
        )
        drop_counts["_short_term_restored"] = len(short_term_rows)
        # Remove the short-term counts since we restored them.
        for k in list(drop_counts.keys()):
            if k.startswith("temporary:"):
                drop_counts.pop(k, None)
        return stage_a, drop_counts

    return stage_b, drop_counts


__all__ = [
    "apply_post_filter",
    "is_non_swiss",
    "is_short_term",
    "is_swap_or_wanted",
    "query_wants_long_term",
    "query_wants_short_term",
]
