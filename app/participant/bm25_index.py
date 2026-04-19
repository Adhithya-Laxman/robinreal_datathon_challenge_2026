"""BM25 retrieval over listing title + description.

We use `rank_bm25.BM25Okapi` for the scorer and our own small multilingual
tokenizer that handles German / French / Italian / English without heavy
dependencies (no spaCy, no nltk data downloads, no pyicu).

Design:

  * Pre-tokenized corpus + BM25 model are built once offline by
    `scripts/build_bm25_index.py` and pickled to `/data/bm25.pkl`.
  * Online, we lazy-load the pickle, tokenize the query the same way, and
    return `[(listing_id, score), ...]`.
  * We deliberately skip language detection per document — applying the
    German stemmer uniformly is good enough for a hackathon and avoids the
    complexity of the `langdetect` sidecar library.

If you later want per-language stemming, swap in `snowballstemmer.language`
based on `detect(text)` during `build_tokens`.
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
import snowballstemmer

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_INDEX_FILENAME = "bm25.pkl"

# Lightweight multilingual stopwords — only obvious function words.
# We keep the list small on purpose; aggressive stopwording hurts BM25 on
# short listing titles.
_STOPWORDS: frozenset[str] = frozenset(
    # German
    "der die das den dem des ein eine einer eines einem einen und oder aber "
    "auch als am im in an auf bei mit für zu von aus über unter vor nach "
    "ist sind war waren wird werden wurde wurden sein haben hat hatte "
    "nicht kein keine mehr sehr noch schon nur ganz sowie sowie."
    # French
    " le la les un une des du de au aux et ou mais donc car dans sur sous "
    "pour par avec sans chez est sont était étaient sera seront "
    "ne pas plus très bien aussi encore déjà"
    # Italian
    " il lo la i gli le un uno una e o ma perché che di da in su per con "
    "senza presso è sono era erano sarà saranno non più molto bene anche"
    # English
    " the a an and or but if for to from of at in on with without by is are "
    "was were be been being this that these those it its".split()
)

_TOKEN_RE = re.compile(r"[\w\u00C0-\u024F]+", re.UNICODE)

# Language-keyword sets used for heuristic detection (see detect_lang).
_LANG_KEYWORDS: dict[str, frozenset[str]] = {
    "de": frozenset(["der", "die", "das", "und", "mit", "ist", "eine", "nicht", "für", "bei", "im", "am", "zum"]),
    "fr": frozenset(["les", "des", "pour", "dans", "avec", "est", "qui", "une", "sur", "pas", "par", "du", "au"]),
    "it": frozenset(["con", "nel", "per", "del", "della", "sono", "una", "gli", "che", "dal", "alle", "dei", "si"]),
    "en": frozenset(["the", "and", "for", "with", "this", "that", "are", "not", "from", "have", "apartment", "rent"]),
}
_LANG_CHARS: dict[str, frozenset[str]] = {
    "de": frozenset("äöüß"),
    "fr": frozenset("éèêëàâùûîôçœ"),
    "it": frozenset("àèìîòùú"),
}

# Single stemmer instance reused across threads — snowballstemmer stemmers
# are not documented as thread-safe, so we guard with a lock.
_stemmer = snowballstemmer.stemmer("german")
_stemmer_lock = threading.Lock()


def detect_lang(text: str) -> str:
    """Heuristic language detection (de/fr/it/en) without external libraries.

    Uses character markers and common-word frequency. Defaults to 'de'
    (most common in Swiss listings) when ambiguous.
    """
    lower = text.lower()
    words = frozenset(_TOKEN_RE.findall(lower))
    word_scores = {lang: len(words & kws) for lang, kws in _LANG_KEYWORDS.items()}
    char_scores = {lang: sum(lower.count(c) for c in chars)
                   for lang, chars in _LANG_CHARS.items()}
    total = {lang: word_scores[lang] * 3 + char_scores.get(lang, 0)
             for lang in _LANG_KEYWORDS}
    best = max(total, key=total.get)
    return best if total[best] > 0 else "de"


def tokenize(text: str) -> list[str]:
    """Multilingual-ish tokenizer: lowercase, split on non-word, stem with DE."""
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    if not tokens:
        return []
    filtered = [t for t in tokens if t not in _STOPWORDS]
    if not filtered:
        return []
    with _stemmer_lock:
        return _stemmer.stemWords(filtered)


# ---- Index on-disk format ------------------------------------------------


@dataclass(slots=True)
class BM25Index:
    listing_ids: list[str]
    bm25: BM25Okapi
    avg_doc_len: float

    def score_query(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.listing_ids), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        candidate_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        scores = self.score_query(query)
        if candidate_ids is not None:
            wanted = set(candidate_ids)
            mask = np.array([lid in wanted for lid in self.listing_ids], dtype=bool)
            scores = np.where(mask, scores, -np.inf)
        # Normalize non-negative scores to [0, 1] via max for easy fusion with
        # cosine sim from the embedding side; `-inf` masked entries survive.
        finite = scores[np.isfinite(scores)]
        if finite.size and finite.max() > 0:
            norm = scores / finite.max()
        else:
            norm = scores
        top = np.argsort(-scores)[:top_k]
        return [
            (self.listing_ids[i], float(norm[i]))
            for i in top
            if np.isfinite(scores[i])
        ]


def default_index_path() -> Path:
    settings = get_settings()
    return settings.db_path.parent / DEFAULT_INDEX_FILENAME


# ---- Build + load --------------------------------------------------------


def build_index(listing_ids: Sequence[str], documents: Sequence[str]) -> BM25Index:
    tokenized = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    lengths = [len(t) for t in tokenized]
    avg = float(np.mean(lengths)) if lengths else 0.0
    return BM25Index(listing_ids=list(listing_ids), bm25=bm25, avg_doc_len=avg)


def save_index(index: BM25Index, path: Path | None = None) -> Path:
    path = path or default_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(
            {
                "listing_ids": index.listing_ids,
                "bm25": index.bm25,
                "avg_doc_len": index.avg_doc_len,
                "version": 1,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info("Wrote BM25 index (%d listings) to %s", len(index.listing_ids), path)
    return path


_load_lock = threading.Lock()
_cached: tuple[BM25Index, float] | None = None  # (index, mtime)


def load_index(path: Path | None = None) -> BM25Index:
    global _cached
    path = path or default_index_path()
    mtime = path.stat().st_mtime

    with _load_lock:
        if _cached is not None and _cached[1] == mtime:
            return _cached[0]
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        index = BM25Index(
            listing_ids=payload["listing_ids"],
            bm25=payload["bm25"],
            avg_doc_len=payload["avg_doc_len"],
        )
        _cached = (index, mtime)
        logger.info("Loaded BM25 index (%d listings) from %s", len(index.listing_ids), path)
        return index


def index_exists(path: Path | None = None) -> bool:
    path = path or default_index_path()
    return path.exists()
