"""Text embeddings with two swappable backends.

Backends, in order of preference:

  1. **Bedrock Cohere embed-multilingual-v3** (1024-d) — our default when
     Bedrock is reachable.
  2. **Local BAAI/bge-m3 via fastembed** (1024-d, multilingual) — used
     when Bedrock is blocked or `USE_LOCAL_EMBEDDINGS=1`. First call
     downloads ~2.3 GB of model weights into `FASTEMBED_CACHE_PATH`.

Both produce 1024-d vectors and we store them in the same SQLite blob
format, keyed on `(listing_id, model_id)`. Swapping backends mid-project
just means re-running the offline script — it's idempotent.

Query path:
  * `embed_query(text)` — single embedding for an online query.
  * `embed_documents(texts)` — batched embeddings for offline ingestion.

Also exposes a lightweight on-disk vector store — a single SQLite table
`listing_embeddings` keyed by `listing_id`, storing the raw float32 blob.
We deliberately avoid FAISS/sqlite-vec for now: 22,819 × 1024 × 4B ≈ 94 MB,
which fits in RAM easily and makes cosine search a single numpy matmul.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.config import get_settings
from app.db import get_connection
from app.participant.bedrock_client import bedrock_available, get_bedrock_client

logger = logging.getLogger(__name__)

EMBED_DIM = 1024


# ---- Bedrock Cohere wrapper ----------------------------------------------


def embed_query(text: str) -> np.ndarray:
    """Embed a single user query. Returns (EMBED_DIM,) float32 ndarray."""
    return _embed([text], input_type="search_query")[0]


def embed_documents(texts: Sequence[str]) -> np.ndarray:
    """Embed a batch of listing documents. Returns (N, EMBED_DIM) float32."""
    return _embed(list(texts), input_type="search_document")


def active_model_id() -> str:
    """Return the model_id that will be written alongside new embeddings.

    Exposed so offline scripts can store + look-up vectors under the right
    key even when the active backend changes mid-project.
    """
    settings = get_settings()
    if _should_use_local(settings):
        return f"local:{settings.local_embedding_model}"
    return settings.bedrock_embedding_model_id


def _should_use_local(settings) -> bool:
    if settings.use_local_embeddings:
        return True
    return not bedrock_available(settings)


def _embed(texts: list[str], *, input_type: str) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    settings = get_settings()

    if _should_use_local(settings):
        return _embed_local(texts, input_type=input_type)

    try:
        return _embed_bedrock(texts, input_type=input_type)
    except Exception as exc:
        logger.warning(
            "Bedrock embedding failed (%s). Falling back to local model %s.",
            exc,
            settings.local_embedding_model,
        )
        return _embed_local(texts, input_type=input_type)


def _embed_bedrock(texts: list[str], *, input_type: str) -> np.ndarray:
    settings = get_settings()
    client = get_bedrock_client()

    all_vectors: list[np.ndarray] = []
    # Cohere v3 accepts up to 96 texts per call.
    for chunk_start in range(0, len(texts), 96):
        chunk = texts[chunk_start : chunk_start + 96]
        body = {
            "texts": chunk,
            "input_type": input_type,
            "truncate": "END",
        }
        response = client.invoke_model(
            modelId=settings.bedrock_embedding_model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
        if vectors.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"Unexpected embedding dim {vectors.shape[1]} "
                f"(expected {EMBED_DIM}) from {settings.bedrock_embedding_model_id}"
            )
        all_vectors.append(_l2_normalize(vectors))

    return np.vstack(all_vectors)


# ---- Local backend (fastembed + BGE-M3) ----------------------------------


_local_model_lock = threading.Lock()
_local_model = None  # type: ignore[assignment]


def _get_local_model():
    """Lazy-load the fastembed model. Downloads weights on first call.

    Uses CUDA when available and falls back to CPU with a warning. Onnxruntime
    decides per-provider at session-create time, so passing both providers is
    safe on CPU-only boxes too.
    """
    global _local_model
    if _local_model is not None:
        return _local_model
    with _local_model_lock:
        if _local_model is not None:
            return _local_model
        from fastembed import TextEmbedding

        settings = get_settings()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info(
            "Loading local embedding model %s (providers=%s, downloads on first use)...",
            settings.local_embedding_model,
            providers,
        )
        try:
            _local_model = TextEmbedding(
                model_name=settings.local_embedding_model,
                providers=providers,
            )
        except (TypeError, ValueError):
            logger.warning("fastembed: requested providers unavailable, falling back to defaults.")
            _local_model = TextEmbedding(model_name=settings.local_embedding_model)
        logger.info("Local embedding model ready.")
        return _local_model


def _embed_local(texts: list[str], *, input_type: str) -> np.ndarray:
    """Local fastembed backend.

    E5-family models require `"query: "` / `"passage: "` prefixes to match
    training. BGE-family doesn't. We detect by model name — conservative
    and cheap.
    """
    settings = get_settings()
    prefix = ""
    name = settings.local_embedding_model.lower()
    if "e5" in name:
        prefix = "query: " if input_type == "search_query" else "passage: "

    prepared = [prefix + t for t in texts] if prefix else list(texts)

    model = _get_local_model()
    vectors_iter = model.embed(prepared, batch_size=256)
    arr = np.asarray(list(vectors_iter), dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Local embed returned shape {arr.shape}")
    return _l2_normalize(arr)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype(np.float32, copy=False)


# ---- On-disk vector store ------------------------------------------------


VECTOR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS listing_embeddings (
    listing_id  TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_listing_embeddings_model
    ON listing_embeddings(model_id);
"""


def ensure_vector_table(db_path: Path) -> None:
    with get_connection(db_path) as con:
        con.executescript(VECTOR_TABLE_SQL)


def upsert_embeddings(
    db_path: Path,
    *,
    listing_ids: Sequence[str],
    vectors: np.ndarray,
    model_id: str,
) -> None:
    if len(listing_ids) != len(vectors):
        raise ValueError("listing_ids and vectors must be the same length")
    rows = [
        (
            str(listing_id),
            model_id,
            int(vec.shape[0]),
            vec.astype(np.float32, copy=False).tobytes(),
        )
        for listing_id, vec in zip(listing_ids, vectors)
    ]
    with get_connection(db_path) as con:
        con.executemany(
            """
            INSERT INTO listing_embeddings (listing_id, model_id, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(listing_id) DO UPDATE SET
                model_id   = excluded.model_id,
                dim        = excluded.dim,
                vector     = excluded.vector,
                created_at = datetime('now')
            """,
            rows,
        )


def listing_ids_missing_embedding(
    db_path: Path,
    *,
    model_id: str,
) -> list[str]:
    with get_connection(db_path) as con:
        rows = con.execute(
            """
            SELECT l.listing_id
            FROM listings l
            LEFT JOIN listing_embeddings e
              ON e.listing_id = l.listing_id AND e.model_id = ?
            WHERE e.listing_id IS NULL
            """,
            [model_id],
        ).fetchall()
    return [str(r["listing_id"]) for r in rows]


# ---- In-memory index for online cosine search ----------------------------


@dataclass(slots=True)
class _VectorIndex:
    listing_ids: list[str]
    matrix: np.ndarray  # (N, EMBED_DIM) float32, rows unit-normed
    model_id: str


_index_lock = threading.Lock()
_cached_index: _VectorIndex | None = None
_cached_index_mtime: float | None = None


def _db_mtime(db_path: Path) -> float | None:
    try:
        return db_path.stat().st_mtime
    except FileNotFoundError:
        return None


def _load_index(db_path: Path, *, model_id: str) -> _VectorIndex:
    with get_connection(db_path) as con:
        rows = con.execute(
            "SELECT listing_id, vector FROM listing_embeddings WHERE model_id = ?",
            [model_id],
        ).fetchall()

    if not rows:
        return _VectorIndex(listing_ids=[], matrix=np.empty((0, EMBED_DIM), dtype=np.float32), model_id=model_id)

    listing_ids = [str(r["listing_id"]) for r in rows]
    matrix = np.stack(
        [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
    ).astype(np.float32, copy=False)
    return _VectorIndex(listing_ids=listing_ids, matrix=matrix, model_id=model_id)


def get_index(db_path: Path | None = None, *, model_id: str | None = None) -> _VectorIndex:
    """Return a cached in-memory index. Reload if the DB mtime changed."""
    global _cached_index, _cached_index_mtime

    settings = get_settings()
    db_path = db_path or settings.db_path
    model_id = model_id or active_model_id()

    with _index_lock:
        mtime = _db_mtime(db_path)
        if (
            _cached_index is not None
            and _cached_index.model_id == model_id
            and mtime == _cached_index_mtime
        ):
            return _cached_index
        _cached_index = _load_index(db_path, model_id=model_id)
        _cached_index_mtime = mtime
        logger.info(
            "Loaded %d embeddings for model=%s from %s",
            len(_cached_index.listing_ids),
            model_id,
            db_path,
        )
        return _cached_index


def search_by_query_text(
    query: str,
    *,
    top_k: int = 50,
    db_path: Path | None = None,
    candidate_ids: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Embed `query` and return [(listing_id, cosine_similarity), ...] sorted desc.

    If `candidate_ids` is given, restrict search to that subset (typical usage
    after hard filtering).
    """
    idx = get_index(db_path)
    if idx.matrix.shape[0] == 0:
        return []

    qvec = embed_query(query)
    sims = idx.matrix @ qvec  # (N,) — both sides are unit-normed, so this is cosine.

    if candidate_ids is not None:
        wanted = set(candidate_ids)
        mask = np.array([lid in wanted for lid in idx.listing_ids], dtype=bool)
        sims = np.where(mask, sims, -np.inf)

    top = np.argsort(-sims)[:top_k]
    return [
        (idx.listing_ids[i], float(sims[i]))
        for i in top
        if np.isfinite(sims[i])
    ]


def count_embeddings(db_path: Path | None = None, *, model_id: str | None = None) -> int:
    settings = get_settings()
    db_path = db_path or settings.db_path
    model_id = model_id or active_model_id()
    try:
        with get_connection(db_path) as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM listing_embeddings WHERE model_id = ?",
                [model_id],
            ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["n"]) if row else 0
