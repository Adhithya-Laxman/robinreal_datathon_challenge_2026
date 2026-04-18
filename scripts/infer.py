"""Offline inference CLI over the already-built embeddings + BM25 index.

Usage (all run inside the `api` container):

    # semantic search only
    python scripts/infer.py "helle moderne 3-Zimmer Wohnung in Zürich" --mode dense

    # lexical BM25 only
    python scripts/infer.py "appartement lumineux à Genève" --mode bm25

    # hybrid dense+BM25 (default)
    python scripts/infer.py "pet-friendly family home near schools"

    # full pipeline: hard filter + hybrid rerank on the survivors
    python scripts/infer.py "3 Zimmer in Zürich mit Balkon unter 3500 CHF" --mode pipeline

Tweak --top-k, --alpha, and --min-hybrid for tuning.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings


def _fetch_listings(ids: Iterable[str]) -> dict[str, sqlite3.Row]:
    ids = list(ids)
    if not ids:
        return {}
    con = sqlite3.connect(get_settings().db_path)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = con.execute(
        f"""
        SELECT listing_id, title, city, canton, price, rooms, area
        FROM listings
        WHERE listing_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {str(r["listing_id"]): r for r in rows}


def _format_row(score: float, r: sqlite3.Row | None, extras: str = "") -> str:
    if r is None:
        return f"[{score:.4f}] (missing)"
    city = (r["city"] or "?")[:14]
    rooms = str(r["rooms"] if r["rooms"] is not None else "?")
    price = str(r["price"] if r["price"] is not None else "?")
    area = str(r["area"] if r["area"] is not None else "?")
    title = (r["title"] or "")[:70]
    base = f"[{score:.4f}] {city:<14} {rooms:>4} rm  {price:>6} CHF  {area:>5} m2  — {title}"
    return base + (f"  {extras}" if extras else "")


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    mx = max(scores.values())
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def run_dense(query: str, top_k: int) -> list[tuple[str, float]]:
    from app.participant.embeddings import search_by_query_text

    return search_by_query_text(query, top_k=top_k)


def run_bm25(query: str, top_k: int) -> list[tuple[str, float]]:
    from app.participant.bm25_index import load_index

    idx = load_index()
    return idx.search(query, top_k=top_k)


def run_hybrid(query: str, top_k: int, alpha: float) -> list[tuple[str, float, float, float]]:
    """Return [(listing_id, fused, dense, bm25)]. alpha weights dense vs bm25."""
    dense = dict(run_dense(query, top_k=max(top_k * 4, 100)))
    bm25 = dict(run_bm25(query, top_k=max(top_k * 4, 100)))
    dn = _normalize(dense)
    bn = _normalize(bm25)
    ids = set(dn) | set(bn)
    fused = [
        (lid, alpha * dn.get(lid, 0.0) + (1 - alpha) * bn.get(lid, 0.0),
         dn.get(lid, 0.0), bn.get(lid, 0.0))
        for lid in ids
    ]
    fused.sort(key=lambda t: -t[1])
    return fused[:top_k]


def run_pipeline(query: str, top_k: int, alpha: float) -> None:
    from app.participant.embeddings import search_by_query_text
    from app.participant.bm25_index import load_index
    from app.participant.hard_fact_extraction import extract_hard_facts
    from app.harness.search_service import filter_hard_facts

    hard = extract_hard_facts(query)
    print(f"\nHard filter extracted: {hard.model_dump(exclude_none=True)}")

    candidates = filter_hard_facts(get_settings().db_path, hard)
    cand_ids = {c["listing_id"] for c in candidates}
    print(f"Candidates after hard filter (+ relaxation if any): {len(cand_ids)}\n")
    if not cand_ids:
        print("no candidates — widen your query")
        return

    dense_full = dict(search_by_query_text(query, top_k=max(top_k * 4, 200),
                                           candidate_ids=cand_ids))
    bm25_full = {
        lid: s for lid, s in load_index().search(query, top_k=max(top_k * 4, 200))
        if lid in cand_ids
    }
    dn, bn = _normalize(dense_full), _normalize(bm25_full)

    fused = sorted(
        ((lid, alpha * dn.get(lid, 0.0) + (1 - alpha) * bn.get(lid, 0.0),
          dn.get(lid, 0.0), bn.get(lid, 0.0)) for lid in cand_ids),
        key=lambda t: -t[1],
    )[:top_k]

    rows = _fetch_listings(lid for lid, *_ in fused)
    print(f"Top {len(fused)} after hybrid rerank (alpha={alpha}):\n")
    for lid, fused_s, d, b in fused:
        print(_format_row(fused_s, rows.get(lid), extras=f"(d={d:.2f} b={b:.2f})"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid", "pipeline"],
                        default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="weight on dense in hybrid (0..1); bm25 gets 1-alpha")
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(args.query, top_k=args.top_k, alpha=args.alpha)
        return

    if args.mode == "dense":
        hits = run_dense(args.query, top_k=args.top_k)
        rows = _fetch_listings(lid for lid, _ in hits)
        print(f"\nQuery: {args.query}  (mode=dense)\n")
        for lid, s in hits:
            print(_format_row(s, rows.get(lid)))
        return

    if args.mode == "bm25":
        hits = run_bm25(args.query, top_k=args.top_k)
        rows = _fetch_listings(lid for lid, _ in hits)
        print(f"\nQuery: {args.query}  (mode=bm25)\n")
        for lid, s in hits:
            print(_format_row(s, rows.get(lid)))
        return

    fused = run_hybrid(args.query, top_k=args.top_k, alpha=args.alpha)
    rows = _fetch_listings(lid for lid, *_ in fused)
    print(f"\nQuery: {args.query}  (mode=hybrid, alpha={args.alpha})\n")
    for lid, s, d, b in fused:
        print(_format_row(s, rows.get(lid), extras=f"(d={d:.2f} b={b:.2f})"))


if __name__ == "__main__":
    main()
