#!/usr/bin/env python3
"""Smoke-test the three offline-text pipelines: query understanding,
embeddings, BM25. Safe to run before Bedrock creds are wired up — it
reports which stages are ready and which aren't.

Usage:

    docker compose exec api python scripts/smoke_text_pipeline.py

Exit codes:
  0  every stage that *can* run did run and returned sensible output
  1  a stage that should have worked crashed
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.participant import bedrock_client, embeddings, query_understanding
from app.participant.bm25_index import default_index_path, index_exists, load_index

logger = logging.getLogger("smoke_text_pipeline")


EXAMPLE_QUERIES = [
    "Bright 3.5-room apartment in Zürich under CHF 2500, ideally with balcony, close to ETH.",
    "Ruhige, moderne Wohnung in Bern mit Balkon, nicht zu teuer.",
    "Appartement calme de 4 pièces à Lausanne, proche des écoles.",
]


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    failures = 0

    _section("Environment")
    print(f"  db_path                 = {settings.db_path}  (exists={settings.db_path.exists()})")
    print(f"  bedrock_region          = {settings.bedrock_region}")
    print(f"  bedrock_available       = {bedrock_client.bedrock_available()}")
    print(f"  force_fallback          = {settings.force_bedrock_fallback}")
    print(f"  embedding_model         = {settings.bedrock_embedding_model_id}")
    print(f"  query_understanding     = {settings.bedrock_query_understanding_model_id}")

    _section("Query understanding (LLM if available, else heuristic fallback)")
    for q in EXAMPLE_QUERIES:
        try:
            qu = query_understanding.understand(q)
        except Exception as exc:
            failures += 1
            print(f"  ERROR on {q!r}: {exc}")
            continue
        print()
        print(f"  query         : {q}")
        print(f"  used_llm      : {qu.used_llm}")
        print(f"  language      : {qu.language}")
        print(f"  interpretation: {qu.interpretation}")
        print("  hard          :")
        print(_indent(json.dumps(qu.hard.model_dump(exclude_none=True), indent=2), 4))
        print("  soft          :")
        print(_indent(json.dumps(qu.soft.model_dump(exclude_none=True), indent=2), 4))

    _section("BM25 index")
    bm25_path = default_index_path()
    if not index_exists(bm25_path):
        print(f"  (no index yet at {bm25_path})")
        print("  run: docker compose exec api python scripts/build_bm25_index.py")
    else:
        try:
            index = load_index(bm25_path)
            print(f"  loaded {len(index.listing_ids)} listings, avg_doc_len={index.avg_doc_len:.1f}")
            for q in EXAMPLE_QUERIES[:2]:
                hits = index.search(q, top_k=5)
                print(f"  top-5 for {q!r}:")
                for lid, score in hits:
                    print(f"    {score:6.3f}  {lid}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR loading BM25 index: {exc}")

    _section("Text embeddings vector store")
    n = embeddings.count_embeddings()
    print(f"  embeddings stored for model={settings.bedrock_embedding_model_id}: {n}")
    if n == 0:
        print("  run: docker compose exec api python scripts/build_text_embeddings.py")
    else:
        if not bedrock_client.bedrock_available():
            print("  (skipping query embedding — Bedrock not configured)")
        else:
            try:
                hits = embeddings.search_by_query_text(EXAMPLE_QUERIES[0], top_k=5)
                print(f"  top-5 cosine for {EXAMPLE_QUERIES[0]!r}:")
                for lid, score in hits:
                    print(f"    {score:6.3f}  {lid}")
            except Exception as exc:
                failures += 1
                print(f"  ERROR embedding/searching: {exc}")

    _section("Summary")
    if failures == 0:
        print("  OK — nothing crashed. Stages not yet built are clearly marked above.")
        return 0
    print(f"  {failures} stage(s) crashed. See log above.")
    return 1


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
