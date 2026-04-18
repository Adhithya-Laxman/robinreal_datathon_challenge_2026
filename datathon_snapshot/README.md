# Datathon preprocessing snapshot

Built 2026-04-18 on the `adhithya/text_embedding_preproc` work,
merged into `main` @ commit `822caa2`.

## Contents

| File | Size | What it is |
|---|---|---|
| `listings.db` | 423 MB | SQLite DB: 22,819 listings + `listing_embeddings` (e5-large, 1024-d) + reverse-geocoded `city` / `canton` for all 11,024 SRED rows + enriched `geo_transit_m` / `geo_supermarket_m` / `geo_school_m` / `geo_university_m` columns on 21,182 listings |
| `bm25.pkl` | 20 MB | Pickled `rank_bm25` index over (title + description), multilingual tokenizer |
| `geo/` | 30 MB | Overpass-API POI response cache (skip if you're not re-running `scripts/geo_enrich.py`) |

## How to use (teammates)

1. Clone the repo & bring up the stack as usual:
   ```bash
   git checkout main && git pull
   docker compose up -d api
   ```

2. Copy these files **into the Docker `data` volume** (not the worktree — `data/` is gitignored and the volume path inside the container is `/data`):
   ```bash
   # from wherever you unpacked this snapshot:
   docker compose cp listings.db  api:/data/listings.db
   docker compose cp bm25.pkl     api:/data/bm25.pkl
   docker compose cp geo          api:/data/geo      # optional
   ```

3. Restart the API so it picks up the new DB/index:
   ```bash
   docker compose restart api
   ```

4. Run a query:
   ```bash
   docker compose exec api python scripts/unified_search.py \
     "bright modern apartment near ETH with a view" \
     --vlm --top-k 20 \
     --json-out results/unified/my_first_query.json
   ```

## Don't have a GPU?

Everything still works on CPU — the `unified_search.py` pipeline auto-detects.
Text-embedding inference is the only step that's noticeably slower on CPU,
but for a single query it's still <1s.

## Want to rebuild from scratch?

All build scripts are idempotent:

```bash
docker compose exec api python scripts/build_text_embeddings.py   # ~10 min CPU / ~3 min GPU
docker compose exec api python scripts/build_bm25_index.py        # <1 min
docker compose exec api python scripts/geo_enrich.py              # ~10 min (Overpass rate-limited)
```
