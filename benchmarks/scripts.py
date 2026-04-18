"""Benchmark generator for the Datathon 2026 listing search challenge.

Usage:
    ANTHROPIC_API_KEY=sk-... python benchmarks/scripts.py
    ANTHROPIC_API_KEY=sk-... python benchmarks/scripts.py --api-url http://localhost:8000 --output benchmarks/benchmarks.json

Produces a JSON file with ~100 benchmark entries, each containing:
  - query: the natural-language search string
  - language: de / fr / it / en
  - hard_filters_expected: structured filters Claude extracted from the query
  - results: ranked listing_ids returned by the search API
  - meta: any metadata returned by the API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import httpx

# ---------------------------------------------------------------------------
# Seed queries from the challenge spec
# ---------------------------------------------------------------------------

SEED_QUERIES: list[str] = [
    "Ich suche eine Wohnung im Raum Zürich, Dübendorf oder Wallisellen, idealerweise 2.5 bis 3.5 Zimmer, ab 70 m², Budget bis 3100 CHF, max 25 Minuten mit dem ÖV bis Stadelhofen, gern mit Balkon, Waschmaschine in der Wohnung oder eigenem Waschturm, und wenn möglich in einer Gegend, die sich ruhig und nicht zu urban hektisch anfühlt.",
    "Wir suchen als Familie zu dritt etwas im Raum Kilchberg, Rüschlikon oder Thalwil, am liebsten nahe am See oder mit schneller Verbindung nach Zürich, mindestens 3.5 Zimmer, ab 90 m², Budget bis 4300 CHF, gern mit Balkon / Terrasse, Lift, Keller, und wichtig wären uns gute Schulen, Parks oder Spielplätze in der Nähe sowie eine Umgebung, in der man sich auch abends sicher fühlt.",
    "I'm looking for an apartment in the greater Zurich area, ideally somewhere like Oerlikon, Altstetten, or Schlieren, with at least 60 sqm, preferably 2 to 3 rooms, a commute under 30 minutes to Zurich HB door to door, and it would be great if the place had a balcony, good light, and access to shops and public transport within walking distance.",
    "We are a family of 3 looking around Basel for something with 2 or 3 bedrooms, ideally 85 sqm or more, budget up to CHF 3500, in an area with good schools, quiet streets, and enough nearby amenities that daily life is easy without needing a car all the time.",
    "Ich suche etwas Kleineres in Lausanne, möglichst in der Nähe von EPFL, gern möbliert, unter 2100 CHF, mit guter Anbindung, und am besten in einer Ecke, die sich sicher, entspannt und nicht komplett anonym anfühlt.",
    "I'm looking for a place near Geneva city center but not right in the busiest part, ideally with 2 bedrooms, budget up to CHF 3600, good transport access, and a neighborhood that feels clean, safe, and a bit more residential than hectic.",
]

# ---------------------------------------------------------------------------
# System prompt for query generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are a dataset generator for a Swiss real-estate search benchmark.

Your task is to generate realistic, diverse natural-language apartment-search queries similar to the seed examples provided.

Rules:
- Cover all four languages: German (de), French (fr), Italian (it), English (en) — roughly 40% German, 25% English, 20% French, 15% Italian.
- Cover a variety of Swiss cities: Zürich, Basel, Bern, Geneva, Lausanne, Winterthur, Lucerne, St. Gallen, Lugano, Zug, Biel, Thun, and suburbs.
- Vary the profile: students, young couples, families, expats, retirees, professionals.
- Vary constraints: price bands (800–6000 CHF), room counts (1–5.5), area (20–180 sqm), features (balcony, parking, pets, elevator, laundry, etc.).
- Mix hard constraints ("max 2500 CHF", "at least 3 rooms") with soft preferences ("bright", "modern kitchen", "quiet street", "close to nature").
- Some queries should be vague; some very specific.
- Keep each query between 30 and 200 words — realistic user text, not a form.

Output a JSON array of objects with keys:
  "query"    : string  — the search query text
  "language" : string  — one of "de", "fr", "it", "en"

Output ONLY the JSON array, no markdown fences, no commentary.
"""

GENERATION_USER_PROMPT = """Here are example queries for inspiration:

{seeds}

Generate {n} additional diverse queries. Return only a JSON array."""


# ---------------------------------------------------------------------------
# Hard-filter extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a structured data extractor for a real-estate search engine.

Given a natural-language apartment query, extract the hard constraints and return them as JSON.

Fields to extract (use null when not mentioned):
{
  "city": [list of city strings] or null,
  "min_price": integer CHF or null,
  "max_price": integer CHF or null,
  "min_rooms": float or null,
  "max_rooms": float or null,
  "min_area": integer sqm or null,
  "max_area": integer sqm or null,
  "features": [list from: "balcony","elevator","parking","garage","fireplace","child_friendly","pets_allowed","temporary","new_build","wheelchair_accessible","private_laundry","minergie_certified"] or null,
  "offer_type": "RENT" or "SALE" or null
}

Output ONLY the JSON object, no markdown fences, no commentary."""


def generate_queries(client: anthropic.Anthropic, n: int = 94) -> list[dict]:
    """Use Claude to generate n diverse benchmark queries."""
    seeds_text = "\n\n".join(f"- {q}" for q in SEED_QUERIES)
    # Generate in two batches of ~47 to stay well under max_tokens
    batch_size = (n + 1) // 2
    results: list[dict] = []
    for batch_num in range(2):
        count = batch_size if batch_num == 0 else n - len(results)
        if count <= 0:
            break
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=GENERATION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": GENERATION_USER_PROMPT.format(seeds=seeds_text, n=count)
                    + (
                        f"\n\nIMPORTANT: This is batch {batch_num + 1}. Generate {count} NEW queries not already in the examples."
                        if batch_num > 0
                        else ""
                    ),
                }
            ],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if Claude wrapped the output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        batch = json.loads(raw)
        results.extend(batch)
        print(f"  → batch {batch_num + 1}: {len(batch)} queries")
    return results


def extract_hard_filters(client: anthropic.Anthropic, query: str) -> dict:
    """Use Claude to extract structured hard filters from a query."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def call_search_api(api_url: str, query: str, limit: int = 25) -> dict:
    """POST /listings and return the parsed response."""
    url = f"{api_url.rstrip('/')}/listings"
    try:
        r = httpx.post(
            url,
            json={"query": query, "limit": limit},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "listings": [], "meta": {}}
    except httpx.RequestError as e:
        return {"error": str(e), "listings": [], "meta": {}}


def build_benchmark_entry(
    query_obj: dict,
    hard_filters: dict,
    api_response: dict,
) -> dict:
    listings = api_response.get("listings", [])
    return {
        "query": query_obj["query"],
        "language": query_obj.get("language", "unknown"),
        "hard_filters_expected": hard_filters,
        "result_listing_ids": [r["listing_id"] for r in listings],
        "results": [
            {
                "listing_id": r["listing_id"],
                "score": r.get("score"),
                "reason": r.get("reason"),
                "city": r.get("listing", {}).get("city"),
                "price_chf": r.get("listing", {}).get("price_chf"),
                "rooms": r.get("listing", {}).get("rooms"),
                "area": r.get("listing", {}).get("living_area_sqm"),
            }
            for r in listings
        ],
        "meta": api_response.get("meta", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate listing search benchmarks.")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running listings API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/benchmarks.json",
        help="Output JSON file path (default: benchmarks/benchmarks.json)",
    )
    parser.add_argument(
        "--n-generate",
        type=int,
        default=94,
        help="Number of additional queries Claude will generate (seed 6 + generated = total)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of results to fetch per query (default: 25)",
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip calling the search API (only generate queries + extract filters)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # -----------------------------------------------------------------------
    # Step 1: generate queries
    # -----------------------------------------------------------------------
    print(f"Generating {args.n_generate} additional queries via Claude…")
    generated = generate_queries(client, n=args.n_generate)
    print(f"  → received {len(generated)} generated queries")

    all_queries: list[dict] = (
        [{"query": q, "language": "de" if "Ich" in q or "Wir" in q else "en"} for q in SEED_QUERIES]
        + generated
    )
    print(f"  → total queries: {len(all_queries)}")

    # -----------------------------------------------------------------------
    # Step 2: extract hard filters + call API for each query
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmarks: list[dict] = []

    for i, query_obj in enumerate(all_queries, 1):
        query = query_obj["query"]
        lang = query_obj.get("language", "unknown")
        print(f"[{i:3d}/{len(all_queries)}] ({lang}) {query[:80]}…")

        # Extract hard filters
        hard_filters = extract_hard_filters(client, query)

        # Call search API
        if args.skip_api:
            api_response: dict = {"listings": [], "meta": {}}
        else:
            api_response = call_search_api(args.api_url, query, limit=args.limit)
            if "error" in api_response:
                print(f"         API error: {api_response['error']}")

        entry = build_benchmark_entry(query_obj, hard_filters, api_response)
        benchmarks.append(entry)

        # Polite rate-limit pause (Haiku is fast but avoid bursting)
        if i % 10 == 0:
            time.sleep(1)

    # -----------------------------------------------------------------------
    # Step 3: write output
    # -----------------------------------------------------------------------
    output_path.write_text(json.dumps(benchmarks, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(benchmarks)} benchmarks → {output_path}")


if __name__ == "__main__":
    main()
