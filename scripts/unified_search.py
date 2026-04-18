"""Unified end-to-end search pipeline, CLI entry point.

Runs:
    query -> understanding -> hard filter -> dense + bm25 [+ vlm]
          -> geo proximity -> price band -> weighted fusion -> top-k

Usage (inside the api container):

    # text-only pipeline, printed to stdout
    python scripts/unified_search.py "helle 3-Zimmer Wohnung in Zürich unter 3000 CHF"

    # include VLM image scoring (slower first call; needs features_vlm/ shards)
    python scripts/unified_search.py "bright modern apartment near ETH" --vlm

    # dump a COMPREHENSIVE JSON for analysis (full descriptions, images, signals)
    python scripts/unified_search.py "family home with garden" \\
        --vlm --json-out results/unified/run_01.json

    # tune how much dense vs bm25 vs vlm contribute
    python scripts/unified_search.py "appartement lumineux" --vlm \\
        --weight dense=0.3 --weight bm25=0.2 --weight vlm=0.3 \\
        --weight geo_transit=0.1 --weight price_band=0.1

The JSON file contains, for every result, the full listing row (title,
description, images, all distance fields), the per-signal scores and
weighted contributions, and the top-3 signals that drove its rank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.participant.unified_ranker import (  # noqa: E402
    UnifiedResponse,
    unified_search,
)


def _parse_weight_overrides(raw: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--weight must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError as exc:
            raise SystemExit(f"invalid weight {item!r}: {exc}") from exc
    return out


def _print_header(resp: UnifiedResponse) -> None:
    hard_dump = resp.hard.model_dump(exclude_none=True)
    for k in ("limit", "offset"):
        hard_dump.pop(k, None)
    soft_dump = resp.understanding.soft.model_dump(
        exclude_defaults=True, exclude_none=True,
    )
    soft_dump.pop("weights", None)

    print("=" * 100)
    print(f"Query:          {resp.query}")
    print(f"Language:       {resp.understanding.language}")
    print(f"LLM used:       {resp.understanding.used_llm}")
    if resp.understanding.interpretation:
        print(f"Interpretation: {resp.understanding.interpretation}")
    print()
    print("Hard filters (MUST hold):")
    print(f"  {json.dumps(hard_dump, ensure_ascii=False) if hard_dump else '(none)'}")
    print("Soft preferences (ranking hints):")
    print(f"  {json.dumps(soft_dump, ensure_ascii=False) if soft_dump else '(none)'}")
    print()
    if resp.llm_weight_hints:
        hint_parts = ", ".join(
            f"{k}={v:.2f}" for k, v in resp.llm_weight_hints.items()
        )
        print(f"LLM weight hints (raw): {hint_parts}")
    active = ", ".join(
        f"{k}={v:.2f}" for k, v in resp.weights_used.items() if v > 0.0
    )
    print(f"Signal weights (L1-normalized, post-mapping): {active or '(none)'}")
    print(f"Candidates after hard filter (+ relaxation): {resp.candidates_before_rerank}")
    print("=" * 100)


def _print_stages(resp: UnifiedResponse) -> None:
    """Show per-stage execution trace so silent failures are visible."""
    if not resp.stages:
        return
    print("\nPipeline stages:")
    icons = {"ok": "[OK  ]", "empty": "[EMP ]", "skipped": "[SKIP]",
             "missing": "[MISS]", "error": "[ERR ]"}
    for s in resp.stages:
        icon = icons.get(s.state, f"[{s.state}]")
        scored = f" n={s.scored}" if s.scored else ""
        detail = f"  {s.detail}" if s.detail else ""
        print(f"  {icon} {s.name:<20}{scored:<8}{detail}")
    print()


def _print_results(resp: UnifiedResponse) -> None:
    if not resp.results:
        print("\n(no results - try widening your query)")
        return

    print(f"\nTop {len(resp.results)} listings:\n")
    for rank, r in enumerate(resp.results, 1):
        row = r.row
        lid = r.listing_id
        city = (row.get("city") or "?")[:14]
        price = str(row.get("price")) if row.get("price") is not None else "?"
        rooms = str(row.get("rooms")) if row.get("rooms") is not None else "?"
        area_raw = row.get("area")
        area = str(int(area_raw)) if area_raw is not None else "?"
        title = (row.get("title") or "")[:80]
        offer = row.get("offer_type") or ""

        # Geo annotations
        extras: list[str] = []
        for key, label in (
            ("geo_transit_m", "transit"),
            ("geo_school_m", "school"),
            ("geo_university_m", "uni"),
            ("distance_public_transport", "pt"),
        ):
            v = row.get(key)
            if v is not None:
                extras.append(f"{label}={v}m")
        extras_str = f"  [{', '.join(extras)}]" if extras else ""

        print(f"#{rank:<2} [{r.score:.4f}] id={lid:<12} {city:<14} "
              f"{rooms:>4}rm  {price:>6} CHF  {area:>4}m2  {offer:<6} "
              f"- {title}{extras_str}")

        contribs = {k: r.signals[k] * r.weights.get(k, 0.0)
                    for k in r.signals if r.signals[k] > 0.0
                    and r.weights.get(k, 0.0) > 0.0}
        if contribs:
            parts = [f"{k}={r.signals[k]:.3f}*w{r.weights[k]:.2f}"
                     f" (+{contribs[k]:.3f})" for k in contribs]
            print(f"       signals: {'  '.join(parts)}")

        desc = row.get("description")
        if desc:
            snippet = " ".join(desc.split())[:160]
            print(f"       desc:    {snippet}{'...' if len(desc) > 160 else ''}")

        url = row.get("original_url")
        if url:
            print(f"       url:     {url}")
        print()


def _write_json(resp: UnifiedResponse, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = resp.to_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    size_kb = path.stat().st_size / 1024
    print(f"\n[json] wrote {len(payload['results'])} results + understanding "
          f"to {path} ({size_kb:.1f} KiB)")
    # Summary of what's in the dump
    active = payload["pipeline"]["active_signals"]
    print(f"[json] active signals: {active}")
    print(f"[json] sample keys per result: "
          f"{list(payload['results'][0].keys()) if payload['results'] else '(empty)'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="natural-language query (de / fr / it / en)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vlm", action="store_true",
                        help="include SigLIP2 image scoring (slower first call)")
    parser.add_argument("--shards-dir", type=Path, default=None,
                        help="override default features_vlm/siglip2/ path")
    parser.add_argument("--weight", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="override signal weight (repeatable). "
                             "keys: dense, bm25, vlm, geo_transit, geo_school, "
                             "geo_anchor, price_band")
    parser.add_argument("--json-out", type=Path, default=None,
                        metavar="PATH",
                        help="write full pipeline output (incl. descriptions, "
                             "images, signal breakdown) to this JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress stdout results (useful with --json-out)")
    args = parser.parse_args()

    overrides = _parse_weight_overrides(args.weight)

    resp = unified_search(
        args.query,
        top_k=args.top_k,
        use_vlm=args.vlm,
        vlm_shards_dir=args.shards_dir,
        override_weights=overrides or None,
    )

    if not args.quiet:
        _print_header(resp)
        _print_stages(resp)
        _print_results(resp)

    if args.json_out:
        _write_json(resp, args.json_out)


if __name__ == "__main__":
    main()
