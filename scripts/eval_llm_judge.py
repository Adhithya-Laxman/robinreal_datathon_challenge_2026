"""LLM-as-judge evaluation for the unified search pipeline.

Runs the pipeline over queries from `benchmarks/benchmarks.json` and asks
Claude (via the Anthropic direct API) to rate each top-K result 0-5 for
relevance. Aggregates into IR-style summary metrics.

We deliberately **do NOT use** the `result_listing_ids` field from the
benchmark as ground-truth. That list was produced by a prior LLM without
access to the full dataset, so it's a weak oracle. Instead, we use the
queries only and let a fresh judge score our top-K independently.

Usage (inside the api container):

    # quick run: 20 queries stratified across languages, top-5 judged by Haiku
    python scripts/eval_llm_judge.py --n 20

    # full run with VLM + Sonnet judge (more expensive, stronger signal)
    python scripts/eval_llm_judge.py --n 40 --vlm --judge-model claude-sonnet-4-5

    # custom out path
    python scripts/eval_llm_judge.py --n 10 --out results/eval/run_a.json

The script produces:
    results/eval/<stem>.json   — per-query details (query, pipeline output,
                                 judge scores + reasons, aggregates)
    results/eval/<stem>.md     — markdown summary table (paste into README)

Environment:
    ANTHROPIC_API_KEY   — required (uses Claude credits, not Bedrock)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.participant.unified_ranker import unified_search  # noqa: E402


# ---------------------------------------------------------------- judge prompt

_JUDGE_SYSTEM = """You are an expert Swiss residential real-estate agent.
You rate how well a candidate apartment listing matches a natural-language
user request. You are STRICT but FAIR:

 - 0 = Contradicts the query (wrong city/region, wrong property type, or a
       hard constraint like price/rooms is clearly violated).
 - 1 = Barely related. Major mismatch in city, type, or budget.
 - 2 = Weak partial match. Some alignment but most soft preferences missed.
 - 3 = Reasonable match. Most soft preferences met, hard constraints OK.
 - 4 = Strong match. Only minor nitpicks.
 - 5 = Ideal match. Would genuinely recommend this to the user first.

Consider:
  * hard constraints mentioned in the query (city, rooms, budget, area, type)
  * soft preferences (quiet/bright/modern/near X/etc.)
  * practical fit (commute anchor like ETH / EPFL / HB)
  * obvious red flags (parking spot instead of apartment, wildly over budget, etc.)

Respond ONLY by calling the `rate_listing` tool. No prose."""

_JUDGE_TOOL = {
    "name": "rate_listing",
    "description": "Return a relevance score 0-5 and a short reason (<=25 words).",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "Relevance score from 0 (contradicts) to 5 (ideal).",
            },
            "reason": {
                "type": "string",
                "description": "≤25 words explaining the score (what matched / what didn't).",
            },
        },
        "required": ["score", "reason"],
    },
}


def _format_listing_for_judge(res: dict[str, Any]) -> str:
    """Compact, LLM-friendly representation of one listing."""
    ident = res.get("identity", {})
    loc = res.get("location", {})
    prop = res.get("property", {})
    enriched = (loc.get("enriched_distances_m") or {})

    dist_parts = []
    for label, key in [("transit", "transit"), ("supermarket", "supermarket"),
                       ("school", "school"), ("uni/anchor", "university")]:
        v = enriched.get(key)
        if v is not None:
            dist_parts.append(f"{label}={v}m")
    dist_str = ", ".join(dist_parts) if dist_parts else "(no POI distances)"

    desc = (ident.get("description") or "").strip()
    # Strip HTML tags crudely to save judge tokens.
    for br in ("<br />", "<br>", "<br/>", "<p>", "</p>"):
        desc = desc.replace(br, " ")
    desc = " ".join(desc.split())
    desc = desc[:500] + ("…" if len(desc) > 500 else "")

    features = ", ".join(prop.get("features") or []) or "none"
    return (
        f"City: {loc.get('city') or '?'} ({loc.get('canton') or '?'})\n"
        f"Type: {ident.get('object_category') or '?'}"
        f" | {prop.get('rooms') or '?'} rooms"
        f" | {prop.get('area_sqm') or '?'} m²"
        f" | CHF {prop.get('price_chf') or '?'}/mo"
        f" | offer={ident.get('offer_type') or '?'}\n"
        f"Distances: {dist_str}\n"
        f"Features: {features}\n"
        f"Title: {ident.get('title') or '(no title)'}\n"
        f"Desc: {desc or '(no description)'}"
    )


# ------------------------------------------------------------------ judge call

@dataclass
class JudgeScore:
    listing_id: str
    rank: int
    final_score: float
    score: int       # 0..5 from judge
    reason: str
    ok: bool = True  # False if judge call failed
    error: str = ""


def _call_judge(
    client,          # anthropic.Anthropic
    *,
    model: str,
    query: str,
    listing_blob: str,
    max_retries: int = 2,
) -> tuple[int, str, bool, str]:
    """Return (score, reason, ok, error)."""
    user_msg = (
        f"User query:\n{query}\n\n"
        f"Candidate listing:\n{listing_blob}\n\n"
        "Call rate_listing with score (0..5) and a brief reason."
    )
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0.0,
                system=_JUDGE_SYSTEM,
                tools=[_JUDGE_TOOL],
                tool_choice={"type": "tool", "name": _JUDGE_TOOL["name"]},
                messages=[{"role": "user", "content": user_msg}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" \
                        and block.name == _JUDGE_TOOL["name"]:
                    payload = dict(block.input or {})
                    score = int(payload.get("score", 0))
                    reason = str(payload.get("reason", ""))[:250]
                    return score, reason, True, ""
            last_err = "no tool_use block in response"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * (attempt + 1))
    return 0, "", False, last_err


# ---------------------------------------------------------------- IR-ish math

def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg(relevances: list[int]) -> float:
    ideal = sorted(relevances, reverse=True)
    denom = _dcg(ideal)
    return (_dcg(relevances) / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------ sampling

def _stratified_sample(
    benchmarks: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Roughly-even sample across languages."""
    if n >= len(benchmarks):
        return benchmarks
    rng = random.Random(seed)
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in benchmarks:
        by_lang[q.get("language", "?")].append(q)
    langs = sorted(by_lang)
    per_lang = max(1, n // len(langs))
    picked: list[dict[str, Any]] = []
    for l in langs:
        pool = by_lang[l][:]
        rng.shuffle(pool)
        picked.extend(pool[:per_lang])
    # Top up if rounding left us short.
    if len(picked) < n:
        remaining = [q for q in benchmarks if q not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    return picked[:n]


# ------------------------------------------------------------------ main loop

@dataclass
class QueryEvalRecord:
    query: str
    language: str
    top_k_judged: int
    used_llm_in_pipeline: bool
    pipeline_interpretation: str
    candidates_before_rerank: int
    active_signals: list[str]
    results: list[dict[str, Any]] = field(default_factory=list)
    judge_scores: list[JudgeScore] = field(default_factory=list)

    # per-query metrics
    mean_score: float = 0.0
    pct_relevant_ge3: float = 0.0
    pct_strong_ge4: float = 0.0
    ndcg_at_k: float = 0.0
    error: str = ""


def _evaluate_one_query(
    q_record: dict[str, Any],
    *,
    top_k: int,
    use_vlm: bool,
    judge_client,
    judge_model: str,
    judge_parallelism: int,
) -> QueryEvalRecord:
    query = q_record["query"]
    lang = q_record.get("language", "?")
    rec = QueryEvalRecord(
        query=query, language=lang, top_k_judged=top_k,
        used_llm_in_pipeline=False,
        pipeline_interpretation="",
        candidates_before_rerank=0,
        active_signals=[],
    )
    try:
        resp = unified_search(query, top_k=top_k, use_vlm=use_vlm)
    except Exception as e:  # noqa: BLE001
        rec.error = f"pipeline: {type(e).__name__}: {e}"
        return rec

    resp_d = resp.to_dict()
    pipe = resp_d.get("pipeline", {})
    rec.candidates_before_rerank = pipe.get("candidates_before_rerank", 0)
    rec.active_signals = pipe.get("active_signals", []) or []
    rec.used_llm_in_pipeline = resp_d.get("understanding", {}).get("used_llm", False)
    rec.pipeline_interpretation = resp_d.get("understanding", {}).get("interpretation") or ""
    rec.results = resp_d.get("results", [])[:top_k]

    if not rec.results:
        rec.error = "pipeline returned 0 results"
        return rec

    # --- judge each result in parallel ---------------------------------
    def _task(idx_and_res):
        idx, res = idx_and_res
        blob = _format_listing_for_judge(res)
        s, reason, ok, err = _call_judge(
            judge_client, model=judge_model, query=query, listing_blob=blob,
        )
        return JudgeScore(
            listing_id=str(res.get("listing_id") or idx),
            rank=idx + 1,
            final_score=float(res.get("final_score") or 0.0),
            score=s, reason=reason, ok=ok, error=err,
        )

    with cf.ThreadPoolExecutor(max_workers=judge_parallelism) as ex:
        rec.judge_scores = list(ex.map(_task, enumerate(rec.results)))

    ok_scores = [js.score for js in rec.judge_scores if js.ok]
    if ok_scores:
        rec.mean_score = statistics.mean(ok_scores)
        rec.pct_relevant_ge3 = sum(1 for s in ok_scores if s >= 3) / len(ok_scores)
        rec.pct_strong_ge4 = sum(1 for s in ok_scores if s >= 4) / len(ok_scores)
        # nDCG using the judge score as the graded relevance.
        rec.ndcg_at_k = _ndcg(ok_scores)
    return rec


# ----------------------------------------------------------------- aggregate

def _aggregate(records: list[QueryEvalRecord]) -> dict[str, Any]:
    ok = [r for r in records if not r.error and r.judge_scores]
    all_scores: list[int] = []
    for r in ok:
        all_scores.extend(js.score for js in r.judge_scores if js.ok)

    def _avg(vals):
        return statistics.mean(vals) if vals else 0.0

    out: dict[str, Any] = {
        "n_queries_total": len(records),
        "n_queries_evaluated_ok": len(ok),
        "n_queries_errored": len(records) - len(ok),
        "total_judge_calls": len(all_scores),
        "overall": {
            "mean_relevance_score": round(_avg(all_scores), 3),
            "pct_relevant_ge3": round(
                sum(1 for s in all_scores if s >= 3) / max(len(all_scores), 1), 3),
            "pct_strong_ge4": round(
                sum(1 for s in all_scores if s >= 4) / max(len(all_scores), 1), 3),
            "mean_ndcg_at_k": round(_avg([r.ndcg_at_k for r in ok]), 3),
        },
        "by_language": {},
    }

    # Per-language breakdown.
    by_lang: dict[str, list[QueryEvalRecord]] = defaultdict(list)
    for r in ok:
        by_lang[r.language].append(r)
    for lang, lst in sorted(by_lang.items()):
        flat = [js.score for r in lst for js in r.judge_scores if js.ok]
        if not flat:
            continue
        out["by_language"][lang] = {
            "n_queries": len(lst),
            "mean_relevance_score": round(_avg(flat), 3),
            "pct_relevant_ge3": round(sum(1 for s in flat if s >= 3) / len(flat), 3),
            "pct_strong_ge4": round(sum(1 for s in flat if s >= 4) / len(flat), 3),
            "mean_ndcg_at_k": round(_avg([r.ndcg_at_k for r in lst]), 3),
        }
    return out


def _render_markdown(summary: dict[str, Any], args: argparse.Namespace,
                     records: list[QueryEvalRecord]) -> str:
    o = summary["overall"]
    lines = [
        f"# LLM-Judge Evaluation",
        "",
        f"- **Queries evaluated**: {summary['n_queries_evaluated_ok']} / "
        f"{summary['n_queries_total']} (errored: {summary['n_queries_errored']})",
        f"- **Top-K judged per query**: {args.top_k}",
        f"- **Judge model**: `{args.judge_model}` (Anthropic direct API)",
        f"- **Pipeline VLM**: {'on' if args.vlm else 'off'}",
        f"- **Total judge calls**: {summary['total_judge_calls']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean Relevance Score (0–5) | **{o['mean_relevance_score']}** |",
        f"| % Relevant @{args.top_k} (score ≥ 3) | **{o['pct_relevant_ge3']*100:.1f}%** |",
        f"| % Strong   @{args.top_k} (score ≥ 4) | **{o['pct_strong_ge4']*100:.1f}%** |",
        f"| Mean nDCG @{args.top_k} (judge-graded) | **{o['mean_ndcg_at_k']}** |",
        "",
        "## By language",
        "",
        "| Lang | N | Mean Score | %Rel@K (≥3) | %Strong@K (≥4) | nDCG@K |",
        "|---|---|---|---|---|---|",
    ]
    for lang, d in summary["by_language"].items():
        lines.append(
            f"| {lang} | {d['n_queries']} | {d['mean_relevance_score']} | "
            f"{d['pct_relevant_ge3']*100:.1f}% | "
            f"{d['pct_strong_ge4']*100:.1f}% | "
            f"{d['mean_ndcg_at_k']} |"
        )

    # A handful of qualitative examples (low + high).
    all_judgments: list[tuple[QueryEvalRecord, JudgeScore]] = [
        (r, js) for r in records if r.judge_scores
        for js in r.judge_scores if js.ok
    ]
    if all_judgments:
        lines += ["", "## Best matches (examples where judge scored 5)", ""]
        top = [x for x in all_judgments if x[1].score == 5][:5]
        if not top:
            top = sorted(all_judgments, key=lambda x: -x[1].score)[:5]
        for r, js in top:
            lines.append(
                f"- **[{js.score}/5]** `{r.language}` *\"{r.query[:70]}…\"* → "
                f"#{js.rank} id={js.listing_id} — {js.reason}"
            )

        worst = sorted(all_judgments, key=lambda x: x[1].score)[:5]
        lines += ["", "## Failure modes (lowest-scored top-K results)", ""]
        for r, js in worst:
            lines.append(
                f"- **[{js.score}/5]** `{r.language}` *\"{r.query[:70]}…\"* → "
                f"#{js.rank} id={js.listing_id} — {js.reason}"
            )

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ entry

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__ or "",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--benchmark", type=Path,
                   default=REPO_ROOT / "benchmarks" / "benchmarks.json",
                   help="path to benchmarks.json")
    p.add_argument("--n", type=int, default=20,
                   help="how many queries to evaluate (stratified across languages)")
    p.add_argument("--top-k", type=int, default=5,
                   help="how many top pipeline results to judge per query")
    p.add_argument("--vlm", action="store_true",
                   help="enable VLM scoring in the pipeline")
    p.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                   help="Anthropic model id used as judge "
                        "(default: claude-haiku-4-5-20251001; "
                        "use claude-sonnet-4-5 for stronger but slower judge)")
    p.add_argument("--query-parallelism", type=int, default=1,
                   help="run N queries in parallel (each query ALSO uses "
                        "--judge-parallelism for its K judgments)")
    p.add_argument("--judge-parallelism", type=int, default=5,
                   help="how many top-K judgments per query to run in parallel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path (default: results/eval/judge_<ts>.json)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in the environment.",
              file=sys.stderr)
        print("       Add it to your .env (or export it) and re-run.",
              file=sys.stderr)
        return 2

    import anthropic  # lazy
    judge_client = anthropic.Anthropic()

    benchmarks: list[dict[str, Any]] = json.loads(args.benchmark.read_text("utf-8"))
    pool = _stratified_sample(benchmarks, args.n, args.seed)

    print(f"[eval] benchmarks: {len(benchmarks)} total -> sampling {len(pool)} "
          f"queries (stratified, seed={args.seed})")
    lang_hist = Counter(q.get("language", "?") for q in pool)
    print(f"[eval] language distribution: {dict(lang_hist)}")
    print(f"[eval] top_k={args.top_k}  vlm={args.vlm}  judge={args.judge_model}")
    print(f"[eval] query_parallelism={args.query_parallelism}  "
          f"judge_parallelism={args.judge_parallelism}")

    t0 = time.time()
    records: list[QueryEvalRecord] = []

    def _run(q):
        return _evaluate_one_query(
            q,
            top_k=args.top_k,
            use_vlm=args.vlm,
            judge_client=judge_client,
            judge_model=args.judge_model,
            judge_parallelism=args.judge_parallelism,
        )

    if args.query_parallelism > 1:
        with cf.ThreadPoolExecutor(max_workers=args.query_parallelism) as ex:
            for i, rec in enumerate(ex.map(_run, pool), 1):
                records.append(rec)
                _print_progress(i, len(pool), rec)
    else:
        for i, q in enumerate(pool, 1):
            rec = _run(q)
            records.append(rec)
            _print_progress(i, len(pool), rec)

    elapsed = time.time() - t0
    print(f"\n[eval] done in {elapsed:.1f}s")

    summary = _aggregate(records)

    # -------- write outputs --------
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out or (out_dir / f"judge_{ts}.json")
    out_md = out_json.with_suffix(".md")

    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
        "per_query": [
            {
                "query": r.query,
                "language": r.language,
                "used_llm_in_pipeline": r.used_llm_in_pipeline,
                "pipeline_interpretation": r.pipeline_interpretation,
                "candidates_before_rerank": r.candidates_before_rerank,
                "active_signals": r.active_signals,
                "mean_score": round(r.mean_score, 3),
                "pct_relevant_ge3": round(r.pct_relevant_ge3, 3),
                "pct_strong_ge4": round(r.pct_strong_ge4, 3),
                "ndcg_at_k": round(r.ndcg_at_k, 3),
                "error": r.error,
                "judge_scores": [asdict(js) for js in r.judge_scores],
                "results": [
                    {
                        "rank": res.get("rank"),
                        "listing_id": res.get("listing_id"),
                        "final_score": res.get("final_score"),
                        "city": (res.get("location") or {}).get("city"),
                        "price_chf": (res.get("property") or {}).get("price_chf"),
                        "rooms": (res.get("property") or {}).get("rooms"),
                        "title": (res.get("identity") or {}).get("title"),
                    }
                    for res in r.results
                ],
            }
            for r in records
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    out_md.write_text(_render_markdown(summary, args, records), "utf-8")

    def _pretty(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"\n[eval] wrote {_pretty(out_json)} "
          f"({out_json.stat().st_size/1024:.1f} KiB)")
    print(f"[eval] wrote {_pretty(out_md)}")

    o = summary["overall"]
    print("\n=== SUMMARY =====================================================")
    print(f"  Mean Relevance Score : {o['mean_relevance_score']} / 5")
    print(f"  % Relevant @{args.top_k} (≥3)  : {o['pct_relevant_ge3']*100:.1f}%")
    print(f"  % Strong   @{args.top_k} (≥4)  : {o['pct_strong_ge4']*100:.1f}%")
    print(f"  Mean nDCG  @{args.top_k}       : {o['mean_ndcg_at_k']}")
    print("=================================================================\n")

    return 0


def _print_progress(i: int, total: int, rec: QueryEvalRecord) -> None:
    status = "OK" if not rec.error else "ERR"
    score = rec.mean_score if not rec.error else 0.0
    q = rec.query.replace("\n", " ")[:60]
    tail = rec.error if rec.error else f"mean={score:.2f}"
    print(f"  [{i:3d}/{total}] [{status}] {rec.language} | {tail} | {q}…")


if __name__ == "__main__":
    raise SystemExit(main())
