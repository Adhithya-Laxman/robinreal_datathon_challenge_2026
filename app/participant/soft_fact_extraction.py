"""Soft-fact extraction entry point used by the harness.

Shares the cached `QueryUnderstanding` from `query_understanding.understand`
with `hard_fact_extraction`, so both calls trigger at most one LLM round-trip
per unique query.

We return a dict (not the pydantic model) because the harness contract is
`dict[str, Any]` and downstream teammates can keep their current signatures;
the `"understanding"` key carries the full typed object for anyone who
wants it.
"""

from __future__ import annotations

from typing import Any

from app.participant.query_understanding import understand


def extract_soft_facts(query: str) -> dict[str, Any]:
    qu = understand(query)
    payload: dict[str, Any] = {
        "raw_query": query,
        "language": qu.language,
        "used_llm": qu.used_llm,
        "interpretation": qu.interpretation,
        "soft": qu.soft.model_dump(),
        "understanding": qu,
    }
    return payload
