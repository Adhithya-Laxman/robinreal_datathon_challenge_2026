"""Event logging endpoints for session-based implicit feedback."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.harness.events import get_recent_events, log_event

router = APIRouter(prefix="/events", tags=["events"])


class LogEventRequest(BaseModel):
    event_type: Literal["click", "dwell", "save"]
    listing_id: str
    dwell_ms: int | None = None


@router.post("")
def post_event(body: LogEventRequest, request: Request) -> dict:
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        return {"ok": False, "reason": "no session"}
    log_event(
        session_id=session_id,
        event_type=body.event_type,
        listing_id=body.listing_id,
        dwell_ms=body.dwell_ms,
    )
    return {"ok": True, "session_id": session_id}


@router.get("/session")
def get_session_events(request: Request) -> dict:
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        return {"session_id": None, "events": []}
    return {"session_id": session_id, "events": get_recent_events(session_id)}
