"""Session cookie middleware.

Sets a `robin_session` UUID cookie on every response that doesn't already
have one, and exposes `request.state.session_id` to downstream handlers.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

COOKIE_NAME = "robin_session"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        session_id = request.cookies.get(COOKIE_NAME)
        is_new = not session_id
        if is_new:
            session_id = str(uuid.uuid4())
        request.state.session_id = session_id
        response: Response = await call_next(request)
        if is_new:
            response.set_cookie(
                COOKIE_NAME,
                session_id,
                max_age=_COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
            )
        return response
