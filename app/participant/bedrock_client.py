"""Lazy boto3 `bedrock-runtime` client for participant modules.

We keep this separate from the S3 credentials (`aws_dataset_access.sh`) because
the S3 user has no Bedrock permissions. Bedrock creds come from the AWS
workshop account via `.env` (or any other process env / AWS profile chain).

Usage:

    from app.participant.bedrock_client import get_bedrock_client, bedrock_available

    if bedrock_available():
        client = get_bedrock_client()
        ...

If Bedrock is not configured (or `FORCE_BEDROCK_FALLBACK=1`), `bedrock_available`
returns False and callers should use their heuristic fallback path.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()
_cached_client: Any | None = None


def bedrock_available(settings: Settings | None = None) -> bool:
    """Whether we should attempt to use Bedrock for this call.

    Returns False if:
      - the FORCE_BEDROCK_FALLBACK kill-switch is on, or
      - there are no explicit creds AND the default AWS credential chain has
        nothing usable either.

    We only do the cheap env-var check here. Actual client construction
    happens lazily in `get_bedrock_client`.
    """
    settings = settings or get_settings()
    if settings.force_bedrock_fallback:
        return False
    if settings.bedrock_access_key_id and settings.bedrock_secret_access_key:
        return True
    # Fall back to the standard boto3 chain (instance profile, AWS_PROFILE,
    # AWS_ACCESS_KEY_ID from the ambient env, etc.). We don't try to
    # construct a client here to avoid network calls on every request.
    import os

    return bool(os.getenv("AWS_ACCESS_KEY_ID")) or bool(os.getenv("AWS_PROFILE"))


def get_bedrock_client() -> Any:
    """Return a cached boto3 bedrock-runtime client.

    Raises RuntimeError if Bedrock is not configured.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    with _client_lock:
        if _cached_client is not None:
            return _cached_client

        if not bedrock_available():
            raise RuntimeError(
                "Bedrock is not configured. Set BEDROCK_AWS_ACCESS_KEY_ID/"
                "BEDROCK_AWS_SECRET_ACCESS_KEY in .env (workshop account creds) "
                "or unset FORCE_BEDROCK_FALLBACK."
            )

        import boto3
        from botocore.config import Config

        settings = get_settings()
        boto_config = Config(
            region_name=settings.bedrock_region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=60,
        )

        if settings.bedrock_access_key_id and settings.bedrock_secret_access_key:
            session = boto3.session.Session(
                aws_access_key_id=settings.bedrock_access_key_id,
                aws_secret_access_key=settings.bedrock_secret_access_key,
                aws_session_token=settings.bedrock_session_token,
                region_name=settings.bedrock_region,
            )
        else:
            session = boto3.session.Session(region_name=settings.bedrock_region)

        _cached_client = session.client("bedrock-runtime", config=boto_config)
        logger.info(
            "Initialized bedrock-runtime client (region=%s)",
            settings.bedrock_region,
        )
        return _cached_client


@lru_cache(maxsize=1)
def _bedrock_health_check() -> bool:
    """Cheap probe — tries to describe a known model. Cached so only runs once."""
    try:
        get_bedrock_client()
        return True
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("Bedrock health check failed: %s", exc)
        return False
