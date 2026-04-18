from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_default_raw_data_dir() -> Path:
    root = _project_root()
    configured = os.getenv("LISTINGS_RAW_DATA_DIR")
    if configured:
        return Path(configured)
    return root / "raw_data"


def _default_db_path() -> Path:
    configured = os.getenv("LISTINGS_DB_PATH")
    if configured:
        return Path(configured)
    return _project_root() / "data" / "listings.db"


@dataclass(slots=True)
class Settings:
    raw_data_dir: Path
    db_path: Path
    s3_bucket: str
    s3_region: str
    s3_prefix: str
    bedrock_region: str
    bedrock_access_key_id: str | None
    bedrock_secret_access_key: str | None
    bedrock_session_token: str | None
    bedrock_query_understanding_model_id: str
    bedrock_embedding_model_id: str
    bedrock_explanation_model_id: str
    force_bedrock_fallback: bool
    # Fallback path 1: Anthropic direct API (for query understanding +
    # explanations). Kicks in if Bedrock returns AccessDenied or the user
    # sets FORCE_BEDROCK_FALLBACK=1.
    anthropic_api_key: str | None
    anthropic_model_id: str
    # Fallback path 2: local multilingual embedding model via fastembed.
    # Used when Bedrock Cohere is unavailable OR when USE_LOCAL_EMBEDDINGS=1.
    use_local_embeddings: bool
    local_embedding_model: str


def get_settings() -> Settings:
    return Settings(
        raw_data_dir=_find_default_raw_data_dir(),
        db_path=_default_db_path(),
        s3_bucket=os.getenv(
            "LISTINGS_S3_BUCKET",
            "crawl-data-951752554117-eu-central-2-an",
        ),
        s3_region=os.getenv("LISTINGS_S3_REGION", "eu-central-2"),
        s3_prefix=os.getenv("LISTINGS_S3_PREFIX", "prod"),
        bedrock_region=os.getenv("BEDROCK_AWS_REGION", "us-east-1"),
        bedrock_access_key_id=os.getenv("BEDROCK_AWS_ACCESS_KEY_ID") or None,
        bedrock_secret_access_key=os.getenv("BEDROCK_AWS_SECRET_ACCESS_KEY") or None,
        bedrock_session_token=os.getenv("BEDROCK_AWS_SESSION_TOKEN") or None,
        bedrock_query_understanding_model_id=os.getenv(
            "BEDROCK_QUERY_UNDERSTANDING_MODEL_ID",
            "anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        bedrock_embedding_model_id=os.getenv(
            "BEDROCK_EMBEDDING_MODEL_ID",
            "cohere.embed-multilingual-v3",
        ),
        bedrock_explanation_model_id=os.getenv(
            "BEDROCK_EXPLANATION_MODEL_ID",
            "anthropic.claude-haiku-4-5-20250925-v1:0",
        ),
        force_bedrock_fallback=os.getenv("FORCE_BEDROCK_FALLBACK", "0") == "1",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_model_id=os.getenv(
            "ANTHROPIC_MODEL_ID",
            "claude-sonnet-4-20250514",
        ),
        use_local_embeddings=os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1",
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
    )
