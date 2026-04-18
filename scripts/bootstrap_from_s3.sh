#!/usr/bin/env bash
# Pull all runtime artifacts from S3 into the paths the containers expect.
#
# This is invoked by the `bootstrap` service in docker-compose.yml at startup;
# `api` and `mcp` block on its successful completion.
#
# Requires AWS credentials in the environment (AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, optionally AWS_SESSION_TOKEN, AWS_DEFAULT_REGION).
#
# `aws s3 sync` only transfers changed files, so re-runs are cheap.
#
# Paths in the container (mirror of ./app/config.py and unified_ranker.py):
#   /data/listings.db                                <- listings + embeddings (SQLite)
#   /data/bm25.pkl                                   <- lexical index
#   /data/geo/overpass_*.json                        <- geospatial POI cache
#   /app/features_vlm/siglip2/shard_*.npz            <- VLM image features
#   /app/downloads/prod/<source>/images/...          <- raw listing images (opt-in)
#
# Environment knobs:
#   ARTIFACTS_S3_BUCKET        (required) bucket name
#   ARTIFACTS_S3_REGION        default: us-west-2
#   BOOTSTRAP_INCLUDE_IMAGES   "1" to also sync raw/downloads (~30k files). Default: 0.
#   BOOTSTRAP_FORCE            "1" to re-sync even if files already exist. Default: 0.

set -euo pipefail

BUCKET="${ARTIFACTS_S3_BUCKET:?ARTIFACTS_S3_BUCKET env var is required}"
REGION="${ARTIFACTS_S3_REGION:-us-west-2}"
INCLUDE_IMAGES="${BOOTSTRAP_INCLUDE_IMAGES:-0}"
FORCE="${BOOTSTRAP_FORCE:-0}"

export AWS_DEFAULT_REGION="$REGION"

mkdir -p /data/geo /app/features_vlm/siglip2

# --- guard: credentials work at all ---
echo "[bootstrap] verifying S3 access to s3://$BUCKET ..."
if ! aws s3 ls "s3://$BUCKET/artifacts/" --region "$REGION" > /dev/null; then
    echo "[bootstrap] ERROR: cannot list s3://$BUCKET/artifacts/ - check AWS creds" >&2
    exit 1
fi

_sync() {
    # _sync <s3_prefix> <local_dir> <sentinel_file>
    # If sentinel exists and FORCE != 1, skip the sync.
    local src="$1" dst="$2" sentinel="$3"
    if [[ "$FORCE" != "1" && -e "$sentinel" ]]; then
        echo "[bootstrap] [skip]  $dst (already populated; set BOOTSTRAP_FORCE=1 to re-sync)"
        return 0
    fi
    echo "[bootstrap] [sync]  $src  ->  $dst"
    aws s3 sync "$src" "$dst" --region "$REGION" --no-progress --only-show-errors
}

# --- 1. SQLite DB (listings + text embeddings live inside this file) ---
_sync "s3://$BUCKET/artifacts/text_embeddings/" /data/                   /data/listings.db

# --- 2. BM25 pickle ---
_sync "s3://$BUCKET/artifacts/bm25/"            /data/                   /data/bm25.pkl

# --- 3. Geo POI cache (Overpass JSON blobs) ---
_sync "s3://$BUCKET/artifacts/geo/"             /data/geo/               /data/geo/overpass_transit.json

# --- 4. VLM SigLIP2 shards ---
_sync "s3://$BUCKET/artifacts/vlm/siglip2/"     /app/features_vlm/siglip2/  /app/features_vlm/siglip2/shard_00000.npz

# --- 5. Raw images (opt-in; ~30k files, big) ---
if [[ "$INCLUDE_IMAGES" == "1" ]]; then
    echo "[bootstrap] [sync]  raw images (BOOTSTRAP_INCLUDE_IMAGES=1)"
    aws s3 sync "s3://$BUCKET/raw/downloads/" /app/downloads/ \
        --region "$REGION" --no-progress --only-show-errors
else
    echo "[bootstrap] [skip]  raw images (set BOOTSTRAP_INCLUDE_IMAGES=1 to enable)"
fi

echo "[bootstrap] summary:"
ls -lh /data/listings.db /data/bm25.pkl 2>/dev/null || true
echo "  geo cache:  $(find /data/geo -maxdepth 1 -name 'overpass_*.json' 2>/dev/null | wc -l) files"
echo "  VLM shards: $(find /app/features_vlm/siglip2 -maxdepth 1 -name 'shard_*.npz' 2>/dev/null | wc -l) files"
if [[ "$INCLUDE_IMAGES" == "1" ]]; then
    echo "  images:     $(find /app/downloads -maxdepth 4 -name '*.jpg' 2>/dev/null | wc -l) files"
fi

echo "[bootstrap] done."
