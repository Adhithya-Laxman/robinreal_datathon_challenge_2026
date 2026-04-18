#!/usr/bin/env bash
# Download listing images from S3 into ./downloads/prod/
# Usage:
#   source ./aws_dataset_access.sh  (or `set -a && . ./aws_dataset_access.sh && set +a`)
#   ./scripts/download_images.sh [robinreal|comparis|all]
#
# Uses the official amazon/aws-cli Docker image so no host install is needed.
# Safe to re-run: aws s3 sync only transfers what's missing / changed.
set -euo pipefail

: "${AWS_ACCESS_KEY_ID:?AWS creds not in env. Source aws_dataset_access.sh first.}"
: "${AWS_SECRET_ACCESS_KEY:?AWS creds not in env. Source aws_dataset_access.sh first.}"
: "${AWS_DEFAULT_REGION:=eu-central-2}"

BUCKET="${LISTINGS_S3_BUCKET:-crawl-data-951752554117-eu-central-2-an}"
PREFIX="${LISTINGS_S3_PREFIX:-prod}"
WHAT="${1:-all}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/downloads/${PREFIX}"
mkdir -p "${DEST}"

run_sync() {
  local source="$1"
  echo "[sync] s3://${BUCKET}/${PREFIX}/${source}/images/  ->  ${DEST}/${source}/images/"
  docker run --rm \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_DEFAULT_REGION \
    -v "${DEST}:/out" \
    amazon/aws-cli s3 sync \
      "s3://${BUCKET}/${PREFIX}/${source}/images/" \
      "/out/${source}/images/" \
      --only-show-errors
}

case "${WHAT}" in
  robinreal) run_sync robinreal ;;
  comparis)  run_sync comparis ;;
  all)       run_sync robinreal && run_sync comparis ;;
  *) echo "Unknown target: ${WHAT} (expected robinreal|comparis|all)" >&2; exit 2 ;;
esac

echo "[done] images under ${DEST}"
du -sh "${DEST}"/*/ 2>/dev/null || true
