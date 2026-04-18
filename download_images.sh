#!/usr/bin/env bash
# Bulk-download all listing images from the shared datathon S3 bucket.
#
# Prereqs (teammate side, one-time):
#   1. Join the same Workshop Studio event with your team's access code.
#   2. Grab AWS CLI creds from the Workshop Studio dashboard and
#      export them (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#      AWS_SESSION_TOKEN, AWS_DEFAULT_REGION=us-west-2).
#
# Then from the repo root:
#   bash download_images.sh            # downloads EVERYTHING (~4 GB)
#   bash download_images.sh 37082125   # only images for one platform_id
#   bash download_images.sh --source comparis   # only comparis source

set -euo pipefail

BUCKET="datathon-robinreal-085455950306-us-west-2"
DEST="downloads"
mkdir -p "$DEST"

if [[ $# -eq 0 ]]; then
  echo "Syncing full image tree ($BUCKET/raw/downloads/ -> $DEST/) ..."
  aws s3 sync "s3://$BUCKET/raw/downloads/" "$DEST/" \
      --only-show-errors --no-progress
  echo "Done."
  exit 0
fi

if [[ "${1:-}" == "--source" ]]; then
  src="${2:?missing source name, e.g. comparis or robinreal}"
  echo "Syncing only source=$src ..."
  aws s3 sync "s3://$BUCKET/raw/downloads/prod/$src/" "$DEST/prod/$src/" \
      --only-show-errors --no-progress
  exit 0
fi

# Otherwise, treat each arg as a platform_id and sync only its images
for pid in "$@"; do
  for src in comparis robinreal; do
    prefix="raw/downloads/prod/$src/images/platform_id=$pid/"
    local_dir="$DEST/prod/$src/images/platform_id=$pid/"
    if aws s3 ls "s3://$BUCKET/$prefix" > /dev/null 2>&1; then
      mkdir -p "$local_dir"
      aws s3 sync "s3://$BUCKET/$prefix" "$local_dir" \
          --only-show-errors --no-progress
      echo "  [$src] platform_id=$pid ✓"
    fi
  done
done
echo "Done."
