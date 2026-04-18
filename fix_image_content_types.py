"""One-shot: set Content-Type = image/<ext> on every listing image in S3.

Run once after the initial `aws s3 sync` upload. After this, presigned URLs
(and any future direct access) will render inline in a browser by default.

Usage (from repo root, AWS creds sourced):

    python3 fix_image_content_types.py            # fix everything
    python3 fix_image_content_types.py --dry-run  # just print what would change
    python3 fix_image_content_types.py --workers 32
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import sys

import boto3
from botocore.config import Config

BUCKET = "datathon-robinreal-085455950306-us-west-2"
PREFIX = "raw/downloads/"

_EXT_TO_MIME = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _mime_for(key: str) -> str | None:
    k = key.lower()
    for ext, mime in _EXT_TO_MIME.items():
        if k.endswith(ext):
            return mime
    return None


def _fix_one(s3, key: str, dry_run: bool) -> tuple[str, str]:
    mime = _mime_for(key)
    if mime is None:
        return key, "skip-notimage"

    head = s3.head_object(Bucket=BUCKET, Key=key)
    current = head.get("ContentType", "")
    if current == mime:
        return key, "ok"

    if dry_run:
        return key, f"would-fix ({current} -> {mime})"

    # Server-side copy in place with new Content-Type.
    s3.copy_object(
        Bucket=BUCKET, Key=key,
        CopySource={"Bucket": BUCKET, "Key": key},
        ContentType=mime,
        MetadataDirective="REPLACE",
    )
    return key, f"fixed ({current} -> {mime})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--prefix", default=PREFIX,
                   help="S3 prefix to walk (default: raw/downloads/)")
    args = p.parse_args()

    # Fatter connection pool so 64 threads don't queue up on the client side.
    s3 = boto3.client(
        "s3",
        config=Config(max_pool_connections=args.workers * 2,
                      retries={"max_attempts": 6, "mode": "adaptive"}),
    )

    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    print(f"[fix] found {len(keys):,} objects under s3://{BUCKET}/{args.prefix}")
    if not keys:
        return 0

    counts = {"fixed": 0, "ok": 0, "skip-notimage": 0, "would-fix": 0, "error": 0}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fix_one, s3, k, args.dry_run): k for k in keys}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            key = futures[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = f"error: {e.__class__.__name__}"
                counts["error"] += 1
            else:
                bucket = status.split()[0] if status.startswith(("would-fix", "fixed"))\
                         else status
                counts[bucket] = counts.get(bucket, 0) + 1
            if i % 500 == 0 or i == len(keys):
                print(f"  [{i:>6}/{len(keys)}]  "
                      + "  ".join(f"{k}={v}" for k, v in counts.items() if v))

    print("\n[fix] summary:")
    for k, v in counts.items():
        if v:
            print(f"  {k:<15} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
