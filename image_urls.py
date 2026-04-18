"""On-demand presigned URL minter for listing images.

Usage (from the repo root, after sourcing AWS creds):

    python image_urls.py 37082125              # URLs for one platform_id
    python image_urls.py 37082125 37090257     # multiple
    python image_urls.py --source comparis --limit 5    # first 5 of a source
"""
from __future__ import annotations
import argparse, json, sys
import boto3

BUCKET = "datathon-robinreal-085455950306-us-west-2"
TTL    = 3600   # seconds; capped by STS session lifetime anyway

def _guess_mime(key: str) -> str:
    k = key.lower()
    if k.endswith(".png"):  return "image/png"
    if k.endswith(".gif"):  return "image/gif"
    if k.endswith(".webp"): return "image/webp"
    return "image/jpeg"   # everything else in this dataset is JPEG


def urls_for_listing(s3, source: str, platform_id: str,
                     inline: bool = True) -> list[str]:
    prefix = f"raw/downloads/prod/{source}/images/platform_id={platform_id}/"
    keys = [o["Key"] for o in
            s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])]
    out: list[str] = []
    for k in keys:
        params = {"Bucket": BUCKET, "Key": k}
        if inline:
            # Force the browser to render the bytes instead of downloading.
            # Works even though the stored Content-Type is octet-stream,
            # because S3 honours these response-header overrides on GET.
            params["ResponseContentType"] = _guess_mime(k)
            params["ResponseContentDisposition"] = "inline"
        out.append(s3.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=TTL))
    return out

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("platform_ids", nargs="*",
                   help="comparis platform_ids to mint URLs for")
    p.add_argument("--source", default=None,
                   choices=["comparis", "robinreal"],
                   help="restrict to one source")
    p.add_argument("--limit", type=int, default=None,
                   help="with --source and no ids, show URLs for first N listings")
    p.add_argument("--download", action="store_true",
                   help="mint download-style URLs (default is inline/viewable "
                        "in a browser tab)")
    args = p.parse_args()
    inline = not args.download

    s3 = boto3.client("s3")
    out: dict = {}

    if args.platform_ids:
        sources = [args.source] if args.source else ["comparis", "robinreal"]
        for pid in args.platform_ids:
            out[pid] = {}
            for src in sources:
                urls = urls_for_listing(s3, src, pid, inline=inline)
                if urls:
                    out[pid][src] = urls
    elif args.source:
        # sample the first N platform_ids for this source
        from itertools import islice
        paginator = s3.get_paginator("list_objects_v2")
        seen_pids: list[str] = []
        for page in paginator.paginate(
            Bucket=BUCKET,
            Prefix=f"raw/downloads/prod/{args.source}/images/",
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                # cp["Prefix"] ends with "platform_id=<id>/"
                pid = cp["Prefix"].rstrip("/").split("platform_id=")[-1]
                seen_pids.append(pid)
                if args.limit and len(seen_pids) >= args.limit:
                    break
            if args.limit and len(seen_pids) >= args.limit:
                break
        for pid in seen_pids:
            out[pid] = {args.source: urls_for_listing(
                s3, args.source, pid, inline=inline)}
    else:
        p.error("pass platform_ids or --source")

    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
