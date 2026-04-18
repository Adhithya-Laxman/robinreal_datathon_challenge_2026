"""CLI: compute SigLIP2 image embeddings for every listing image.

Walks `downloads/prod/<source>/images/platform_id=*/` and writes L2-normalized
image features to sharded .npz files under `downloads/siglip2/`. Resumable:
on rerun, any image already present in an existing shard is skipped.

Usage:
    uv run python scripts/siglip2_embed.py                         # full run
    uv run python scripts/siglip2_embed.py --limit 32              # smoke test
    uv run python scripts/siglip2_embed.py --model google/siglip2-large-patch16-256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision.device import pick_device  # noqa: E402
from vision.embed import run_embedding  # noqa: E402

DEFAULT_IMG_ROOT = ROOT / "downloads" / "prod"
DEFAULT_SRED_ROOT = ROOT / "raw_data" / "sred_images"
DEFAULT_OUT_DIR = ROOT / "downloads" / "siglip2"
DEFAULT_MODEL = "google/siglip2-base-patch16-256"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--sred-root", type=Path, default=DEFAULT_SRED_ROOT,
                   help="Directory of SRED montage images (split into 4 tiles each).")
    p.add_argument("--no-sred", action="store_true", help="Skip SRED images.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N new images/tiles (for smoke tests).")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = p.parse_args()

    device = pick_device()
    print(f"[siglip2] device={device} model={args.model}")
    run_embedding(
        img_root=args.img_root,
        out_dir=args.out_dir,
        model_name=args.model,
        device=device,
        sred_root=None if args.no_sred else args.sred_root,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        limit=args.limit,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
