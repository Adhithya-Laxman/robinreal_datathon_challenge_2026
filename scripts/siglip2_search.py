"""CLI: query SigLIP2 embeddings with a text prompt, print top-k, and save them.

Saved outputs live under `results/<slug>/` (override with --results-dir).
Each image is renamed `rankNN_score+0.XXXX_platform_id=....jpg` and a
`results.json` manifest captures prompt, model, scores, and source paths.

Usage:
    uv run python scripts/siglip2_search.py --prompt "bright modern kitchen"
    uv run python scripts/siglip2_search.py --prompt "lake view" -k 20 --group-by-listing
    uv run python scripts/siglip2_search.py --prompt "balcony" --no-save     # print only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vision.device import pick_device  # noqa: E402
from vision.search import (  # noqa: E402
    encode_text,
    load_shards,
    rank,
    save_results,
    slugify,
)

DEFAULT_SHARDS_DIR = ROOT / "downloads" / "siglip2"
DEFAULT_IMG_ROOT = ROOT / "downloads" / "prod"
DEFAULT_SRED_ROOT = ROOT / "raw_data" / "sred_images"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_MODEL = "google/siglip2-base-patch16-256"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=True)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    p.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--results-dir", type=Path, default=None,
                   help=f"Default: {DEFAULT_RESULTS_DIR}/<slug-of-prompt>")
    p.add_argument("--sred-root", type=Path, default=DEFAULT_SRED_ROOT,
                   help="SRED montage directory (for result image extraction).")
    p.add_argument("--model", default=None,
                   help="Override text-tower model (default: read from shard metadata).")
    p.add_argument("--group-by-listing", action="store_true",
                   help="Return at most one image per listing (max-pool).")
    p.add_argument("--no-save", action="store_true",
                   help="Print-only, don't copy images to results/.")
    args = p.parse_args()

    paths, embs, shard_model = load_shards(args.shards_dir)
    model_name = args.model or shard_model or DEFAULT_MODEL
    device = pick_device()
    print(f"[siglip2] loaded {len(paths)} embeddings (dim={embs.shape[1]}) "
          f"from {args.shards_dir} | model={model_name} | device={device}")

    q = encode_text(args.prompt, model_name, device)
    rows = rank(embs, q, paths, k=args.k, group_by_listing=args.group_by_listing)

    print()
    print(f"Top {len(rows)} for: {args.prompt!r}")
    for i, (s, rel) in enumerate(rows, 1):
        print(f"  {i:>2}. {s:+.4f}  {rel}")

    if not args.no_save:
        results_dir = args.results_dir or (DEFAULT_RESULTS_DIR / slugify(args.prompt))
        save_results(rows, args.img_root, results_dir, args.prompt, model_name,
                     sred_root=args.sred_root)


if __name__ == "__main__":
    main()
