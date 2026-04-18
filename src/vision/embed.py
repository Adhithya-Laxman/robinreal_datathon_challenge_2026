from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# SRED montages are always 224×224 with a 2×2 grid of 112×112 tiles.
SRED_TILE_POSITIONS: list[tuple[int, int, int, int]] = [
    (0,   0,   112, 112),   # tile0: top-left
    (112, 0,   224, 112),   # tile1: top-right
    (0,   112, 112, 224),   # tile2: bottom-left
    (112, 112, 224, 224),   # tile3: bottom-right
]


def discover_images(img_root: Path) -> list[Path]:
    return sorted(img_root.rglob("*.jpg"))


def already_embedded(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for shard in sorted(out_dir.glob("shard_*.npz")):
        z = np.load(shard, allow_pickle=True)
        done.update(str(p) for p in z["paths"])
    return done


def load_model(model_name: str, device: torch.device):
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def embed_image_batch(model, processor, images, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def _iter_tasks(
    img_root: Path,
    sred_root: Path | None,
    done: set[str],
) -> Iterator[tuple[Image.Image, str]]:
    """Yield (PIL image, rel_path) for every image/tile not yet embedded."""
    for path in discover_images(img_root):
        rel = str(path.relative_to(img_root))
        if rel in done:
            continue
        try:
            yield Image.open(path).convert("RGB"), rel
        except Exception as e:
            print(f"[siglip2] skip {path}: {e}", file=sys.stderr)

    if sred_root is None or not sred_root.exists():
        return
    for path in sorted(sred_root.glob("*.jpeg")):
        stem = path.stem
        pending = [
            (n, f"sred/{stem}_tile{n}")
            for n in range(4)
            if f"sred/{stem}_tile{n}" not in done
        ]
        if not pending:
            continue
        try:
            montage = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[siglip2] skip {path}: {e}", file=sys.stderr)
            continue
        for n, rel in pending:
            yield montage.crop(SRED_TILE_POSITIONS[n]), rel


def _count_todo(img_root: Path, sred_root: Path | None, done: set[str]) -> int:
    regular = sum(
        1 for p in discover_images(img_root)
        if str(p.relative_to(img_root)) not in done
    )
    sred = 0
    if sred_root and sred_root.exists():
        for path in sred_root.glob("*.jpeg"):
            stem = path.stem
            sred += sum(
                1 for n in range(4)
                if f"sred/{stem}_tile{n}" not in done
            )
    return regular + sred


def run_embedding(
    img_root: Path,
    out_dir: Path,
    model_name: str,
    device: torch.device,
    sred_root: Path | None = None,
    batch_size: int = 16,
    shard_size: int = 1024,
    limit: int | None = None,
    dtype: str = "float16",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    done = already_embedded(out_dir)
    if done:
        print(f"[siglip2] {len(done)} already embedded, skipping")

    total = _count_todo(img_root, sred_root, done)
    if limit is not None:
        total = min(total, limit)
    print(f"[siglip2] {total} images/tiles to embed")
    if total == 0:
        return

    print("[siglip2] loading model…")
    model, processor = load_model(model_name, device)

    np_dtype = np.float16 if dtype == "float16" else np.float32
    existing = sorted(out_dir.glob("shard_*.npz"))
    next_idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0

    shard_paths: list[str] = []
    shard_embs: list[np.ndarray] = []

    def flush() -> None:
        nonlocal next_idx, shard_paths, shard_embs
        if not shard_paths:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"shard_{next_idx:05d}.npz"
        embs = np.concatenate(shard_embs, axis=0).astype(np_dtype)
        np.savez_compressed(
            out,
            paths=np.array(shard_paths, dtype=object),
            embeddings=embs,
            model=np.array(model_name),
        )
        print(f"[siglip2] wrote {out} ({len(shard_paths)} × {embs.shape[1]})")
        shard_paths = []
        shard_embs = []
        next_idx += 1

    tasks = _iter_tasks(img_root, sred_root, done)
    if limit is not None:
        tasks = islice(tasks, limit)

    pbar = tqdm(total=total, desc="embed", unit="img")
    batch_imgs: list[Image.Image] = []
    batch_rels: list[str] = []

    for img, rel in tasks:
        batch_imgs.append(img)
        batch_rels.append(rel)
        if len(batch_imgs) >= batch_size:
            arr = embed_image_batch(model, processor, batch_imgs, device)
            shard_embs.append(arr)
            shard_paths.extend(batch_rels)
            pbar.update(len(batch_imgs))
            batch_imgs, batch_rels = [], []
            if len(shard_paths) >= shard_size:
                flush()

    if batch_imgs:  # flush remaining partial batch
        arr = embed_image_batch(model, processor, batch_imgs, device)
        shard_embs.append(arr)
        shard_paths.extend(batch_rels)
        pbar.update(len(batch_imgs))
    flush()
    pbar.close()
    print("[siglip2] done")
