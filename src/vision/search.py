from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

# One SigLIP2 text tower per process. Without this, `encode_text` called
# `from_pretrained` on every query (~4 GB disk read + HF hub metadata checks),
# and a single DNS blip could close huggingface_hub's global httpx client —
# breaking VLM for the rest of the eval run ("client has been closed").
_siglip_processor: AutoProcessor | None = None
_siglip_model: AutoModel | None = None
_siglip_cached_name: str | None = None


def reset_siglip_text_cache() -> None:
    """Drop the in-memory SigLIP text stack so the next encode reloads.

    Call after a VLM failure so a later query is not stuck with a half-built
    model or a poisoned HF hub session.
    """
    global _siglip_processor, _siglip_model, _siglip_cached_name
    _siglip_processor = None
    _siglip_model = None
    _siglip_cached_name = None


def _load_siglip_text_stack(
    model_name: str,
    device: torch.device,
    *,
    local_files_only: bool,
) -> tuple[AutoProcessor, AutoModel]:
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).eval().to(device)
    return processor, model


def _ensure_siglip_text_stack(model_name: str, device: torch.device) -> None:
    """Populate module-level SigLIP processor+model once per model_name."""
    global _siglip_processor, _siglip_model, _siglip_cached_name

    if (
        _siglip_processor is not None
        and _siglip_model is not None
        and _siglip_cached_name == model_name
    ):
        return

    reset_siglip_text_cache()
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")

    try:
        logger.info("[siglip2] loading text tower %r (local_files_only=%s)", model_name, offline)
        proc, mdl = _load_siglip_text_stack(model_name, device, local_files_only=offline)
    except Exception as first_exc:
        if offline:
            raise
        logger.warning(
            "[siglip2] remote load failed (%s); retrying local_files_only=True",
            first_exc,
        )
        proc, mdl = _load_siglip_text_stack(model_name, device, local_files_only=True)

    _siglip_processor = proc
    _siglip_model = mdl
    _siglip_cached_name = model_name


def load_shards(shards_dir: Path) -> tuple[list[str], np.ndarray, str | None]:
    shards = sorted(shards_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shards_dir}; run the embed script first")
    paths: list[str] = []
    embs: list[np.ndarray] = []
    model_name: str | None = None
    for s in shards:
        z = np.load(s, allow_pickle=True)
        paths.extend(z["paths"].tolist())
        embs.append(z["embeddings"])
        if "model" in z.files and model_name is None:
            model_name = str(z["model"])
    return paths, np.concatenate(embs, axis=0).astype(np.float32), model_name


def extract_listing_id(rel_path: str) -> str:
    # comparis / robinreal: .../platform_id=29387655/xxx.jpg
    for seg in rel_path.split("/"):
        if seg.startswith("platform_id="):
            return seg
    # SRED tiles: sred/1154156_tile2 → "sred_1154156"
    if rel_path.startswith("sred/"):
        stem = Path(rel_path).name.split("_tile")[0]
        return f"sred_{stem}"
    return rel_path


def encode_text(prompt: str, model_name: str, device: torch.device) -> np.ndarray:
    _ensure_siglip_text_stack(model_name, device)
    assert _siglip_processor is not None and _siglip_model is not None
    processor, model = _siglip_processor, _siglip_model
    with torch.no_grad():
        inputs = processor(
            text=[prompt], padding="max_length", return_tensors="pt"
        ).to(device)
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype(np.float32)[0]


def rank(
    embs: np.ndarray,
    query: np.ndarray,
    paths: list[str],
    k: int,
    group_by_listing: bool = False,
) -> list[tuple[float, str]]:
    scores = embs @ query
    if group_by_listing:
        best: dict[str, tuple[float, int]] = {}
        for i, rel in enumerate(paths):
            lid = extract_listing_id(rel)
            s = float(scores[i])
            if lid not in best or s > best[lid][0]:
                best[lid] = (s, i)
        items = sorted(best.values(), key=lambda x: -x[0])[:k]
        return [(s, paths[i]) for s, i in items]
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), paths[i]) for i in top]


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "_", s)
    return s[:max_len] or "query"


def _load_image(rel: str, img_root: Path, sred_root: Path | None) -> Path | tuple[Path, int]:
    """Return a file path (regular) or (montage_path, tile_idx) for SRED tiles."""
    if rel.startswith("sred/"):
        stem_tile = Path(rel).name           # e.g. "1154156_tile2"
        parts = stem_tile.split("_tile")
        sred_id, tile_n = parts[0], int(parts[1])
        root = sred_root or img_root
        return (root / f"{sred_id}.jpeg", tile_n)
    return img_root / rel


def save_results(
    rows: list[tuple[float, str]],
    img_root: Path,
    results_dir: Path,
    prompt: str,
    model_name: str,
    sred_root: Path | None = None,
) -> None:
    from .embed import SRED_TILE_POSITIONS

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    manifest = {"prompt": prompt, "model": model_name, "results": []}
    for rank_idx, (score, rel) in enumerate(rows, 1):
        listing_id = extract_listing_id(rel)
        dst_name = f"rank{rank_idx:02d}_score{score:+.4f}_{listing_id}.jpg"
        dst = results_dir / dst_name

        src_info = _load_image(rel, img_root, sred_root)
        if isinstance(src_info, tuple):
            montage_path, tile_n = src_info
            img = Image.open(montage_path).convert("RGB")
            img.crop(SRED_TILE_POSITIONS[tile_n]).save(dst, quality=92)
        else:
            shutil.copy2(src_info, dst)

        manifest["results"].append(
            {
                "rank": rank_idx,
                "score": float(score),
                "listing_id": listing_id,
                "source_path": rel,
                "saved_as": dst_name,
            }
        )
    (results_dir / "results.json").write_text(json.dumps(manifest, indent=2))
    print(f"[siglip2] saved {len(rows)} images to {results_dir}")
