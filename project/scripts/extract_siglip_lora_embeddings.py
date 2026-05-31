"""
Re-extract SigLIP embeddings using the LoRA-adapted vision tower from
finetune_siglip_lora.py. Mirrors extract_siglip_embeddings.py I/O contract so
that downstream code only needs --backbone siglip_lora to switch.

This script reads from the same pre-computed pixel-value cache that the LoRA
trainer uses (built by cache_pixel_values.py) for street imagery — bit-exact
deterministic preprocessing, ~60x faster than re-decoding JPEGs from /mnt/c.
NAIP imagery still streams from /mnt/c (only ~14k tiles per city, acceptable).

Outputs (project/artifacts/embeddings/):
    siglip_lora_street_{city}.npy             (N_locked, 1152) mean-pooled
    siglip_lora_street_{city}_per_heading.npy (N_locked, 4, 1152)
    siglip_lora_street_{city}_ids.npy         (N_locked,) int point_ids
    siglip_lora_naip_{city}.npy               (N_locked, 1152)
    siglip_lora_naip_{city}_ids.npy           (N_locked,) int point_ids

Run (WSL, GPU):
    python project/scripts/extract_siglip_lora_embeddings.py \\
        --source all --city all \\
        --adapter project/artifacts/models/siglip_lora_v1/adapter \\
        --cache-dir ~/walkclip_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR  = REPO_ROOT / "project/data/processed/locks"
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
IMG_ROOT  = REPO_ROOT / "project/data/raw/imagery"

NAIP_DIRS = {
    "msp":         IMG_ROOT / "aerial_naip" / "Minneapolis",
    "seattle":     IMG_ROOT / "aerial_naip" / "Seattle",
    "dc":          IMG_ROOT / "aerial_naip" / "DC",
    "pittsburgh":  IMG_ROOT / "aerial_naip" / "Pittsburgh",
}
STREET_DIRS = {
    # Only used for cities not in the pixel-value cache (currently only Pittsburgh).
    "pittsburgh":  IMG_ROOT / "street_view_mapillary" / "Pittsburgh",
}
CITY_LOCK = {
    "msp":         "msp_v2_final_lock_ids.txt",
    "seattle":     "seattle_v2_final_lock_ids.txt",
    "dc":          "dc_v2_final_lock_ids.txt",
    "pittsburgh":  "pittsburgh_v2_final_lock_ids.txt",
}
# Cities whose street images are in the pixel-value cache (built by cache_pixel_values.py).
# Pittsburgh is absent because it was added after the cache was built; it falls back to
# direct JPEG loading via extract_street_direct().
CACHE_CITIES = {"msp", "seattle", "dc"}

MODEL_ID  = "google/siglip-so400m-patch14-384"
HEADINGS  = [0, 90, 180, 270]
EMBED_DIM = 1152
BACKBONE_TAG = "siglip_lora"


def load_lora_vision(adapter_dir: Path, device: torch.device) -> torch.nn.Module:
    from peft import PeftModel

    logging.info("Loading base SigLIP-SO400M …")
    base = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).vision_model
    logging.info("Attaching LoRA adapter from %s …", adapter_dir)
    return PeftModel.from_pretrained(base, str(adapter_dir)).to(device).eval()


@torch.inference_mode()
def forward_batch_pixels(pixel_values: torch.Tensor, model, device) -> np.ndarray:
    px = pixel_values.to(device, dtype=torch.float16, non_blocking=True)
    out = model(pixel_values=px)
    embs = out.pooler_output.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def extract_street_from_cache(
    city: str,
    cache_arr: np.ndarray,
    cache_index: pd.DataFrame,
    lock_ids: list[int],
    model,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    out_mean = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}.npy"
    out_per  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_per_heading.npy"
    out_ids  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_ids.npy"
    if not overwrite and out_mean.exists() and out_per.exists():
        logging.info("[%s] street exists, skipping (--overwrite to redo)", city); return

    sub = cache_index[cache_index["city"] == city].copy()
    pid_to_pos = {pid: pos for pos, pid in enumerate(lock_ids)}
    sub = sub[sub["point_id"].isin(pid_to_pos)].copy()
    sub["pt_pos"]  = sub["point_id"].map(pid_to_pos).astype(int)
    sub["h_idx"]   = sub["heading"].map({h: i for i, h in enumerate(HEADINGS)}).astype(int)

    n = len(lock_ids)
    mean_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    per_embs  = np.zeros((n, 4, EMBED_DIM), dtype=np.float32)
    heading_present = np.zeros((n, 4), dtype=bool)

    cidxs   = sub["cache_idx"].to_numpy()
    pt_pos  = sub["pt_pos"].to_numpy()
    h_idxs  = sub["h_idx"].to_numpy()

    t0 = time.time()
    for i in tqdm(range(0, len(cidxs), batch_size), desc=f"street/{city}", unit="batch", dynamic_ncols=True):
        sl = slice(i, i + batch_size)
        batch_pixels = torch.from_numpy(np.array(cache_arr[cidxs[sl]]))   # (B, 3, 384, 384) fp16
        embs = forward_batch_pixels(batch_pixels, model, device)
        for j, e in enumerate(embs):
            p, h = int(pt_pos[i + j]), int(h_idxs[i + j])
            per_embs[p, h] = e
            heading_present[p, h] = True

    for p in range(n):
        present = heading_present[p]
        if present.any():
            mean_embs[p] = per_embs[p][present].mean(axis=0)

    logging.info("[%s] street: %d cache rows in %.1fs", city, len(cidxs), time.time() - t0)
    np.save(out_mean, mean_embs); np.save(out_per, per_embs)
    np.save(out_ids,  np.array(lock_ids, dtype=np.int32))
    logging.info("Saved: %s %s %s", out_mean.name, out_per.name, out_ids.name)


def extract_street_direct(
    city: str,
    lock_ids: list[int],
    processor: AutoProcessor,
    model,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    """Fallback street extractor for cities not in the pixel-value cache (Pittsburgh).

    Loads JPEGs directly and runs the SigLIP processor inline — slower than the
    cache path but bit-compatible (same processor, same model).
    """
    out_mean = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}.npy"
    out_per  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_per_heading.npy"
    out_ids  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_ids.npy"
    if not overwrite and out_mean.exists() and out_per.exists():
        logging.info("[%s] street exists, skipping (--overwrite to redo)", city)
        return

    img_dir = STREET_DIRS[city]
    n = len(lock_ids)
    mean_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    per_embs  = np.zeros((n, 4, EMBED_DIM), dtype=np.float32)
    heading_present = np.zeros((n, 4), dtype=bool)

    entries: list[tuple[int, int, Path]] = []
    for pt_pos, pid in enumerate(lock_ids):
        for h_idx, heading in enumerate(HEADINGS):
            p = img_dir / f"{pid}_{heading}.jpg"
            if p.exists():
                entries.append((pt_pos, h_idx, p))

    t0 = time.time()
    for i in tqdm(range(0, len(entries), batch_size), desc=f"street_direct/{city}",
                  unit="batch", dynamic_ncols=True):
        chunk = entries[i: i + batch_size]
        imgs = [Image.open(p).convert("RGB") for _, _, p in chunk]
        pv = processor(images=imgs, return_tensors="pt", padding=True)["pixel_values"]
        embs = forward_batch_pixels(pv, model, device)
        for j, (pt_pos, h_idx, _) in enumerate(chunk):
            per_embs[pt_pos, h_idx] = embs[j]
            heading_present[pt_pos, h_idx] = True

    for pt_pos in range(n):
        present = heading_present[pt_pos]
        if present.any():
            mean_embs[pt_pos] = per_embs[pt_pos][present].mean(axis=0)

    logging.info("[%s] street_direct: %d images in %.1fs", city, len(entries), time.time() - t0)
    np.save(out_mean, mean_embs)
    np.save(out_per, per_embs)
    np.save(out_ids, np.array(lock_ids, dtype=np.int32))
    logging.info("Saved: %s %s %s", out_mean.name, out_per.name, out_ids.name)


def extract_naip(
    city: str,
    lock_ids: list[int],
    processor: AutoProcessor,
    model,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    out_emb = EMBED_DIR / f"{BACKBONE_TAG}_naip_{city}.npy"
    out_ids = EMBED_DIR / f"{BACKBONE_TAG}_naip_{city}_ids.npy"
    if not overwrite and out_emb.exists():
        logging.info("[%s] naip exists, skipping", city); return

    img_dir = NAIP_DIRS[city]
    valid = [(pid, img_dir / f"{pid}.jpg") for pid in lock_ids if (img_dir / f"{pid}.jpg").exists()]
    n = len(valid)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    ids  = np.array([v[0] for v in valid], dtype=np.int32)

    t0 = time.time()
    for i in tqdm(range(0, n, batch_size), desc=f"naip/{city}", unit="batch", dynamic_ncols=True):
        chunk = valid[i:i+batch_size]
        imgs  = [Image.open(p).convert("RGB") for _, p in chunk]
        pv    = processor(images=imgs, return_tensors="pt", padding=True)["pixel_values"]
        embs[i:i+len(chunk)] = forward_batch_pixels(pv, model, device)

    logging.info("[%s] naip: %d tiles in %.1fs", city, n, time.time() - t0)
    np.save(out_emb, embs); np.save(out_ids, ids)
    logging.info("Saved: %s %s", out_emb.name, out_ids.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract SigLIP-LoRA embeddings (Stage E+ Table 4 row 3)")
    p.add_argument("--adapter", required=True, help="Path to PEFT adapter dir (saved by finetune_siglip_lora.py)")
    p.add_argument("--cache-dir", default=str(Path.home() / "walkclip_cache"),
                   help="Pixel-value cache (used for street; NAIP still reads from /mnt/c)")
    p.add_argument("--source", choices=["street", "naip", "all"], default="all")
    p.add_argument("--city",   choices=["msp", "seattle", "dc", "pittsburgh", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backbone-tag", default="siglip_lora",
                   help="Filename prefix for embeddings (default: siglip_lora). For "
                        "leave-one-city-out adapters use e.g. siglip_lora_loco_dc — the trainer "
                        "loads from {tag}_{source}_{city}.npy.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    global BACKBONE_TAG
    BACKBONE_TAG = args.backbone_tag
    logging.info("Embedding filename prefix: %s_*.npy", BACKBONE_TAG)

    cities  = ["msp", "seattle", "dc", "pittsburgh"] if args.city == "all" else [args.city]
    sources = ["street", "naip"]                     if args.source == "all" else [args.source]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for embedding extraction.")

    model = load_lora_vision(Path(args.adapter), device)

    # Processor is needed for NAIP (all cities) and Pittsburgh street (direct loading).
    needs_processor = "naip" in sources or (
        "street" in sources and any(c not in CACHE_CITIES for c in cities)
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID) if needs_processor else None

    cache_arr = None
    cache_index = None
    cache_cities_requested = [c for c in cities if c in CACHE_CITIES]
    if "street" in sources and cache_cities_requested:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        arr_path = cache_dir / "siglip_pixels_street_384.npy"
        idx_path = cache_dir / "siglip_pixels_street_384_index.csv"
        if not arr_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"Street cache missing at {cache_dir}. Run cache_pixel_values.py first."
            )
        cache_arr   = np.load(arr_path, mmap_mode="r")
        cache_index = pd.read_csv(idx_path)

    for city in cities:
        lock_ids = [int(x) for x in (LOCK_DIR / CITY_LOCK[city]).read_text().strip().splitlines()]
        logging.info("[%s] %d lock IDs", city, len(lock_ids))
        for source in sources:
            if source == "street":
                if city in CACHE_CITIES:
                    extract_street_from_cache(city, cache_arr, cache_index, lock_ids,
                                              model, device, args.batch_size, args.overwrite)
                else:
                    # Pittsburgh: not in cache → direct JPEG loading
                    extract_street_direct(city, lock_ids, processor, model, device,
                                          args.batch_size, args.overwrite)
            else:
                extract_naip(city, lock_ids, processor, model, device, args.batch_size, args.overwrite)

    logging.info("LoRA embedding extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
