"""
Stage E Step 1: Extract SigLIP 2 embeddings from street-view and NAIP imagery.

Model: google/siglip-so400m-patch14-384
  - 400M parameter ViT-SO400M vision encoder
  - Output: 1152-dim L2-normalised embedding per image
  - Input: 384×384 RGB, SigLIP 2 normalisation

Street-view: 4 headings per point → 4 embeddings → mean-pooled → (N, 1152)
NAIP:        1 tile per point → (N, 1152)

Per-heading embeddings are also saved for ablation (max-pool, concat, attention).

Outputs (project/artifacts/embeddings/):
    siglip_street_{city}.npy          # (4824, 1152) mean-pooled, lock-ordered
    siglip_street_{city}_per_heading.npy  # (4824, 4, 1152) all headings
    siglip_street_{city}_ids.npy      # (4824,) int point_ids in row order
    siglip_naip_{city}.npy            # (N_locked, 1152)
    siglip_naip_{city}_ids.npy        # (N_locked,) int point_ids

Run:
    & ".venv\Scripts\python.exe" project/scripts/extract_siglip_embeddings.py --source street --city msp
    & ".venv\Scripts\python.exe" project/scripts/extract_siglip_embeddings.py --source street --city seattle
    & ".venv\Scripts\python.exe" project/scripts/extract_siglip_embeddings.py --source naip --city msp
    & ".venv\Scripts\python.exe" project/scripts/extract_siglip_embeddings.py --source naip --city seattle

    # Run all four in sequence (fastest on single GPU):
    & ".venv\Scripts\python.exe" project/scripts/extract_siglip_embeddings.py --source all --city all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

REPO_ROOT   = Path(__file__).resolve().parents[2]
LOCK_FILE   = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"
EMBED_DIR   = REPO_ROOT / "project" / "artifacts" / "embeddings"
IMG_ROOT    = REPO_ROOT / "project" / "data" / "raw" / "imagery"

STREET_DIRS = {
    "msp":         IMG_ROOT / "street_view_mapillary" / "Minneapolis",
    "seattle":     IMG_ROOT / "street_view_mapillary" / "Seattle",
    "dc":          IMG_ROOT / "street_view_mapillary" / "DC",
    "pittsburgh":  IMG_ROOT / "street_view_mapillary" / "Pittsburgh",
}
NAIP_DIRS = {
    "msp":         IMG_ROOT / "aerial_naip" / "Minneapolis",
    "seattle":     IMG_ROOT / "aerial_naip" / "Seattle",
    "dc":          IMG_ROOT / "aerial_naip" / "DC",
    "pittsburgh":  IMG_ROOT / "aerial_naip" / "Pittsburgh",
}
CITY_LOCK_FILES: dict[str, str] = {
    "msp":         "msp_v2_final_lock_ids.txt",
    "seattle":     "seattle_v2_final_lock_ids.txt",
    "dc":          "dc_v2_final_lock_ids.txt",
    "pittsburgh":  "pittsburgh_v2_final_lock_ids.txt",
}

MODEL_ID = "google/siglip-so400m-patch14-384"
HEADINGS = [0, 90, 180, 270]
EMBED_DIM = 1152


def load_model(device: torch.device) -> tuple[AutoModel, AutoProcessor]:
    logging.info("Loading %s …", MODEL_ID)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = model.vision_model.to(device).eval()
    logging.info("Model loaded in %.1fs  (device=%s)", time.time() - t0, device)
    return model, processor


@torch.inference_mode()
def embed_batch(
    images: list[Image.Image],
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, 1152), float32."""
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
    outputs = model(pixel_values=pixel_values)
    # pooler_output: (N, 1152) — CLS token after projection
    embs = outputs.pooler_output.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def load_image_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_street(
    city: str,
    lock_ids: list[int],
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
) -> None:
    img_dir = STREET_DIRS[city]
    out_mean = EMBED_DIR / f"siglip_street_{city}.npy"
    out_per  = EMBED_DIR / f"siglip_street_{city}_per_heading.npy"
    out_ids  = EMBED_DIR / f"siglip_street_{city}_ids.npy"

    if out_mean.exists() and out_per.exists():
        logging.info("Street embeddings for %s already exist — skipping. Use --overwrite to redo.", city)
        return

    n = len(lock_ids)
    mean_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    per_embs  = np.zeros((n, 4, EMBED_DIM), dtype=np.float32)
    missing_points: list[int] = []

    # Build flat batch list: [(point_idx, heading_idx, path), ...]
    # Process in batches of batch_size images regardless of point boundaries
    entries: list[tuple[int, int, Path]] = []
    for pt_idx, pid in enumerate(lock_ids):
        for h_idx, heading in enumerate(HEADINGS):
            p = img_dir / f"{pid}_{heading}.jpg"
            if p.exists():
                entries.append((pt_idx, h_idx, p))
            else:
                missing_points.append(pid)

    if missing_points:
        logging.warning("%s street: %d heading images missing (points: %s…)",
                        city, len(missing_points), missing_points[:5])

    # Track per-point heading embeddings to average
    # Use accumulator: sum_embs[pt_idx][h_idx] = embedding or zeros
    heading_present = np.zeros((n, 4), dtype=bool)

    t0 = time.time()
    for batch_start in tqdm(range(0, len(entries), batch_size),
                            desc=f"street/{city}", unit="batch", dynamic_ncols=True):
        batch = entries[batch_start: batch_start + batch_size]
        images = [load_image_rgb(e[2]) for e in batch]
        embs = embed_batch(images, model, processor, device)  # (B, 1152)
        for (pt_idx, h_idx, _), emb in zip(batch, embs):
            per_embs[pt_idx, h_idx] = emb
            heading_present[pt_idx, h_idx] = True

    # Mean-pool over available headings
    for pt_idx in range(n):
        present = heading_present[pt_idx]
        if present.any():
            mean_embs[pt_idx] = per_embs[pt_idx][present].mean(axis=0)
        # If no headings (should not happen post-lock): leave as zeros

    elapsed = time.time() - t0
    total_imgs = len(entries)
    logging.info(
        "%s street: %d images in %.1fs (%.1f ms/img)  %d points",
        city, total_imgs, elapsed, elapsed / max(total_imgs, 1) * 1000, n,
    )

    np.save(out_mean, mean_embs)
    np.save(out_per,  per_embs)
    np.save(out_ids,  np.array(lock_ids, dtype=np.int32))
    logging.info("Saved: %s  %s  %s", out_mean.name, out_per.name, out_ids.name)


def extract_naip(
    city: str,
    lock_ids: list[int],
    model: AutoModel,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
) -> None:
    img_dir = NAIP_DIRS[city]
    out_emb = EMBED_DIR / f"siglip_naip_{city}.npy"
    out_ids = EMBED_DIR / f"siglip_naip_{city}_ids.npy"

    if out_emb.exists():
        logging.info("NAIP embeddings for %s already exist — skipping. Use --overwrite to redo.", city)
        return

    # Only embed points that are in lock AND have a NAIP tile
    valid: list[tuple[int, Path]] = []
    missing: list[int] = []
    for pid in lock_ids:
        p = img_dir / f"{pid}.jpg"
        if p.exists():
            valid.append((pid, p))
        else:
            missing.append(pid)

    if missing:
        logging.warning("%s NAIP: %d tiles missing from lock: %s", city, len(missing), missing)

    n = len(valid)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    ids  = np.array([v[0] for v in valid], dtype=np.int32)

    t0 = time.time()
    for batch_start in tqdm(range(0, n, batch_size),
                            desc=f"naip/{city}", unit="batch", dynamic_ncols=True):
        batch = valid[batch_start: batch_start + batch_size]
        images = [load_image_rgb(p) for _, p in batch]
        batch_embs = embed_batch(images, model, processor, device)
        embs[batch_start: batch_start + len(batch)] = batch_embs

    elapsed = time.time() - t0
    logging.info(
        "%s NAIP: %d tiles in %.1fs (%.1f ms/img)",
        city, n, elapsed, elapsed / max(n, 1) * 1000,
    )

    np.save(out_emb, embs)
    np.save(out_ids, ids)
    logging.info("Saved: %s  %s", out_emb.name, out_ids.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract SigLIP 2 embeddings for WalkCLIP v2")
    p.add_argument("--source", choices=["street", "naip", "all"], default="all",
                   help="Which imagery source to embed")
    p.add_argument("--city", choices=["msp", "seattle", "dc", "pittsburgh", "all"], default="all",
                   help="Which city to process")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Images per GPU batch (default 64, reduce if OOM)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if output files already exist")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    cities   = ["msp", "seattle", "dc", "pittsburgh"] if args.city == "all" else [args.city]

    # Per-city lock files (dc has its own; msp/seattle fall back to shared lock)
    def _load_lock(city: str) -> list[int]:
        primary = REPO_ROOT / "project" / "data" / "processed" / "locks" / CITY_LOCK_FILES[city]
        if primary.exists():
            return [int(x) for x in primary.read_text().strip().splitlines()]
        if LOCK_FILE.exists():
            return [int(x) for x in LOCK_FILE.read_text().strip().splitlines()]
        raise FileNotFoundError(f"No lock file found for {city}. Run build_v2_final_lock.py first.")

    if args.overwrite:
        for f in EMBED_DIR.glob("siglip_*.npy"):
            f.unlink()
        logging.info("--overwrite: cleared existing embeddings")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA not available — running on CPU, will be slow")

    model, processor = load_model(device)

    sources  = ["street", "naip"] if args.source == "all" else [args.source]

    for city in cities:
        lock_ids = _load_lock(city)
        logging.info("%s: %d lock IDs", city, len(lock_ids))
        for source in sources:
            if source == "street":
                extract_street(city, lock_ids, model, processor, device, args.batch_size)
            else:
                extract_naip(city, lock_ids, model, processor, device, args.batch_size)


    logging.info("Embedding extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
