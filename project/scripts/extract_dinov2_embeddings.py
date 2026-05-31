"""
Extract DINOv2 ViT-L/14 embeddings for WalkCLIP SRC Stage 2 (Extension #4).

Model: facebook/dinov2-large
  - Vision-only self-supervised ViT-L/14 (no text encoder)
  - Output: 1024-dim CLS token (pooler_output) per image
  - Trained with self-distillation with no labels (DINOv2, Oquab et al. 2023)

Street-view: 4 headings → mean-pooled → (N, 1024)
NAIP:        1 tile     → (N, 1024)

Outputs (project/artifacts/embeddings/):
    dinov2_street_{city}.npy            # (N, 1024) mean-pooled
    dinov2_street_{city}_per_heading.npy  # (N, 4, 1024)
    dinov2_street_{city}_ids.npy        # (N,) int32
    dinov2_naip_{city}.npy              # (N_locked, 1024)
    dinov2_naip_{city}_ids.npy

Run (WSL, CUDA):
    .venv_wsl/bin/python project/scripts/extract_dinov2_embeddings.py --source all --cities msp seattle dc pittsburgh
    .venv_wsl/bin/python project/scripts/extract_dinov2_embeddings.py --source naip --cities pittsburgh
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
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR  = REPO_ROOT / "project/data/processed/locks"
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
IMG_ROOT  = REPO_ROOT / "project/data/raw/imagery"

STREET_DIRS: dict[str, Path] = {
    "msp":         IMG_ROOT / "street_view_mapillary/Minneapolis",
    "seattle":     IMG_ROOT / "street_view_mapillary/Seattle",
    "dc":          IMG_ROOT / "street_view_mapillary/DC",
    "pittsburgh":  IMG_ROOT / "street_view_mapillary/Pittsburgh",
}
NAIP_DIRS: dict[str, Path] = {
    "msp":         IMG_ROOT / "aerial_naip/Minneapolis",
    "seattle":     IMG_ROOT / "aerial_naip/Seattle",
    "dc":          IMG_ROOT / "aerial_naip/DC",
    "pittsburgh":  IMG_ROOT / "aerial_naip/Pittsburgh",
}
CITY_LOCK_FILES: dict[str, str] = {
    "msp":         "msp_v2_final_lock_ids.txt",
    "seattle":     "seattle_v2_final_lock_ids.txt",
    "dc":          "dc_v2_final_lock_ids.txt",
    "pittsburgh":  "pittsburgh_v2_final_lock_ids.txt",
}

MODEL_ID = "facebook/dinov2-large"
BACKBONE_TAG = "dinov2"
HEADINGS = [0, 90, 180, 270]
EMBED_DIM = 1024


def load_lock_ids(city: str) -> list[int]:
    lock_path = LOCK_DIR / CITY_LOCK_FILES[city]
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Lock file not found: {lock_path}. Run build_v2_final_lock.py --include-cities {city} first."
        )
    return [int(x) for x in lock_path.read_text().strip().splitlines()]


def load_model(device: torch.device) -> tuple[AutoModel, AutoImageProcessor]:
    logging.info("Loading %s …", MODEL_ID)
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = model.to(device).eval()
    logging.info("Model loaded in %.1fs  (device=%s)", time.time() - t0, device)
    return model, processor


@torch.inference_mode()
def embed_batch(
    images: list[Image.Image],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
) -> np.ndarray:
    """Return L2-normalised float32 CLS embeddings, shape (N, 1024)."""
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
    outputs = model(pixel_values=pixel_values)
    # pooler_output = linear projection of CLS token — the canonical DINOv2 image feature
    embs = outputs.pooler_output.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def extract_street(
    city: str,
    lock_ids: list[int],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    out_mean = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}.npy"
    out_per  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_per_heading.npy"
    out_ids  = EMBED_DIR / f"{BACKBONE_TAG}_street_{city}_ids.npy"

    if out_mean.exists() and out_per.exists() and not overwrite:
        logging.info("%s street already exists — skipping. Use --overwrite to redo.", city)
        return

    img_dir = STREET_DIRS[city]
    n = len(lock_ids)
    mean_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    per_embs  = np.zeros((n, 4, EMBED_DIM), dtype=np.float32)
    heading_present = np.zeros((n, 4), dtype=bool)

    entries: list[tuple[int, int, Path]] = []
    for pt_idx, pid in enumerate(lock_ids):
        for h_idx, heading in enumerate(HEADINGS):
            p = img_dir / f"{pid}_{heading}.jpg"
            if p.exists():
                entries.append((pt_idx, h_idx, p))

    if not entries:
        logging.error("%s: no street images found in %s", city, img_dir)
        return

    t0 = time.time()
    for start in tqdm(range(0, len(entries), batch_size),
                      desc=f"{BACKBONE_TAG}_street/{city}", unit="batch", dynamic_ncols=True):
        batch = entries[start: start + batch_size]
        images = [Image.open(e[2]).convert("RGB") for e in batch]
        embs = embed_batch(images, model, processor, device)
        for (pt_idx, h_idx, _), emb in zip(batch, embs):
            per_embs[pt_idx, h_idx] = emb
            heading_present[pt_idx, h_idx] = True

    for pt_idx in range(n):
        present = heading_present[pt_idx]
        if present.any():
            mean_embs[pt_idx] = per_embs[pt_idx][present].mean(axis=0)

    elapsed = time.time() - t0
    logging.info("%s street: %d images in %.1fs (%.1f ms/img)",
                 city, len(entries), elapsed, elapsed / max(len(entries), 1) * 1000)
    np.save(out_mean, mean_embs)
    np.save(out_per,  per_embs)
    np.save(out_ids,  np.array(lock_ids, dtype=np.int32))
    logging.info("Saved: %s  %s  %s", out_mean.name, out_per.name, out_ids.name)


def extract_naip(
    city: str,
    lock_ids: list[int],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    out_emb = EMBED_DIR / f"{BACKBONE_TAG}_naip_{city}.npy"
    out_ids = EMBED_DIR / f"{BACKBONE_TAG}_naip_{city}_ids.npy"

    if out_emb.exists() and not overwrite:
        logging.info("%s NAIP already exists — skipping. Use --overwrite to redo.", city)
        return

    img_dir = NAIP_DIRS[city]
    valid: list[tuple[int, Path]] = []
    for pid in lock_ids:
        p = img_dir / f"{pid}.jpg"
        if p.exists():
            valid.append((pid, p))

    n = len(valid)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)

    t0 = time.time()
    for start in tqdm(range(0, n, batch_size),
                      desc=f"{BACKBONE_TAG}_naip/{city}", unit="batch", dynamic_ncols=True):
        batch = valid[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for _, p in batch]
        embs[start: start + len(batch)] = embed_batch(images, model, processor, device)

    elapsed = time.time() - t0
    logging.info("%s NAIP: %d tiles in %.1fs (%.1f ms/img)",
                 city, n, elapsed, elapsed / max(n, 1) * 1000)
    np.save(out_emb, embs)
    np.save(out_ids, np.array([v[0] for v in valid], dtype=np.int32))
    logging.info("Saved: %s  %s", out_emb.name, out_ids.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DINOv2 ViT-L/14 embeddings for WalkCLIP SRC")
    p.add_argument("--cities", nargs="+",
                   choices=["msp", "seattle", "dc", "pittsburgh"], required=True)
    p.add_argument("--source", choices=["street", "naip", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Images per GPU batch (reduce to 32 if OOM)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA not available — running on CPU, will be very slow")

    model, processor = load_model(device)
    sources = ["street", "naip"] if args.source == "all" else [args.source]

    for city in args.cities:
        lock_ids = load_lock_ids(city)
        logging.info("%s: %d lock IDs", city, len(lock_ids))
        for source in sources:
            if source == "street":
                extract_street(city, lock_ids, model, processor, device, args.batch_size, args.overwrite)
            else:
                extract_naip(city, lock_ids, model, processor, device, args.batch_size, args.overwrite)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
