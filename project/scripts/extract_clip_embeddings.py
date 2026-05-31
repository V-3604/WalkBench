"""
Extract CLIP ViT-L/14 embeddings for WalkCLIP-3C backbone ablation (Table 3).

Mirrors extract_siglip_embeddings.py exactly; produces:
  project/artifacts/embeddings/clip_street_{city}.npy   (N, 1024) mean-pooled
  project/artifacts/embeddings/clip_street_{city}_ids.npy
  project/artifacts/embeddings/clip_naip_{city}.npy
  project/artifacts/embeddings/clip_naip_{city}_ids.npy

Run:
    & ".venv\Scripts\python.exe" project/scripts/extract_clip_embeddings.py --cities msp seattle dc
    & ".venv\Scripts\python.exe" project/scripts/extract_clip_embeddings.py --cities msp seattle dc --source street
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
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
IMG_ROOT = REPO_ROOT / "project/data/raw/imagery"

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
LOCK_FALLBACK = REPO_ROOT / "project/data/processed/locks/v2_final_lock_ids.txt"

MODEL_ID = "openai/clip-vit-large-patch14"
# vision_model.pooler_output is the raw 1024-dim ViT-L CLS token (pre-projection).
# The 768-dim contrastive projection is not used here — we want visual features, not
# alignment-optimised embeddings. The 3-city embeddings on disk are (N, 1024).
EMBED_DIM = 1024
HEADINGS = [0, 90, 180, 270]


def load_lock_ids(city: str) -> list[int]:
    primary = REPO_ROOT / "project/data/processed/locks" / CITY_LOCK_FILES[city]
    for p in [primary, LOCK_FALLBACK]:
        if p.exists():
            return [int(x) for x in p.read_text().strip().splitlines()]
    raise FileNotFoundError(f"No lock file for {city}.")


def load_model(device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    logging.info("Loading %s …", MODEL_ID)
    t0 = time.time()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = model.vision_model.to(device).eval()
    logging.info("Loaded in %.1fs", time.time() - t0)
    return model, processor


@torch.inference_mode()
def embed_batch(
    images: list[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
    outputs = model(pixel_values=pixel_values)
    embs = outputs.pooler_output.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def extract_street(
    city: str,
    lock_ids: list[int],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    img_dir = STREET_DIRS[city]
    out_mean = EMBED_DIR / f"clip_street_{city}.npy"
    out_ids = EMBED_DIR / f"clip_street_{city}_ids.npy"
    if out_mean.exists() and not overwrite:
        logging.info("clip_street_%s already exists — skipping. Use --overwrite.", city)
        return

    n = len(lock_ids)
    mean_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    heading_present = np.zeros((n, 4), dtype=bool)

    entries: list[tuple[int, int, Path]] = []
    for pt_idx, pid in enumerate(lock_ids):
        for h_idx, heading in enumerate(HEADINGS):
            p = img_dir / f"{pid}_{heading}.jpg"
            if p.exists():
                entries.append((pt_idx, h_idx, p))

    per_embs = np.zeros((n, 4, EMBED_DIM), dtype=np.float32)

    t0 = time.time()
    for batch_start in tqdm(range(0, len(entries), batch_size),
                            desc=f"clip_street/{city}", unit="batch"):
        batch = entries[batch_start: batch_start + batch_size]
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
    logging.info("%s CLIP street: %d images in %.1fs", city, len(entries), elapsed)
    np.save(out_mean, mean_embs)
    np.save(out_ids, np.array(lock_ids, dtype=np.int32))
    logging.info("Saved: %s  %s", out_mean.name, out_ids.name)


def extract_naip(
    city: str,
    lock_ids: list[int],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> None:
    img_dir = NAIP_DIRS[city]
    out_emb = EMBED_DIR / f"clip_naip_{city}.npy"
    out_ids = EMBED_DIR / f"clip_naip_{city}_ids.npy"
    if out_emb.exists() and not overwrite:
        logging.info("clip_naip_%s already exists — skipping. Use --overwrite.", city)
        return

    valid: list[tuple[int, Path]] = []
    for pid in lock_ids:
        p = img_dir / f"{pid}.jpg"
        if p.exists():
            valid.append((pid, p))

    n = len(valid)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)

    t0 = time.time()
    for batch_start in tqdm(range(0, n, batch_size), desc=f"clip_naip/{city}", unit="batch"):
        batch = valid[batch_start: batch_start + batch_size]
        images = [Image.open(p).convert("RGB") for _, p in batch]
        batch_embs = embed_batch(images, model, processor, device)
        embs[batch_start: batch_start + len(batch)] = batch_embs

    logging.info("%s CLIP NAIP: %d tiles in %.1fs", city, n, time.time() - t0)
    np.save(out_emb, embs)
    np.save(out_ids, np.array([v[0] for v in valid], dtype=np.int32))
    logging.info("Saved: %s  %s", out_emb.name, out_ids.name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract CLIP ViT-L/14 embeddings for WalkCLIP-3C")
    ap.add_argument("--cities", nargs="+", choices=["msp", "seattle", "dc", "pittsburgh"], required=True)
    ap.add_argument("--source", choices=["street", "naip", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logging.warning("CUDA not available — CPU only, will be slow")

    model, processor = load_model(device)

    for city in args.cities:
        lock_ids = load_lock_ids(city)
        logging.info("%s: %d lock IDs", city, len(lock_ids))
        if args.source in ("street", "all"):
            extract_street(city, lock_ids, model, processor, device, args.batch_size, args.overwrite)
        if args.source in ("naip", "all"):
            extract_naip(city, lock_ids, model, processor, device, args.batch_size, args.overwrite)

    logging.info("CLIP extraction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
