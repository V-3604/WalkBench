"""
Stage E+ : pre-compute SigLIP processor outputs for the LoRA fine-tune corpus.

Why this script exists
----------------------
WSL on /mnt/c is bottlenecked by the 9P protocol: each JPEG read from the
Windows filesystem costs 5–20 ms. A batch of 32 images × 4 workers spends
~30 s/iter purely in I/O. A 5070 Ti running SigLIP-SO400M with LoRA at bf16
needs only ~120 ms of GPU compute per iter — so we are running at <1% of
hardware capacity. (Confirmed by 68 W power draw at "100 %" WDDM utilization,
versus the 250–280 W expected under sustained compute.)

Solution: run the SigLIP image processor once, save the output tensors as a
float16 memory-mapped array on a native ext4 partition (~/walkclip_cache/),
and have the trainer read from that array instead of decoding JPEGs every
epoch. Resulting per-iter time drops from ~41 s to ~120 ms — a 270× speedup.
The cache is fully deterministic (PIL → SigLIP processor → bilinear resize +
normalize), shared between training and re-extraction, and SHA-256 hashed for
inclusion in the data manifest.

Outputs (default --output-dir = ~/walkclip_cache/):
    siglip_pixels_street_384.npy         # (N, 3, 384, 384) float16, np.lib.format
    siglip_pixels_street_384_index.csv   # (cache_idx, point_id, city, heading,
                                         #  sidewalk_present, crosswalk_present, image_path)
    cache_manifest.json                  # SHA-256, model_id, image_size, n_rows, dtype, build_time, env

Reproducibility note for the paper
----------------------------------
The cache is parameterised by (model_id, image_size). Regeneration is bit-exact
given the same Pillow / transformers versions and source JPEGs. The manifest
records all of these so reviewers can verify or rebuild.

Run (WSL bash, conda base or wherever PyTorch+CUDA is wired up):
    python project/scripts/cache_pixel_values.py \\
        --output-dir ~/walkclip_cache \\
        --batch-size 64 --num-workers 8

Wall clock: ~25–40 min one time (I/O bound on /mnt/c). Disk: ~52 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT  = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/heading_overture_labels.csv"
LOCK_DIR   = REPO_ROOT / "project/data/processed/locks"
IMG_ROOT   = REPO_ROOT / "project/data/raw/imagery/street_view_mapillary"

CITY_DIR = {
    "msp":     IMG_ROOT / "Minneapolis",
    "seattle": IMG_ROOT / "Seattle",
    "dc":      IMG_ROOT / "DC",
}
CITY_LOCK = {
    "msp":     "msp_v2_final_lock_ids.txt",
    "seattle": "seattle_v2_final_lock_ids.txt",
    "dc":      "dc_v2_final_lock_ids.txt",
}

MODEL_ID    = "google/siglip-so400m-patch14-384"
IMAGE_SIZE  = 384
SHA_CHUNK   = 1 << 23   # 8 MB read chunks for hashing


@dataclass
class CachePaths:
    arr_path:      Path
    index_path:    Path
    manifest_path: Path


def cache_paths_for(output_dir: Path, modality: str, image_size: int) -> CachePaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"siglip_pixels_{modality}_{image_size}"
    return CachePaths(
        arr_path      = output_dir / f"{stem}.npy",
        index_path    = output_dir / f"{stem}_index.csv",
        manifest_path = output_dir / "cache_manifest.json",
    )


def build_index(cities: list[str]) -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    df = df[df["city"].isin(cities)].copy()
    df["sidewalk_present"]  = df["sidewalk_present"].astype(int)
    df["crosswalk_present"] = df["crosswalk_present"].astype(int)
    df["point_id"] = df["point_id"].astype(int)
    df["heading"]  = df["heading"].astype(int)

    keep = []
    for city in cities:
        ids = {int(x) for x in (LOCK_DIR / CITY_LOCK[city]).read_text().strip().splitlines()}
        keep.append(df[(df["city"] == city) & (df["point_id"].isin(ids))])
    df = pd.concat(keep, ignore_index=True)

    df["image_path"] = [
        str(CITY_DIR[c] / f"{p}_{h}.jpg")
        for c, p, h in zip(df["city"].to_numpy(), df["point_id"].to_numpy(), df["heading"].to_numpy())
    ]
    exists = np.array([os.path.exists(p) for p in df["image_path"].to_numpy()])
    if not exists.all():
        logging.warning("Dropping %d rows with missing images", (~exists).sum())
    df = df.loc[exists].reset_index(drop=True)
    df.insert(0, "cache_idx", np.arange(len(df), dtype=np.int64))
    return df


class _PrecomputeDataset(Dataset):
    """Loads + processes one image per __getitem__. Designed for DataLoader."""

    def __init__(self, paths: list[str], processor: AutoProcessor) -> None:
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]  # (3, H, W) float32
        return pv.to(torch.float16)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(SHA_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    paths: CachePaths,
    df: pd.DataFrame,
    image_size: int,
    elapsed: float,
    *,
    image_size_check: int,
) -> None:
    arr_sha   = sha256_of_file(paths.arr_path)
    index_sha = sha256_of_file(paths.index_path)
    manifest = {
        "model_id":    MODEL_ID,
        "image_size":  image_size,
        "modality":    "street",
        "n_rows":      int(len(df)),
        "tensor_shape": [int(len(df)), 3, image_size_check, image_size_check],
        "tensor_dtype": "float16",
        "files": {
            paths.arr_path.name:   {"sha256": arr_sha,   "bytes": paths.arr_path.stat().st_size},
            paths.index_path.name: {"sha256": index_sha, "bytes": paths.index_path.stat().st_size},
        },
        "by_city": df["city"].value_counts().to_dict(),
        "by_label": {
            "sidewalk_pos_rate":  float(df["sidewalk_present"].mean()),
            "crosswalk_pos_rate": float(df["crosswalk_present"].mean()),
        },
        "build_time_seconds": round(elapsed, 1),
        "build_started_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed)),
        "build_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python":       sys.version.split()[0],
            "platform":     platform.platform(),
            "torch":        torch.__version__,
            "transformers": __import__("transformers").__version__,
            "pillow":       Image.__version__,
            "numpy":        np.__version__,
        },
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote manifest %s", paths.manifest_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute SigLIP pixel_values cache")
    p.add_argument("--output-dir", default=str(Path.home() / "walkclip_cache"))
    p.add_argument("--cities", nargs="+", default=["msp", "seattle", "dc"])
    p.add_argument("--batch-size", type=int, default=64,
                   help="Pre-compute DataLoader batch size (I/O parallelism, not GPU)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = cache_paths_for(output_dir, "street", args.image_size)

    if paths.arr_path.exists() and not args.overwrite:
        logging.info("Cache already exists at %s. Pass --overwrite to rebuild.", paths.arr_path)
        return 0

    logging.info("Building index from heading_overture_labels.csv …")
    df = build_index(args.cities)
    logging.info(
        "Cache will hold %d (point, heading) rows | by_city=%s | sidewalk_pos=%.3f crosswalk_pos=%.3f",
        len(df), df["city"].value_counts().to_dict(),
        df["sidewalk_present"].mean(), df["crosswalk_present"].mean(),
    )

    df.to_csv(paths.index_path, index=False)
    logging.info("Wrote index %s", paths.index_path)

    logging.info("Loading SigLIP processor (%s) …", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    n = len(df)
    arr = np.lib.format.open_memmap(
        paths.arr_path, mode="w+", dtype=np.float16,
        shape=(n, 3, args.image_size, args.image_size),
    )
    logging.info("Allocated memmap %s (%.1f GB)", paths.arr_path,
                 paths.arr_path.stat().st_size / 1e9)

    ds = _PrecomputeDataset(df["image_path"].tolist(), processor)
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, shuffle=False, drop_last=False,
    )

    t0 = time.time()
    pos = 0
    for batch in tqdm(loader, desc="cache", unit="batch", dynamic_ncols=True):
        b = batch.numpy()  # (B, 3, H, W) float16
        arr[pos: pos + len(b)] = b
        pos += len(b)
    arr.flush()
    elapsed = time.time() - t0

    if pos != n:
        logging.error("Wrote %d rows but expected %d — cache may be incomplete", pos, n)
        return 1

    logging.info(
        "Cache complete: %d rows in %.1fs (%.1f img/sec). Hashing …",
        n, elapsed, n / max(elapsed, 1e-6),
    )
    write_manifest(paths, df, args.image_size, elapsed, image_size_check=args.image_size)
    logging.info("Done. Cache root: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
