"""
Extract RemoteCLIP ViT-L/14 embeddings for WalkCLIP SRC Stage 2 (Extension #6).

Model: RemoteCLIP-ViT-L-14 (Liu et al. 2024, arXiv:2306.11029)
  - CLIP ViT-L/14 fine-tuned on remote sensing imagery (RSICD, RSITMD, UCM, etc.)
  - Output: 768-dim L2-normalised CLS embedding
  - NAIP only — aerial modality benefits from aerial-domain pretraining;
    street-view is out of distribution for RemoteCLIP.

Checkpoint loaded via HuggingFace Hub:
  repo:  chendelong/RemoteCLIP
  file:  RemoteCLIP-ViT-L-14.pt
  (requires open_clip_torch; pip install open_clip_torch huggingface_hub)

Outputs (project/artifacts/embeddings/):
    remoteclip_naip_{city}.npy     # (N_locked, 768)
    remoteclip_naip_{city}_ids.npy # (N_locked,) int32

Run (WSL, CUDA):
    .venv_wsl/bin/python project/scripts/extract_remoteclip_embeddings.py --cities msp seattle dc pittsburgh
    .venv_wsl/bin/python project/scripts/extract_remoteclip_embeddings.py --cities pittsburgh --overwrite
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

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR  = REPO_ROOT / "project/data/processed/locks"
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
IMG_ROOT  = REPO_ROOT / "project/data/raw/imagery"

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

HF_REPO = "chendelong/RemoteCLIP"
HF_FILENAME = "RemoteCLIP-ViT-L-14.pt"
OPEN_CLIP_ARCH = "ViT-L-14"
BACKBONE_TAG = "remoteclip"
EMBED_DIM = 768


def load_lock_ids(city: str) -> list[int]:
    lock_path = LOCK_DIR / CITY_LOCK_FILES[city]
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Lock file not found: {lock_path}. Run build_v2_final_lock.py --include-cities {city} first."
        )
    return [int(x) for x in lock_path.read_text().strip().splitlines()]


def load_model(device: torch.device) -> tuple:
    """Return (visual_model, preprocess_fn) for NAIP embedding."""
    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open_clip_torch not installed. Run: pip install open_clip_torch huggingface_hub"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    logging.info("Downloading RemoteCLIP checkpoint from %s …", HF_REPO)
    t0 = time.time()
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    logging.info("Checkpoint at %s (%.1fs)", ckpt_path, time.time() - t0)

    model, _, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_ARCH)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    # Use only the visual encoder — we embed images, not text.
    # Cast to float16 to match the half-precision input tensors in embed_batch_remoteclip.
    visual = model.visual.to(device=device, dtype=torch.float16).eval()

    logging.info("RemoteCLIP ViT-L/14 loaded  (device=%s)", device)
    return visual, preprocess


@torch.inference_mode()
def embed_batch_remoteclip(
    images: list[Image.Image],
    visual_model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (N, 768)."""
    tensors = torch.stack([preprocess(img) for img in images]).to(device, dtype=torch.float16)
    embs = visual_model(tensors).float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-8, None)


def extract_naip(
    city: str,
    lock_ids: list[int],
    visual_model: torch.nn.Module,
    preprocess,
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

    if not valid:
        logging.error("%s: no NAIP tiles found in %s", city, img_dir)
        return

    n = len(valid)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)

    t0 = time.time()
    for start in tqdm(range(0, n, batch_size),
                      desc=f"{BACKBONE_TAG}_naip/{city}", unit="batch", dynamic_ncols=True):
        batch = valid[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for _, p in batch]
        embs[start: start + len(batch)] = embed_batch_remoteclip(
            images, visual_model, preprocess, device
        )

    elapsed = time.time() - t0
    logging.info("%s NAIP: %d tiles in %.1fs (%.1f ms/img)",
                 city, n, elapsed, elapsed / max(n, 1) * 1000)
    np.save(out_emb, embs)
    np.save(out_ids, np.array([v[0] for v in valid], dtype=np.int32))
    logging.info("Saved: %s  %s", out_emb.name, out_ids.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract RemoteCLIP ViT-L/14 NAIP embeddings for WalkCLIP SRC (NAIP only)"
    )
    p.add_argument("--cities", nargs="+",
                   choices=["msp", "seattle", "dc", "pittsburgh"], required=True)
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
        logging.warning("CUDA not available — running on CPU, will be slow")

    visual_model, preprocess = load_model(device)

    for city in args.cities:
        lock_ids = load_lock_ids(city)
        logging.info("%s: %d lock IDs", city, len(lock_ids))
        extract_naip(city, lock_ids, visual_model, preprocess, device, args.batch_size, args.overwrite)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
