"""
Stage D Channel B: SAM2 deterministic segmentation for WalkCLIP.

Processes street-view (4-heading Mapillary) and NAIP aerial images using
Grounded-SAM2 (Grounding DINO + SAM2) to produce per-class pixel-fraction
features. Output is sam2_mask_stats_{city}_{source}.csv.

Run examples:
    # Dry run first 100 images, Seattle street-view
    python segment_sam2.py --city seattle --source street --limit 100

    # Full run, MSP NAIP, smaller model for speed
    python segment_sam2.py --city msp --source naip --model-size small

    # Fallback mode (no Grounding DINO, uses SAM2 AMG + CLIP zero-shot)
    python segment_sam2.py --city seattle --source street --no-grounding-dino
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SAM2_ROOT = REPO_ROOT / "sam2"
CHECKPOINTS_DIR = SAM2_ROOT / "checkpoints"
GDINO_DIR = REPO_ROOT / "project" / "models" / "grounding_dino"
RAW_IMAGERY_ROOT = REPO_ROOT / "project" / "data" / "raw" / "imagery"
STREET_VIEW_DIR = RAW_IMAGERY_ROOT / "street_view_mapillary"
NAIP_DIR = RAW_IMAGERY_ROOT / "aerial_naip"
POINTS_DIR = REPO_ROOT / "project" / "data" / "processed" / "points"
OUTPUT_DIR = REPO_ROOT / "project" / "data" / "processed" / "audit"
METRICS_LOG = OUTPUT_DIR / "metrics.jsonl"

# Maps CLI city name to filesystem directory name
CITY_DIR_MAP = {
    "seattle": "Seattle",
    "msp": "Minneapolis",  # no new imagery yet; street fallback to old paths
    "dc": "DC",
}

CHECKPOINT_NAMES = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}

SAM2_CONFIG_NAMES = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# Categories and text prompts sent to Grounding DINO
STREET_CATEGORIES = [
    "sidewalk",
    "crosswalk",
    "vegetation",
    "sky",
    "building",
    "vehicle",
    "person",
]

NAIP_CATEGORIES = [
    "vegetation",
    "building",
    "road",
    "water",
    "bare soil",
]

HEADINGS = [0, 90, 180, 270]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MaskStats:
    point_id: str
    heading: Optional[int]  # None for NAIP
    image_path: str
    sidewalk_frac: float = 0.0
    crosswalk_frac: float = 0.0
    vegetation_frac: float = 0.0
    sky_frac: float = 0.0
    building_frac: float = 0.0
    vehicle_frac: float = 0.0
    person_frac: float = 0.0
    road_frac: float = 0.0
    water_frac: float = 0.0
    bare_soil_frac: float = 0.0
    sidewalk_width_px: float = 0.0
    n_masks: int = 0
    mask_coverage: float = 0.0
    processing_time_s: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# SAM2 + Grounding DINO pipeline
# ---------------------------------------------------------------------------


def load_sam2(model_size: str, device: torch.device):
    """Load SAM2 predictor from checkpoint."""
    sys.path.insert(0, str(SAM2_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_path = CHECKPOINTS_DIR / CHECKPOINT_NAMES[model_size]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt_path}. "
            f"Run: wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/{CHECKPOINT_NAMES[model_size]} "
            f"-P {CHECKPOINTS_DIR}"
        )

    config_name = SAM2_CONFIG_NAMES[model_size]
    model = build_sam2(config_name, str(ckpt_path), device=device)
    predictor = SAM2ImagePredictor(model)
    logging.info("SAM2 %s loaded from %s", model_size, ckpt_path.name)
    return predictor


def load_grounding_dino(device: torch.device):
    """Load Grounding DINO model for text-prompted detection."""
    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise ImportError(
            "groundingdino not installed. Run: pip install groundingdino-py\n"
            "Or rerun with --no-grounding-dino to use SAM2 AMG + CLIP fallback."
        ) from exc

    config_path = GDINO_DIR / "GroundingDINO_SwinT_OGC.py"
    ckpt_path = GDINO_DIR / "groundingdino_swint_ogc.pth"

    for p in [config_path, ckpt_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Grounding DINO file not found: {p}\n"
                f"See docs/v2_SAM2_LOCAL_INFERENCE.md Section 7 for download commands."
            )

    gdino = load_model(str(config_path), str(ckpt_path))
    gdino.to(device)
    gdino.eval()
    logging.info("Grounding DINO loaded from %s", ckpt_path.name)
    return gdino


def run_grounded_sam2(
    image: np.ndarray,
    categories: list[str],
    sam2_predictor,
    gdino_model,
    gdino_threshold: float = 0.3,
    device: torch.device = None,
) -> tuple[dict[str, float], int, float, float, dict[str, np.ndarray]]:
    """
    Run Grounding DINO to get boxes, then SAM2 to get pixel masks.
    Returns (fracs, n_masks, coverage, sidewalk_width_px, category_masks).
    category_masks is the raw bool-array per category, used for visualization.
    """
    from groundingdino.util.inference import preprocess_caption
    from groundingdino.util.utils import get_phrases_from_posmap
    import torchvision.transforms as T

    h, w = image.shape[:2]
    total_pixels = h * w

    # Transform image for Grounding DINO
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gdino_image = transform(Image.fromarray(image))

    # Build text prompt (categories joined by " . ")
    text_prompt = " . ".join(categories) + " ."

    t_gdino = time.perf_counter()
    # NOTE:
    # groundingdino.util.inference.predict() calls model.to(device) on every image.
    # In some torch/groundingdino builds this can mis-handle device typing and raise:
    # "to() received ... got (dtype=torch.device, ...)".
    # We run equivalent inference inline and avoid repeated model.to(...) calls.
    caption = preprocess_caption(text_prompt)
    infer_device = device or next(gdino_model.parameters()).device
    gdino_image = gdino_image.to(infer_device)
    with torch.no_grad():
        outputs = gdino_model(gdino_image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]              # (nq, 4)
    keep = prediction_logits.max(dim=1)[0] > gdino_threshold
    boxes = prediction_boxes[keep]
    logits = prediction_logits[keep].max(dim=1)[0]

    tokenized = gdino_model.tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(logit > gdino_threshold, tokenized, gdino_model.tokenizer).replace(".", "")
        for logit in prediction_logits[keep]
    ]
    logging.debug(
        "  GDINO: %d boxes in %.2fs | phrases=%s",
        len(boxes), time.perf_counter() - t_gdino, phrases,
    )

    # Map predicted phrases to canonical categories
    category_masks: dict[str, np.ndarray] = {cat: np.zeros((h, w), dtype=bool) for cat in categories}

    if len(boxes) == 0:
        logging.debug("  GDINO: no detections — all fracs=0")
        return {cat: 0.0 for cat in categories}, 0, 0.0, 0.0, category_masks

    # Convert normalized boxes to pixel coordinates
    boxes_px = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
    boxes_xyxy = _cxcywh_to_xyxy(boxes_px).numpy()

    # SAM2 prediction
    sam2_predictor.set_image(image)
    input_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).to(device)

    t_sam2 = time.perf_counter()
    with torch.no_grad():
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    logging.debug("  SAM2: %d masks in %.2fs", len(masks), time.perf_counter() - t_sam2)
    # masks: (N, 1, H, W) or (N, H, W). Newer SAM2 returns numpy directly.
    if masks.ndim == 4:
        masks = masks[:, 0]
    masks_np = (masks if isinstance(masks, np.ndarray) else masks.cpu().numpy()).astype(bool)

    # Assign each mask to its closest category phrase
    for mask_np, phrase in zip(masks_np, phrases):
        matched_cat = _match_phrase_to_category(phrase, categories)
        if matched_cat:
            category_masks[matched_cat] |= mask_np

    fracs = {cat: float(category_masks[cat].sum()) / total_pixels for cat in categories}
    n_masks = len(boxes)
    any_mask = np.zeros((h, w), dtype=bool)
    for m in category_masks.values():
        any_mask |= m
    coverage = float(any_mask.sum()) / total_pixels

    sidewalk_width = _estimate_sidewalk_width(category_masks.get("sidewalk"))

    return fracs, n_masks, coverage, sidewalk_width, category_masks


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _match_phrase_to_category(phrase: str, categories: list[str]) -> Optional[str]:
    phrase_lower = phrase.lower().strip()
    for cat in categories:
        if cat in phrase_lower or phrase_lower in cat:
            return cat
    return None


def _estimate_sidewalk_width(sidewalk_mask: Optional[np.ndarray]) -> float:
    """Estimate median horizontal width of the sidewalk mask in pixels."""
    if sidewalk_mask is None or not sidewalk_mask.any():
        return 0.0
    row_widths = sidewalk_mask.sum(axis=1)
    nonzero = row_widths[row_widths > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.median(nonzero))


# Per-category colors for visualization (BGR → PIL RGB order)
_VIZ_COLORS: dict[str, tuple[int, int, int]] = {
    "sidewalk":   (255, 200,   0),   # yellow
    "crosswalk":  (255, 100,   0),   # orange
    "vegetation": ( 34, 180,  34),   # green
    "sky":        (135, 206, 235),   # sky blue
    "building":   (180,  90,  90),   # brick red
    "vehicle":    (220,  20,  60),   # crimson
    "person":     (255,   0, 255),   # magenta
    "road":       (100, 100, 100),   # gray
    "water":      ( 30, 100, 255),   # blue
    "bare soil":  (180, 130,  70),   # tan
}


def save_viz(
    image: np.ndarray,
    category_masks: dict[str, np.ndarray],
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    """Save original image with semi-transparent colored category mask overlay."""
    overlay = image.copy().astype(np.float32)
    for cat, mask in category_masks.items():
        if not mask.any():
            continue
        color = _VIZ_COLORS.get(cat, (255, 255, 255))
        for ch, c in enumerate(color):
            overlay[:, :, ch] = np.where(mask, overlay[:, :, ch] * (1 - alpha) + c * alpha, overlay[:, :, ch])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(out_path, quality=88)


# ---------------------------------------------------------------------------
# SAM2 AMG + CLIP fallback (no Grounding DINO)
# ---------------------------------------------------------------------------


def run_sam2_amg_clip(
    image: np.ndarray,
    categories: list[str],
    sam2_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    device: torch.device,
) -> tuple[dict[str, float], int, float, float]:
    """
    SAM2 automatic mask generation, then CLIP zero-shot classification per mask patch.
    Slower than Grounded-SAM2 but no Grounding DINO dependency.
    """
    import open_clip

    sys.path.insert(0, str(SAM2_ROOT))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Use reduced settings for speed
    # SAM2.1 stores the underlying model as .model (not ._model)
    sam2_model = getattr(sam2_predictor, "model", None) or getattr(sam2_predictor, "_model", None)
    if sam2_model is None:
        raise AttributeError("SAM2ImagePredictor missing both .model and ._model")
    generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=16,        # default 32; halved for speed
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=500,
    )

    auto_masks = generator.generate(image)
    if not auto_masks:
        return {cat: 0.0 for cat in categories}, 0, 0.0, 0.0

    h, w = image.shape[:2]
    total_pixels = h * w
    pil_image = Image.fromarray(image)

    # Classify each mask patch with CLIP
    category_masks: dict[str, np.ndarray] = {cat: np.zeros((h, w), dtype=bool) for cat in categories}
    text_tokens = clip_tokenizer(categories).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    for mask_info in auto_masks:
        seg = mask_info["segmentation"]  # bool (H, W)
        bbox = mask_info["bbox"]  # x, y, w, h
        x, y, bw, bh = [int(v) for v in bbox]

        # Crop mask region from image
        crop = pil_image.crop((x, y, x + bw, y + bh))
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = clip_model.encode_image(crop_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        similarities = (img_feat @ text_feats.T).squeeze(0)
        best_cat_idx = similarities.argmax().item()
        best_cat = categories[best_cat_idx]
        category_masks[best_cat] |= seg

    fracs = {cat: float(category_masks[cat].sum()) / total_pixels for cat in categories}
    n_masks = len(auto_masks)
    any_mask = np.zeros((h, w), dtype=bool)
    for m in category_masks.values():
        any_mask |= m
    coverage = float(any_mask.sum()) / total_pixels
    sidewalk_width = _estimate_sidewalk_width(category_masks.get("sidewalk"))

    return fracs, n_masks, coverage, sidewalk_width


# ---------------------------------------------------------------------------
# Image listing helpers
# ---------------------------------------------------------------------------


def _load_point_ids(points_csv: Path) -> list[str]:
    """
    Load point_ids from a points CSV.
    Handles both named 'point_id' column and unnamed first-column (walkscore.csv style).
    Returns string representations of the ids.
    """
    import pandas as pd

    df = pd.read_csv(points_csv)
    if "point_id" in df.columns:
        return df["point_id"].astype(str).tolist()
    # walkscore.csv has an unnamed first column that is the integer index
    return df.iloc[:, 0].astype(str).tolist()


def list_street_images(city: str, points_csv: Path) -> list[tuple[str, int, Path]]:
    """Return list of (point_id, heading, image_path) for all street-view images."""
    point_ids = _load_point_ids(points_csv)
    city_dir_name = CITY_DIR_MAP.get(city, city.capitalize())
    street_dir = STREET_VIEW_DIR / city_dir_name

    if not street_dir.exists():
        raise FileNotFoundError(
            f"Street-view directory not found: {street_dir}\n"
            f"Expected layout: {STREET_VIEW_DIR}/<City>/<point_id>_<heading>.jpg"
        )

    entries = []
    for point_id in point_ids:
        for heading in HEADINGS:
            img_path = street_dir / f"{point_id}_{heading}.jpg"
            if img_path.exists():
                entries.append((point_id, heading, img_path))

    return entries


def list_naip_images(city: str, points_csv: Path) -> list[tuple[str, None, Path]]:
    """Return list of (point_id, None, image_path) for NAIP tiles."""
    point_ids = _load_point_ids(points_csv)
    city_dir_name = CITY_DIR_MAP.get(city, city.capitalize())
    naip_city_dir = NAIP_DIR / city_dir_name

    if not naip_city_dir.exists():
        raise FileNotFoundError(
            f"NAIP directory not found: {naip_city_dir}\n"
            f"Expected layout: {NAIP_DIR}/<City>/<point_id>.jpg"
        )

    entries = []
    for point_id in point_ids:
        img_path = naip_city_dir / f"{point_id}.jpg"
        if img_path.exists():
            entries.append((point_id, None, img_path))

    return entries


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_images(
    entries: list[tuple[str, Optional[int], Path]],
    categories: list[str],
    sam2_predictor,
    gdino_model,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    use_grounding_dino: bool,
    gdino_threshold: float,
    device: torch.device,
    output_csv: Path,
    resume: bool = True,
    do_save_viz: bool = False,
    viz_dir: Optional[Path] = None,
) -> None:
    """Process a list of (point_id, heading, image_path) entries and write CSV."""

    # Load already-processed point_ids to support resuming
    done_keys: set[str] = set()
    if resume and output_csv.exists():
        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['point_id']}_{row.get('heading', '')}"
                done_keys.add(key)
        logging.info("Resuming: %d images already processed", len(done_keys))

    write_header = not output_csv.exists() or len(done_keys) == 0
    fieldnames = list(MaskStats.__dataclass_fields__.keys())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if not write_header else "w"

    total = len(entries)
    times: list[float] = []

    with open(output_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        from tqdm.contrib.logging import logging_redirect_tqdm
        with logging_redirect_tqdm():
            for idx, (point_id, heading, img_path) in enumerate(tqdm(entries, desc="Segmenting", dynamic_ncols=True)):
                resume_key = f"{point_id}_{heading}"
                if resume_key in done_keys:
                    continue

                heading_str = str(heading) if heading is not None else "naip"
                logging.info(
                    "[%d/%d] point=%s heading=%s | %s",
                    idx + 1, total, point_id, heading_str, img_path.name,
                )

                stats = MaskStats(
                    point_id=point_id,
                    heading=heading,
                    image_path=str(img_path),
                )
                t0 = time.perf_counter()
                fracs: dict[str, float] = {}
                cat_masks: dict[str, np.ndarray] = {}
                try:
                    image = np.array(Image.open(img_path).convert("RGB"))
                    logging.debug("  Image shape: %s", image.shape)

                    if use_grounding_dino:
                        fracs, n_masks, coverage, sw_width, cat_masks = run_grounded_sam2(
                            image, categories, sam2_predictor, gdino_model,
                            gdino_threshold=gdino_threshold, device=device,
                        )
                    else:
                        fracs, n_masks, coverage, sw_width = run_sam2_amg_clip(
                            image, categories, sam2_predictor,
                            clip_model, clip_preprocess, clip_tokenizer, device,
                        )

                    for cat, frac in fracs.items():
                        if hasattr(stats, f"{cat}_frac"):
                            setattr(stats, f"{cat}_frac", round(frac, 6))

                    stats.n_masks = n_masks
                    stats.mask_coverage = round(coverage, 6)
                    stats.sidewalk_width_px = round(sw_width, 2)

                    if do_save_viz and cat_masks:
                        heading_str_viz = str(heading) if heading is not None else "naip"
                        viz_path = viz_dir / f"{point_id}_{heading_str_viz}.jpg"
                        save_viz(image, cat_masks, viz_path)
                        logging.debug("  Saved viz: %s", viz_path.name)

                    nonzero = {k: f"{v:.3f}" for k, v in fracs.items() if v > 0.005}
                    logging.info(
                        "  => masks=%d cov=%.3f sw_px=%.1f time=%.2fs | %s",
                        n_masks, coverage, sw_width,
                        time.perf_counter() - t0,
                        " ".join(f"{k}={v}" for k, v in nonzero.items()) or "(no detections)",
                    )

                except Exception as exc:
                    stats.error = str(exc)
                    logging.warning("  FAIL point=%s heading=%s: %s", point_id, heading_str, exc)

                stats.processing_time_s = round(time.perf_counter() - t0, 3)
                times.append(stats.processing_time_s)

                # ETA every 10 images
                if len(times) % 10 == 0:
                    avg = sum(times) / len(times)
                    remaining = total - (idx + 1)
                    logging.info(
                        "  [ETA] avg=%.2fs/img  remaining=%d  eta=%.1fmin",
                        avg, remaining, avg * remaining / 60,
                    )

                writer.writerow(asdict(stats))
                f.flush()


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


def run_single_image_debug(image_path: str) -> None:
    """Interactive debug: print Grounding DINO detections for one image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdino = load_grounding_dino(device)
    image = np.array(Image.open(image_path).convert("RGB"))
    fracs, n_masks, coverage, sw_width, _ = run_grounded_sam2(
        image, STREET_CATEGORIES, sam2_predictor=None, gdino_model=gdino,
        gdino_threshold=0.25, device=device,
    )
    print(f"Image: {image_path}")
    print(f"Masks found: {n_masks}, coverage: {coverage:.3f}, sidewalk_width: {sw_width:.1f}px")
    for cat, frac in fracs.items():
        bar = "#" * int(frac * 40)
        print(f"  {cat:<15} {frac:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 deterministic segmentation — Stage D Channel B"
    )
    parser.add_argument(
        "--city",
        choices=["seattle", "msp", "dc"],
        required=True,
        help="City to process: seattle, msp, or dc",
    )
    parser.add_argument(
        "--source",
        choices=["street", "naip"],
        required=True,
        help="Image source: street-view or NAIP aerial",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base_plus", "large"],
        default="base_plus",
        help="SAM2 checkpoint size (default: base_plus)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process per GPU batch (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N images (dry run / testing)",
    )
    parser.add_argument(
        "--no-grounding-dino",
        action="store_true",
        help="Fall back to SAM2 AMG + CLIP zero-shot (no Grounding DINO)",
    )
    parser.add_argument(
        "--gdino-threshold",
        type=float,
        default=0.3,
        help="Grounding DINO confidence threshold (default: 0.3, lower = more detections)",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B-32",
        help="CLIP model name for fallback mode (default: ViT-B-32)",
    )
    parser.add_argument(
        "--imagery-root",
        type=Path,
        default=None,
        help="Override imagery root directory (useful when data is on WSL2 native fs)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess from scratch, overwriting existing output",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save colored mask overlay JPGs alongside the CSV for visual inspection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (per-step GDINO/SAM2 timing inside each image)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("Seed: %d", args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info(
            "GPU: %s | VRAM: %.1f GB | Compute: %s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            torch.cuda.get_device_capability(0),
        )
    else:
        logging.warning("CUDA not available — inference will be very slow on CPU.")

    if args.imagery_root:
        global STREET_VIEW_DIR, NAIP_DIR
        custom_root = Path(args.imagery_root)
        STREET_VIEW_DIR = custom_root / "street_view_mapillary"
        NAIP_DIR = custom_root / "aerial_naip"

    # Resolve points CSV — prefer final-lock CSV if it exists
    city_csv_map = {
        "seattle": POINTS_DIR / "seattle_points_v2.csv",
        "msp":     POINTS_DIR / "msp_points_v2.csv",
        "dc":      POINTS_DIR / "dc_points_v2.csv",
    }
    points_csv = city_csv_map[args.city]
    lock_csv = points_csv.with_name(points_csv.stem + "_final_lock.csv")
    if lock_csv.exists():
        points_csv = lock_csv
    if not points_csv.exists():
        logging.error("Points CSV not found: %s", points_csv)
        return 1

    # Load image list
    if args.source == "street":
        entries = list_street_images(args.city, points_csv)
        categories = STREET_CATEGORIES
    else:
        entries = list_naip_images(args.city, points_csv)
        categories = NAIP_CATEGORIES

    if not entries:
        logging.error(
            "No images found for city=%s source=%s. "
            "Street-view dir: %s  NAIP dir: %s",
            args.city, args.source, STREET_VIEW_DIR, NAIP_DIR,
        )
        return 1

    logging.info("Images found: %d", len(entries))

    if args.limit:
        entries = entries[: args.limit]
        logging.info("--limit %d: processing only first %d images", args.limit, len(entries))

    output_csv = OUTPUT_DIR / f"sam2_mask_stats_{args.city}_{args.source}.csv"
    logging.info("Output: %s", output_csv)

    # Load models
    logging.info("Loading SAM2 %s checkpoint...", args.model_size)
    sam2_predictor = load_sam2(args.model_size, device)

    gdino_model = None
    clip_model = clip_preprocess = clip_tokenizer = None
    use_grounding_dino = not args.no_grounding_dino

    if use_grounding_dino:
        logging.info("Loading Grounding DINO...")
        try:
            gdino_model = load_grounding_dino(device)
        except (ImportError, FileNotFoundError) as exc:
            logging.warning("Grounding DINO unavailable (%s). Falling back to SAM2 AMG + CLIP.", exc)
            use_grounding_dino = False

    if not use_grounding_dino:
        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained="openai"
        )
        clip_tokenizer = open_clip.get_tokenizer(args.clip_model)
        clip_model.to(device).eval()
        logging.info("CLIP model %s loaded for fallback classification", args.clip_model)

    # Log run metadata
    run_meta = {
        "city": args.city,
        "source": args.source,
        "model_size": args.model_size,
        "use_grounding_dino": use_grounding_dino,
        "gdino_threshold": args.gdino_threshold,
        "n_images": len(entries),
        "limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "output_csv": str(output_csv),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps({"event": "run_start", **run_meta, "ts": time.time()}) + "\n")

    logging.info("Run config: %s", run_meta)

    # Visualization output directory (only used when --save-viz)
    viz_dir = OUTPUT_DIR / f"viz_{args.city}_{args.source}"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Viz overlays will be written to: %s", viz_dir)

    # Run segmentation
    t_start = time.time()
    process_images(
        entries=entries,
        categories=categories,
        sam2_predictor=sam2_predictor,
        gdino_model=gdino_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        use_grounding_dino=use_grounding_dino,
        gdino_threshold=args.gdino_threshold,
        device=device,
        output_csv=output_csv,
        resume=not args.no_resume,
        do_save_viz=args.save_viz,
        viz_dir=viz_dir,
    )
    elapsed = time.time() - t_start
    per_image = elapsed / len(entries) if entries else 0

    logging.info(
        "Done. %.0f images in %.1f min (%.2f s/image). Output: %s",
        len(entries),
        elapsed / 60,
        per_image,
        output_csv,
    )

    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps({
            "event": "run_end",
            "city": args.city,
            "source": args.source,
            "n_images": len(entries),
            "elapsed_s": round(elapsed, 1),
            "s_per_image": round(per_image, 3),
            "output_csv": str(output_csv),
            "ts": time.time(),
        }) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
