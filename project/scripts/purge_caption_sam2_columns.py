"""Purge SAM2 mask-stat and caption-derived columns from features_labels_agreement.csv.

Decision 2026-04-26: SAM2 and captions dropped from all ablations (O9+O21, O22).
Emits features_labels_agreement_v2_clean.csv in the same directory and prints the
SHA-256[:12] line to paste into v2_FINAL_RESULTS_LOCK.md.

Does NOT overwrite the original.

Run:
    & ".venv\\Scripts\\python.exe" project/scripts/purge_caption_sam2_columns.py
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[2]
INPUT_CSV  = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
OUTPUT_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement_v2_clean.csv"

# Exactly SAM2_INPUT_COLS from walkclip_stage_e_v2.py
SAM2_COLS = [
    "n_headings",
    "sidewalk_frac_mean", "crosswalk_frac_mean", "vegetation_frac_mean",
    "sky_frac_mean", "building_frac_mean",
    "sidewalk_frac_max", "crosswalk_frac_max", "vegetation_frac_max",
    "building_frac_max", "sidewalk_width_px", "street_mask_coverage_mean",
    "naip_vegetation_frac", "naip_building_frac", "naip_road_frac",
    "naip_water_frac", "naip_mask_coverage", "naip_n_masks",
    "naip_missing", "naip_unclassified",
]

# Caption-derived column name prefixes (none expected in this CSV — captions are separate)
CAPTION_PREFIXES = ("caption_", "cap_", "gpt_")


def sha256_first12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = pd.read_csv(INPUT_CSV)
    logging.info("Input: %s  shape=%s", INPUT_CSV.name, df.shape)

    present_sam2 = [c for c in SAM2_COLS if c in df.columns]
    if len(present_sam2) < len(SAM2_COLS):
        missing = [c for c in SAM2_COLS if c not in df.columns]
        logging.warning("Expected SAM2 columns not found (already purged?): %s", missing)

    caption_found = [c for c in df.columns if any(c.startswith(p) for p in CAPTION_PREFIXES)]
    if caption_found:
        logging.warning("Caption-derived columns found — will drop: %s", caption_found)
    else:
        logging.info(
            "No caption-derived columns in CSV "
            "(captions live in caption_features_final_lock.csv, not merged here)."
        )

    sam2_derived_kept = [
        c for c in df.columns
        if c.startswith("ca_") or c in (
            "agree_sidewalk", "agree_crosswalk", "agree_tree_canopy",
            "agree_transit_stop", "agreement_score", "agreement_tier",
        )
    ]
    if sam2_derived_kept:
        logging.info(
            "Audit-agreement columns derived from SAM2 predictions are RETAINED "
            "(not part of SAM2_INPUT_COLS; not used as model inputs): %s",
            sam2_derived_kept,
        )

    cols_to_drop = present_sam2 + caption_found
    df_clean = df.drop(columns=cols_to_drop)
    logging.info(
        "Dropped %d columns: %s\nRetained %d columns: %s",
        len(cols_to_drop), cols_to_drop,
        len(df_clean.columns), list(df_clean.columns),
    )

    if OUTPUT_CSV.exists():
        logging.warning("Overwriting existing %s", OUTPUT_CSV.name)
    df_clean.to_csv(OUTPUT_CSV, index=False)

    sha = sha256_first12(OUTPUT_CSV)
    logging.info("Wrote %s  shape=%s  SHA-256[:12]=%s", OUTPUT_CSV.name, df_clean.shape, sha)
    logging.info(
        "\nPaste into v2_FINAL_RESULTS_LOCK.md data-files table:\n"
        "  | `%s` | `%s` | %d | "
        "Clean features CSV — SAM2 mask-stats and captions purged (P0.1) |",
        OUTPUT_CSV.relative_to(REPO_ROOT).as_posix(),
        sha,
        len(df_clean),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
