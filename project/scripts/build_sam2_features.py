"""
Build clean per-point SAM2 feature vectors from raw mask stat CSVs.

Reads the raw SAM2 outputs (street + naip) for all cities with data, aggregates
street-view headings to per-point statistics, merges with NAIP, drops
always-zero columns, and filters each city to its own per-city lock file.

Outputs:
    project/data/processed/audit/sam2_features_seattle.csv
    project/data/processed/audit/sam2_features_msp.csv
    project/data/processed/audit/sam2_features_dc.csv
    project/data/processed/audit/sam2_features_combined.csv

Run:
    python project/scripts/build_sam2_features.py
    python project/scripts/build_sam2_features.py --no-lock  # all points, no lock filter
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = REPO_ROOT / "project" / "data" / "processed" / "audit"
LOCKS_DIR = REPO_ROOT / "project" / "data" / "processed" / "locks"

# Per-city lock files — each city has its own independent point_id space
CITY_LOCK_FILES: dict[str, Path] = {
    "msp":     LOCKS_DIR / "msp_v2_final_lock_ids.txt",
    "seattle": LOCKS_DIR / "seattle_v2_final_lock_ids.txt",
    "dc":      LOCKS_DIR / "dc_v2_final_lock_ids.txt",
}

# Kept for backward compat (used by --no-lock path and legacy code)
LOCK_FILE = LOCKS_DIR / "v2_final_lock_ids.txt"

# Raw CSV paths
RAW = {
    "seattle_street": AUDIT_DIR / "sam2_mask_stats_seattle_street.csv",
    "msp_street":     AUDIT_DIR / "sam2_mask_stats_msp_street.csv",
    "seattle_naip":   AUDIT_DIR / "sam2_mask_stats_seattle_naip.csv",
    "msp_naip":       AUDIT_DIR / "sam2_mask_stats_msp_naip.csv",
    "dc_street":      AUDIT_DIR / "sam2_mask_stats_dc_street.csv",
    "dc_naip":        AUDIT_DIR / "sam2_mask_stats_dc_naip.csv",
}

ALL_CITIES = ["seattle", "msp", "dc"]

# Street features to aggregate across 4 headings
STREET_FRACS = ["sidewalk_frac", "crosswalk_frac", "vegetation_frac", "sky_frac", "building_frac"]

# NAIP features to keep (always-zero columns dropped: bare_soil, sidewalk, crosswalk, sky, vehicle, person)
NAIP_FRACS = ["vegetation_frac", "building_frac", "road_frac", "water_frac"]


def load_city_lock_ids(city: str) -> set[str]:
    lock_path = CITY_LOCK_FILES.get(city)
    if lock_path is None or not lock_path.exists():
        raise FileNotFoundError(
            f"Per-city lock file not found for '{city}': {lock_path}\n"
            "Run select_{city}_points.py / build_v2_data_lock.py first, or pass --no-lock to skip."
        )
    ids = set(lock_path.read_text().strip().splitlines())
    logging.info("%s lock IDs loaded: %d", city, len(ids))
    return ids


def build_street_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 4-heading street rows → 1 per-point row.

    Aggregations:
    - mean across all 4 headings (including zeros) for all fracs
    - max across 4 headings for sidewalk, crosswalk, vegetation (presence signal)
    - mean of nonzero sidewalk_width_px values (0 if never detected)
    - n_headings: count of headings with any mask coverage > 0
    """
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)

    def agg_point(g: pd.DataFrame) -> pd.Series:
        out: dict = {"n_headings": int((g["mask_coverage"] > 0).sum())}
        for col in STREET_FRACS:
            out[f"{col}_mean"] = round(g[col].mean(), 6)
        for col in ["sidewalk_frac", "crosswalk_frac", "vegetation_frac", "building_frac"]:
            out[f"{col}_max"] = round(g[col].max(), 6)
        sw = g.loc[g["sidewalk_width_px"] > 0, "sidewalk_width_px"]
        out["sidewalk_width_px"] = round(sw.mean(), 2) if len(sw) > 0 else 0.0
        out["street_mask_coverage_mean"] = round(g["mask_coverage"].mean(), 6)
        return pd.Series(out)

    agg = df.groupby("point_id").apply(agg_point).reset_index()
    logging.info("Street features: %d points", len(agg))
    return agg


def build_naip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NAIP rows: one per point. Prefix columns with naip_, flag
    missing (zero mask_coverage) and unclassified (coverage > 0 but
    all category fracs = 0) images.
    """
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)

    frac_sum = df[NAIP_FRACS].sum(axis=1)
    df["naip_missing"] = (df["mask_coverage"] == 0).astype(int)
    df["naip_unclassified"] = ((df["mask_coverage"] > 0) & (frac_sum < 0.001)).astype(int)

    rename = {c: f"naip_{c}" for c in NAIP_FRACS}
    rename["mask_coverage"] = "naip_mask_coverage"
    rename["n_masks"] = "naip_n_masks"

    keep_cols = ["point_id"] + NAIP_FRACS + ["mask_coverage", "n_masks", "naip_missing", "naip_unclassified"]
    out = df[keep_cols].rename(columns=rename)

    n_miss = out["naip_missing"].sum()
    n_unc = out["naip_unclassified"].sum()
    logging.info(
        "NAIP features: %d points  missing=%d (%.1f%%)  unclassified=%d (%.1f%%)",
        len(out), n_miss, n_miss / len(out) * 100, n_unc, n_unc / len(out) * 100,
    )
    return out


def build_city(city: str, use_lock: bool) -> pd.DataFrame:
    logging.info("Processing %s ...", city)

    street_df = pd.read_csv(RAW[f"{city}_street"])
    naip_df   = pd.read_csv(RAW[f"{city}_naip"])

    street_feats = build_street_features(street_df)
    naip_feats   = build_naip_features(naip_df)

    combined = street_feats.merge(naip_feats, on="point_id", how="outer")
    combined["city"] = city

    if use_lock:
        lock_ids = load_city_lock_ids(city)
        before = len(combined)
        combined = combined[combined["point_id"].isin(lock_ids)]
        logging.info(
            "%s: %d → %d points after lock filter (%d dropped)",
            city, before, len(combined), before - len(combined),
        )

    # Sort and put point_id + city first
    combined = combined.sort_values("point_id").reset_index(drop=True)
    cols = ["point_id", "city"] + [c for c in combined.columns if c not in ("point_id", "city")]
    combined = combined[cols]

    # Verify no NaN leakage from outer merge
    missing_naip = combined["naip_missing"].isna().sum()
    if missing_naip > 0:
        logging.warning("%s: %d points have no NAIP row (filling zeros)", city, missing_naip)
        combined = combined.fillna(0)
        combined["naip_missing"] = combined["naip_missing"].clip(lower=1)

    return combined


def print_summary(df: pd.DataFrame, city: str) -> None:
    feat_cols = [c for c in df.columns if c not in ("point_id", "city")]
    always_zero = [c for c in feat_cols if df[c].max() == 0]
    logging.info(
        "%s: %d points, %d feature cols, always-zero cols: %s",
        city, len(df), len(feat_cols), always_zero or "none",
    )
    for col in feat_cols:
        if col.startswith("naip_missing") or col.startswith("naip_unclassified"):
            continue
        nz = (df[col] > 0.001).sum()
        mean = df[col].mean()
        logging.info("  %-35s mean=%.4f  nonzero=%d/%d (%.1f%%)", col, mean, nz, len(df), nz/len(df)*100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build clean SAM2 feature CSVs")
    p.add_argument("--no-lock", action="store_true", help="Skip lock ID filter (include all points)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    present_cities = [c for c in ALL_CITIES if RAW[f"{c}_street"].exists() and RAW[f"{c}_naip"].exists()]
    if not present_cities:
        logging.error("No SAM2 raw CSVs found. Run segment_sam2.py for each city first.")
        return 1

    use_lock = not args.no_lock

    city_dfs = {}
    for city in present_cities:
        df = build_city(city, use_lock)
        print_summary(df, city)
        out_path = AUDIT_DIR / f"sam2_features_{city}.csv"
        df.to_csv(out_path, index=False)
        logging.info("Saved: %s", out_path)
        city_dfs[city] = df

    combined = pd.concat(city_dfs.values(), ignore_index=True)
    out_path = AUDIT_DIR / "sam2_features_combined.csv"
    combined.to_csv(out_path, index=False)
    logging.info(
        "Combined saved: %s  (%d rows × %d cols)",
        out_path, len(combined), len(combined.columns),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
