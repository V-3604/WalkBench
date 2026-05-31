"""
Build meter-based spatial context features for WalkCLIP v2.

This is a practical SAFE replacement that stays fully reproducible offline:
    - neighborhood counts in meters (not degrees)
    - kNN aggregates over nearby sampled points within each city
    - no target leakage: only caption/SAM-style observable features are aggregated

Inputs:
    project/data/processed/labels/features_labels_agreement.csv
    project/data/processed/text/caption_features_final_lock.csv

Outputs:
    project/data/processed/labels/spatial_context_features.csv
    project/data/processed/labels/spatial_context_summary.json

Run:
    python project/scripts/build_spatial_context_features.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
CAPTION_CSV = REPO_ROOT / "project" / "data" / "processed" / "text" / "caption_features_final_lock.csv"
OUT_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "spatial_context_features.csv"
OUT_JSON = REPO_ROOT / "project" / "data" / "processed" / "labels" / "spatial_context_summary.json"

EARTH_RADIUS_M = 6_371_008.8
K_NEIGHBORS = 8
COUNT_RADII_M = [200.0, 400.0, 800.0]
MEAN_RADIUS_M = 400.0

BASE_COLS = [
    "sidewalk_frac_mean",
    "crosswalk_frac_mean",
    "vegetation_frac_mean",
    "sky_frac_mean",
    "building_frac_mean",
    "sidewalk_width_px",
    "street_mask_coverage_mean",
    "agreement_score",
    "cap_sidewalk_ord",
    "cap_crosswalk_ord",
    "cap_traffic_volume_ord",
    "cap_tree_canopy_ord",
    "cap_transit_stop_ord",
    "cap_lighting_ord",
    "cap_maintenance_ord",
    "cap_unknown_count",
]


def _query_radius_excluding_self(tree: BallTree, coords_rad: np.ndarray, radius_m: float) -> list[np.ndarray]:
    idxs = tree.query_radius(coords_rad, r=radius_m / EARTH_RADIUS_M, return_distance=False)
    out: list[np.ndarray] = []
    for i, ids in enumerate(idxs):
        out.append(ids[ids != i])
    return out


def build_city_features(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.sort_values("point_id").reset_index(drop=True).copy()
    coords_rad = np.radians(city_df[["lat", "lon"]].to_numpy(dtype=np.float64))
    tree = BallTree(coords_rad, metric="haversine")

    dists_rad, idxs = tree.query(coords_rad, k=min(K_NEIGHBORS + 1, len(city_df)))
    dists_m = dists_rad[:, 1:] * EARTH_RADIUS_M
    idxs = idxs[:, 1:]

    out = city_df[["point_id", "city"]].copy()
    out["spatial_knn_mean_dist_m"] = dists_m.mean(axis=1).astype("float32")
    out["spatial_knn_max_dist_m"] = dists_m.max(axis=1).astype("float32")

    for radius_m in COUNT_RADII_M:
        neighbors = _query_radius_excluding_self(tree, coords_rad, radius_m)
        out[f"spatial_neighbor_count_{int(radius_m)}m"] = np.array([len(x) for x in neighbors], dtype=np.float32)

    base = city_df[BASE_COLS].copy()
    for col in BASE_COLS:
        median = float(base[col].median()) if base[col].notna().any() else 0.0
        base[col] = base[col].fillna(median)
    base_arr = base.to_numpy(dtype=np.float32)

    weights = 1.0 / np.clip(dists_m, 1.0, None)
    weights = weights / weights.sum(axis=1, keepdims=True)

    knn_mean = np.zeros((len(city_df), len(BASE_COLS)), dtype=np.float32)
    for i in range(len(city_df)):
        knn_mean[i] = (base_arr[idxs[i]] * weights[i][:, None]).sum(axis=0)

    radius_ids = _query_radius_excluding_self(tree, coords_rad, MEAN_RADIUS_M)
    radius_mean = np.zeros((len(city_df), len(BASE_COLS)), dtype=np.float32)
    for i, nbrs in enumerate(radius_ids):
        if len(nbrs) == 0:
            radius_mean[i] = base_arr[i]
        else:
            radius_mean[i] = base_arr[nbrs].mean(axis=0)

    for j, col in enumerate(BASE_COLS):
        out[f"spatial_knn_wmean__{col}"] = knn_mean[:, j]
        out[f"spatial_r{int(MEAN_RADIUS_M)}m_mean__{col}"] = radius_mean[:, j]

    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    for path in (LABELS_CSV, CAPTION_CSV):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            return 1

    labels = pd.read_csv(LABELS_CSV)
    captions = pd.read_csv(CAPTION_CSV)
    labels["point_id"] = labels["point_id"].astype(int)
    captions["point_id"] = captions["point_id"].astype(int)

    merged = labels.merge(captions, on=["point_id", "city"], how="left")
    missing = merged["cap_unknown_count"].isna().sum()
    if missing:
        logging.error(
            "Caption features missing for %d rows. Run build_caption_features.py first and verify final-lock coverage.",
            missing,
        )
        return 1

    pieces = []
    summary: dict[str, object] = {"n_rows": int(len(merged)), "per_city": {}}
    present_cities = [c for c in ("msp", "seattle", "dc") if c in merged["city"].values]
    for city in present_cities:
        city_df = merged[merged["city"] == city].copy()
        feats = build_city_features(city_df)
        pieces.append(feats)
        summary["per_city"][city] = {
            "n_points": int(len(city_df)),
            "mean_knn_dist_m": round(float(feats["spatial_knn_mean_dist_m"].mean()), 2),
            "mean_neighbors_400m": round(float(feats["spatial_neighbor_count_400m"].mean()), 2),
        }

    out = pd.concat(pieces, ignore_index=True).sort_values(["city", "point_id"])
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    logging.info("Saved: %s  (%d rows × %d cols)", OUT_CSV, len(out), len(out.columns))
    logging.info("Saved: %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
