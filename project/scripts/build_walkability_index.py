"""
Compute a continuous composite walkability index per locked point.

The index is the first principal component of 8 standardized features that
span visually-grounded (sidewalk, crosswalk, intersection density, building
density), spatial (transit access), and administrative (NatWalkInd, D3B,
D4A) walkability. PCA is fit on the union of MSP + Seattle + DC training
points; the same projection is applied to all 14,692 locked points.

The first PC is sign-flipped (if necessary) so that high values correspond
to "more walkable" — we anchor the sign by requiring positive correlation
with sidewalk_present, which is the most directly visual feature.

This is the headline regression target for the paper. It is the closest
analogue to old WalkCLIP's continuous Walk Score regression.

Inputs:
    project/data/processed/labels/features_labels_agreement.csv
    project/data/processed/labels/overture_targets.csv
    project/data/processed/labels/heading_overture_labels_v2.csv  (after A.3)

Outputs:
    project/data/processed/labels/walkability_index.csv
        Columns: point_id, city, walkability_index_v1, components_json
    project/artifacts/reports/walkability_index_build.json
        PCA loadings, explained variance, sign convention

Run:
    & $PY project/scripts/build_walkability_index.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "project/data/processed/labels"
FEATURES_CSV = LABELS_DIR / "features_labels_agreement.csv"
OVERTURE_CSV = LABELS_DIR / "overture_targets.csv"
HEADING_V2_CSV = LABELS_DIR / "heading_overture_labels_v2.csv"
OUT_CSV = LABELS_DIR / "walkability_index.csv"
REPORT = REPO_ROOT / "project/artifacts/reports/walkability_index_build.json"

# Component features. Each row is (column_name, source_csv_token, transform).
# source_csv_token: "F" = features_labels_agreement.csv,
#                   "O" = overture_targets.csv,
#                   "H" = heading_overture_labels_v2.csv (aggregated to point).
COMPONENTS: list[tuple[str, str, str]] = [
    ("sidewalk_present",                       "O", "raw"),
    ("crosswalk_present_near",                 "H", "raw"),     # see A.3
    ("intersection_count_200m",                "O", "log1p"),
    ("building_footprint_frac_100m",           "O", "raw"),
    ("NatWalkInd",                             "F", "raw"),
    ("D3B",                                    "F", "raw"),
    ("D4A",                                    "F", "raw"),
    ("stops_400m",                             "F", "log1p"),
]
SIGN_ANCHOR = "sidewalk_present"  # final PC sign must correlate positively with this


def load_components() -> pd.DataFrame:
    feats = pd.read_csv(FEATURES_CSV)
    over = pd.read_csv(OVERTURE_CSV)
    if HEADING_V2_CSV.exists():
        heading = pd.read_csv(HEADING_V2_CSV)
        # Aggregate per-(point_id, city) by max across the 4 headings — a point
        # is "near a crosswalk" if any of its 4 headings hits one.
        cw_near = (heading.groupby(["point_id", "city"])
                          ["crosswalk_present"].max()
                          .rename("crosswalk_present_near").reset_index())
    else:
        logging.warning("heading_overture_labels_v2.csv not found; falling back to "
                        "overture_targets.crosswalk_present (the unfixed proxy).")
        cw_near = over[["point_id", "city", "crosswalk_present"]].copy()
        cw_near = cw_near.rename(columns={"crosswalk_present": "crosswalk_present_near"})

    df = feats.merge(over, on=["point_id", "city"], how="inner",
                     suffixes=("", "_over"))
    df = df.merge(cw_near, on=["point_id", "city"], how="left")

    # Resolve column names: overture_targets uses unprefixed names; some merges
    # may have introduced overture_*-prefixed copies.
    rename_map = {}
    for c in ("sidewalk_present", "crosswalk_present",
              "intersection_count_200m", "building_footprint_frac_100m"):
        if c not in df.columns and f"overture_{c}" in df.columns:
            rename_map[f"overture_{c}"] = c
    df = df.rename(columns=rename_map)
    return df


def assemble_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    cols, mat = [], []
    for name, _src, transform in COMPONENTS:
        if name not in df.columns:
            raise SystemExit(
                f"Missing column '{name}' in joined frame. Columns found: "
                f"{sorted(df.columns)[:30]}... Verify upstream CSVs.")
        x = df[name].to_numpy(dtype=np.float64)
        if transform == "log1p":
            x = np.log1p(np.clip(x, a_min=0.0, a_max=None))
        elif transform != "raw":
            raise ValueError(f"unknown transform: {transform}")
        cols.append(name)
        mat.append(x)
    X = np.stack(mat, axis=1)

    # Median-impute missing per column.
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            col[mask] = med
            X[:, j] = col
            logging.warning("Imputed %d NaNs in %s with median %.4f",
                            int(mask.sum()), cols[j], med)
    return X, cols, df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-cities", nargs="+", default=["msp", "seattle", "dc"],
                   help="Cities whose points are used to fit StandardScaler+PCA. "
                        "Default = pool all 3 (the index is meant to be a shared scale).")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    df = load_components()
    X, cols, df = assemble_matrix(df)
    fit_mask = df["city"].isin(args.fit_cities).to_numpy()

    scaler = StandardScaler().fit(X[fit_mask])
    Xs = scaler.transform(X)
    pca = PCA(n_components=3).fit(Xs[fit_mask])
    pc = pca.transform(Xs)[:, 0]
    # Sign-anchor PC1 to sidewalk_present.
    anchor_idx = cols.index(SIGN_ANCHOR)
    if np.corrcoef(pc, X[:, anchor_idx])[0, 1] < 0:
        pc = -pc
        flipped = True
    else:
        flipped = False

    out = df[["point_id", "city"]].copy()
    out["walkability_index_v1"] = pc.astype(np.float32)
    out["components_json"] = [json.dumps({c: float(X[i, j]) for j, c in enumerate(cols)})
                              for i in range(len(out))]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    report = {
        "components": cols,
        "transforms": [t for _, _, t in COMPONENTS],
        "fit_cities": args.fit_cities,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_loadings_pc1": pca.components_[0].tolist(),
        "pca_loadings_pc2": pca.components_[1].tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sign_flipped": flipped,
        "anchor": SIGN_ANCHOR,
        "n_total": int(len(out)),
        "n_fit": int(fit_mask.sum()),
        "out_csv": str(OUT_CSV),
        "index_quantiles": {
            "p05": float(np.quantile(pc, 0.05)),
            "p50": float(np.quantile(pc, 0.50)),
            "p95": float(np.quantile(pc, 0.95)),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("Wrote %s (%d rows)", OUT_CSV, len(out))
    logging.info("PC1 explains %.1f%% of variance",
                 100.0 * pca.explained_variance_ratio_[0])
    logging.info("Loadings (PC1):")
    for c, w in zip(cols, pca.components_[0]):
        logging.info("    %-40s %+.3f", c, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
