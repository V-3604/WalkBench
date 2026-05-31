"""Rebuild composite walkability index from 5 visually-grounded features
(no EPA: drops NatWalkInd, D3B, D4A) and report correlation with the original
8-feature index. Reads the existing per-component values from walkability_index.csv
so no upstream rebuild is required.

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\rebuild_walkability_index_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "walkability_index.csv"
OUT_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "walkability_index_v2_5feat.csv"
OUT_JSON = REPO_ROOT / "project" / "artifacts" / "reports" / "walkability_index_v2_build.json"

CLEAN_COMPONENTS: list[str] = [
    "sidewalk_present",
    "crosswalk_present_near",
    "intersection_count_200m",  # already log1p in stored JSON
    "building_footprint_frac_100m",
    "stops_400m",  # already log1p in stored JSON
]


def main() -> int:
    df = pd.read_csv(INDEX_CSV)
    parsed = df["components_json"].apply(json.loads).tolist()
    feat = pd.DataFrame(parsed)

    X = feat[CLEAN_COMPONENTS].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    pca = PCA(n_components=3, random_state=42).fit(Xs)
    z = pca.transform(Xs)[:, 0]

    # sign-anchor: positive correlation with sidewalk_present
    sw = feat["sidewalk_present"].to_numpy(dtype=np.float64)
    if np.corrcoef(z, sw)[0, 1] < 0:
        z = -z
        loadings_pc1 = -pca.components_[0]
        sign_flipped = True
    else:
        loadings_pc1 = pca.components_[0]
        sign_flipped = False

    z_old = df["walkability_index_v1"].to_numpy(dtype=np.float64)
    pearson = float(np.corrcoef(z, z_old)[0, 1])
    spearman = float(spearmanr(z, z_old).statistic)

    out = df[["point_id", "city"]].copy()
    out["walkability_index_v2"] = z
    out.to_csv(OUT_CSV, index=False)

    report = {
        "components": CLEAN_COMPONENTS,
        "transforms": ["raw", "raw", "log1p_prestored", "raw", "log1p_prestored"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_loadings_pc1": loadings_pc1.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sign_flipped": sign_flipped,
        "anchor": "sidewalk_present",
        "n_total": int(len(out)),
        "correlation_with_v1_pearson": pearson,
        "correlation_with_v1_spearman": spearman,
        "index_quantiles": {
            "p05": float(np.quantile(z, 0.05)),
            "p50": float(np.quantile(z, 0.50)),
            "p95": float(np.quantile(z, 0.95)),
        },
        "out_csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))

    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(f"PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"Loadings (PC1, sign-anchored to sidewalk_present):")
    for name, load in zip(CLEAN_COMPONENTS, loadings_pc1):
        print(f"  {name:<35s} {load:+.4f}")
    print(f"Spearman with 8-feature v1 index: {spearman:.4f}")
    print(f"Pearson  with 8-feature v1 index: {pearson:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
