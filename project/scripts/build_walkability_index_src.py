"""
Rebuild composite walkability_index_v2 for WalkCLIP SRC (all 4 cities).

Motivation: the crosswalk component changed after CROSSWALK_CLASSES was
narrowed to {"marked"} only (re-filtered 2026-05-20). This script recomputes
the 5-feature visual PCA index from scratch, fitting StandardScaler + PCA on
the 3-city reference set (MSP + Seattle + DC) and applying frozen loadings to
Pittsburgh — so Pittsburgh is never seen during construct fitting.

Five visual components:
    sidewalk_present          ← overture_sidewalk_present in FLA (50m buffer)
    crosswalk_present_near    ← max(crosswalk_present) across 4 headings in
                                 heading_overture_labels_v2.csv (15m ±30° wedge)
    intersection_count_200m   ← log1p(overture_intersection_count_200m)
    building_footprint_frac   ← overture_building_footprint_frac_100m
    stops_400m                ← log1p(stops_400m)

Outputs:
    project/data/processed/labels/walkability_index.csv
        Columns: point_id, city, walkability_index_v1 (preserved), walkability_index_v2
        Rows: all 4 cities (19,624 rows after Pittsburgh append).
        walkability_index_v1 is NaN for Pittsburgh (no EPA composite available
        at construct-build time; train on v2 for Pittsburgh evaluation).

    project/artifacts/reports/walkability_index_src_build.json
        Frozen loadings, scaler params, explained-variance — used to reproduce
        Pittsburgh inference without refitting.

Run (Windows, CPU):
    & ".venv\\Scripts\\python.exe" project/scripts/build_walkability_index_src.py
    & ".venv\\Scripts\\python.exe" project/scripts/build_walkability_index_src.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
FLA_PATH = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
HEADING_LABELS_PATH = (
    REPO_ROOT / "project/data/processed/labels/heading_overture_labels_v2.csv"
)
WALKABILITY_CSV = REPO_ROOT / "project/data/processed/labels/walkability_index.csv"
BUILD_JSON = REPO_ROOT / "project/artifacts/reports/walkability_index_src_build.json"

REFERENCE_CITIES = ("msp", "seattle", "dc")   # PCA fit on these only
ALL_CITIES = ("msp", "seattle", "dc", "pittsburgh")

FEAT_COLS = [
    "sidewalk_present",
    "crosswalk_present_near",
    "intersection_count_200m_log1p",
    "building_footprint_frac_100m",
    "stops_400m_log1p",
]


def build_components(fla: pd.DataFrame, heading_labels: pd.DataFrame) -> pd.DataFrame:
    """Merge FLA columns with per-point heading-based crosswalk_present_near."""
    base = fla[["point_id", "city",
                "overture_sidewalk_present",
                "overture_intersection_count_200m",
                "overture_building_footprint_frac_100m",
                "stops_400m"]].copy()

    # crosswalk_present_near: 1 if any of the 4 heading wedges sees a marked crossing
    cw_any = (
        heading_labels.groupby(["point_id", "city"])["crosswalk_present"]
        .max()
        .reset_index()
        .rename(columns={"crosswalk_present": "crosswalk_present_near"})
    )
    base = base.merge(cw_any, on=["point_id", "city"], how="left")
    base["crosswalk_present_near"] = base["crosswalk_present_near"].fillna(0).astype(float)

    # log1p transforms for heavy-tailed counts
    base["intersection_count_200m_log1p"] = np.log1p(
        base["overture_intersection_count_200m"].clip(lower=0)
    )
    base["stops_400m_log1p"] = np.log1p(base["stops_400m"].clip(lower=0))

    base = base.rename(columns={
        "overture_sidewalk_present": "sidewalk_present",
        "overture_building_footprint_frac_100m": "building_footprint_frac_100m",
    })
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild walkability_index_v2 for all 4 cities")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print stats without writing files.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    for p in (FLA_PATH, HEADING_LABELS_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Ensure the label pipeline (derive_overture_targets, "
                "derive_heading_overture_labels, stage_c_build_labels) has been run."
            )

    logging.info("Loading FLA (%s) …", FLA_PATH)
    fla = pd.read_csv(FLA_PATH)
    fla["point_id"] = fla["point_id"].astype(int)
    logging.info("FLA: %d rows  cities=%s", len(fla), sorted(fla["city"].unique()))

    logging.info("Loading heading labels (%s) …", HEADING_LABELS_PATH)
    heading = pd.read_csv(HEADING_LABELS_PATH)
    heading["point_id"] = heading["point_id"].astype(int)
    logging.info("Heading labels: %d rows", len(heading))

    components = build_components(fla, heading)

    # Sanity check
    missing_feat = [c for c in FEAT_COLS if components[c].isna().all()]
    if missing_feat:
        raise ValueError(f"All-NaN feature columns: {missing_feat}. Check FLA and heading labels.")

    # ---- Fit on 3-city reference only ----------------------------------------
    ref_mask = components["city"].isin(REFERENCE_CITIES)
    X_ref = components.loc[ref_mask, FEAT_COLS].to_numpy(dtype=np.float64)
    logging.info("Reference fit: %d rows (%s)", X_ref.shape[0], ", ".join(REFERENCE_CITIES))

    scaler = StandardScaler().fit(X_ref)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(X_ref))

    # ---- Apply to all 4 cities -----------------------------------------------
    X_all = components[FEAT_COLS].to_numpy(dtype=np.float64)
    # Impute any NaN in feature columns with the reference column means
    col_means = np.nanmean(X_all[ref_mask], axis=0)
    for j in range(X_all.shape[1]):
        nan_mask = np.isnan(X_all[:, j])
        if nan_mask.any():
            logging.warning("Feature '%s': %d NaN values — imputed with reference mean %.4f",
                            FEAT_COLS[j], nan_mask.sum(), col_means[j])
            X_all[nan_mask, j] = col_means[j]

    Xs = scaler.transform(X_all)
    z = pca.transform(Xs)[:, 0]

    # Sign-anchor: positive correlation with sidewalk_present
    sw = components["sidewalk_present"].to_numpy(dtype=np.float64)
    if np.corrcoef(z, sw)[0, 1] < 0:
        z = -z
        loadings_pc1 = -pca.components_[0]
        sign_flipped = True
    else:
        loadings_pc1 = pca.components_[0]
        sign_flipped = False

    components["walkability_index_v2"] = z

    # ---- Stats ---------------------------------------------------------------
    logging.info("PC1 explained-variance ratio: %.4f", pca.explained_variance_ratio_[0])
    logging.info("Loadings (PC1, sign-anchored):")
    for name, load in zip(FEAT_COLS, loadings_pc1):
        logging.info("  %-40s %+.4f", name, load)

    for city in ALL_CITIES:
        city_z = z[components["city"] == city]
        logging.info("  %s  n=%d  mean=%.3f  std=%.3f",
                     city, len(city_z), city_z.mean(), city_z.std())

    # ---- Correlation with existing v1 (3-city only) --------------------------
    if WALKABILITY_CSV.exists():
        wi_old = pd.read_csv(WALKABILITY_CSV)
        merged = components.merge(
            wi_old[["point_id", "city", "walkability_index_v1"]],
            on=["point_id", "city"], how="left",
        )
        v1 = merged["walkability_index_v1"].to_numpy(dtype=np.float64)
        v2 = merged["walkability_index_v2"].to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(v1) & np.isfinite(v2)
        if valid_mask.sum() > 0:
            rho = spearmanr(v1[valid_mask], v2[valid_mask]).statistic
            logging.info("Spearman(v1, new_v2) on 3-city overlap: %.4f", rho)
    else:
        logging.warning("walkability_index.csv not found; skipping v1 correlation check.")
        merged = components.copy()
        merged["walkability_index_v1"] = np.nan

    if args.dry_run:
        logging.info("[dry-run] Not writing files.")
        return 0

    # ---- Write walkability_index.csv ----------------------------------------
    # Preserve any existing walkability_index_v1 for 3-city data.
    # Add NaN walkability_index_v1 for Pittsburgh (not available from construct).
    if WALKABILITY_CSV.exists():
        wi_old = pd.read_csv(WALKABILITY_CSV)
        v1_lookup = wi_old.set_index(["point_id", "city"])["walkability_index_v1"].to_dict()
    else:
        v1_lookup: dict = {}

    out = components[["point_id", "city"]].copy()
    out["walkability_index_v1"] = [
        v1_lookup.get((int(row.point_id), row.city), np.nan)
        for row in out.itertuples()
    ]
    out["walkability_index_v2"] = components["walkability_index_v2"].values

    WALKABILITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WALKABILITY_CSV, index=False)
    logging.info("Wrote %s  (%d rows)", WALKABILITY_CSV, len(out))

    # ---- Write build report -------------------------------------------------
    report = {
        "reference_cities": list(REFERENCE_CITIES),
        "all_cities": list(ALL_CITIES),
        "feat_cols": FEAT_COLS,
        "crosswalk_source": "heading_overture_labels_v2.csv max across 4 headings (marking=only)",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_loadings_pc1": loadings_pc1.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sign_flipped": sign_flipped,
        "anchor": "sidewalk_present (positive correlation enforced)",
        "n_reference": int(ref_mask.sum()),
        "n_total": int(len(out)),
        "per_city": {
            city: {
                "n": int((components["city"] == city).sum()),
                "mean": float(z[components["city"] == city].mean()),
                "std":  float(z[components["city"] == city].std()),
                "p05":  float(np.quantile(z[components["city"] == city], 0.05)),
                "p95":  float(np.quantile(z[components["city"] == city], 0.95)),
            }
            for city in ALL_CITIES
        },
    }
    BUILD_JSON.parent.mkdir(parents=True, exist_ok=True)
    BUILD_JSON.write_text(json.dumps(report, indent=2))
    logging.info("Wrote build report: %s", BUILD_JSON)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
