"""
Calibrate SAM2 and VLM caption channels against external ground truth.

References supported:
  - overture: Overture sidewalk_present / crosswalk_present (geometric proxy; legacy)
  - project_sidewalk: PS crowdsourced sidewalk audits (Seattle, DC)
  - ape: Annotations for Pedestrian Environment (Seattle subset)

For each (channel, reference, city) computes precision, recall, F1, AUROC.
PS/APE labels are point-wise; we spatially join them to v2 lock points by
nearest neighbour within --ps-match-radius-m metres.

Outputs:
    project/artifacts/reports/calibration_audit_results.json
    project/artifacts/reports/calibration_table.json     (when --references != [overture])

Run:
    python project/scripts/calibrate_audit_channels.py
    python project/scripts/calibrate_audit_channels.py --sam2-sidewalk-threshold 0.05
    python project/scripts/calibrate_audit_channels.py \
        --references project_sidewalk ape overture \
        --channels caption sam2 \
        --cities seattle dc
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERTURE_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "overture_targets.csv"
LABELS_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"
REPORT_DIR = REPO_ROOT / "project" / "artifacts" / "reports"
POINTS_DIR = REPO_ROOT / "project" / "data" / "processed" / "points"
APE_CSV = REPO_ROOT / "project" / "data" / "external" / "ape" / "ape_labels_combined.csv"

CITY_POINTS_CSV = {
    "msp":     POINTS_DIR / "msp_points_v2_final_lock.csv",
    "seattle": POINTS_DIR / "seattle_points_v2_final_lock.csv",
    "dc":      POINTS_DIR / "dc_points_v2_final_lock.csv",
}
CITY_UTM_EPSG = {"msp": 26915, "seattle": 32610, "dc": 32618}

# PS label-type → which derived channel they validate
PS_SIDEWALK_POSITIVE = {"CurbRamp", "Crosswalk"}    # presence-implying
PS_SIDEWALK_NEGATIVE = {"NoSidewalk", "NoCurbRamp"} # absence-implying
PS_CROSSWALK_POSITIVE = {"Crosswalk"}

# SAM2 continuous feature → binary sidewalk/crosswalk thresholds (tunable)
DEFAULT_SIDEWALK_THRESH = 0.05   # sidewalk_frac_max > this → sidewalk present
DEFAULT_CROSSWALK_THRESH = 0.02  # crosswalk_frac_max > this → crosswalk present


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "auroc": float("nan"), "n": int(len(y_true)), "n_pos": int(y_true.sum())}
    out = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n": int(len(y_true)),
        "n_pos_gt": int(y_true.sum()),
        "n_pos_pred": int(y_pred.sum()),
    }
    if scores is not None:
        try:
            out["auroc"] = round(float(roc_auc_score(y_true, scores)), 4)
        except Exception:
            out["auroc"] = float("nan")
    return out


def run_calibration(
    df: pd.DataFrame,
    city_filter: str | None,
    sidewalk_thresh: float,
    crosswalk_thresh: float,
) -> dict[str, dict]:
    if city_filter:
        df = df[df["city"] == city_filter].copy()
    n = len(df)
    if n == 0:
        return {}

    def _resolve_col(frame: pd.DataFrame, base: str) -> str:
        if base in frame.columns:
            return base
        # Handle merge suffixes like _x/_y if present.
        candidates = [c for c in frame.columns if c.startswith(base)]
        if not candidates:
            raise KeyError(f"Missing required column '{base}'. Available: {list(frame.columns)}")
        return candidates[0]

    sw_col = _resolve_col(df, "overture_sidewalk_present")
    cw_col = _resolve_col(df, "overture_crosswalk_present")
    gt_sw = df[sw_col].to_numpy(dtype=np.int32)
    gt_cw = df[cw_col].to_numpy(dtype=np.int32)

    # SAM2 sidewalk: binary from threshold on sidewalk_frac_max
    sam2_sw_score = df["sidewalk_frac_max"].fillna(0).to_numpy(dtype=np.float32)
    sam2_sw_pred = (sam2_sw_score > sidewalk_thresh).astype(np.int32)

    # SAM2 crosswalk: binary from threshold on crosswalk_frac_max
    sam2_cw_score = df["crosswalk_frac_max"].fillna(0).to_numpy(dtype=np.float32)
    sam2_cw_pred = (sam2_cw_score > crosswalk_thresh).astype(np.int32)

    # Caption sidewalk: ca_sidewalk in {"one_side", "both_sides"} → present
    # In the labels file, caption sidewalk is stored as ca_sidewalk (numeric/string)
    cap_sw_raw = df["ca_sidewalk"].fillna(0)
    if cap_sw_raw.dtype == object:
        cap_sw_pred = cap_sw_raw.isin({"one_side", "both_sides"}).astype(np.int32).to_numpy()
        cap_sw_score = None
    else:
        cap_sw_pred = (cap_sw_raw > 0).astype(np.int32).to_numpy()
        cap_sw_score = cap_sw_raw.to_numpy(dtype=np.float32)

    # Caption crosswalk: ca_crosswalk == "present" or > 0
    cap_cw_raw = df["ca_crosswalk"].fillna(0)
    if cap_cw_raw.dtype == object:
        cap_cw_pred = (cap_cw_raw == "present").astype(np.int32).to_numpy()
        cap_cw_score = None
    else:
        cap_cw_pred = (cap_cw_raw > 0).astype(np.int32).to_numpy()
        cap_cw_score = cap_cw_raw.to_numpy(dtype=np.float32)

    return {
        "sam2_vs_overture_sidewalk": binary_metrics(gt_sw, sam2_sw_pred, sam2_sw_score),
        "caption_vs_overture_sidewalk": binary_metrics(gt_sw, cap_sw_pred, cap_sw_score),
        "sam2_vs_overture_crosswalk": binary_metrics(gt_cw, sam2_cw_pred, sam2_cw_score),
        "caption_vs_overture_crosswalk": binary_metrics(gt_cw, cap_cw_pred, cap_cw_score),
        "n_points": n,
    }


def print_calibration_table(results: dict[str, dict]) -> None:
    comparisons = [
        "sam2_vs_overture_sidewalk",
        "caption_vs_overture_sidewalk",
        "sam2_vs_overture_crosswalk",
        "caption_vs_overture_crosswalk",
    ]
    scopes = ["msp", "seattle", "dc", "combined"]

    logging.info("")
    logging.info("─" * 100)
    logging.info("CALIBRATION AUDIT: channel vs Overture geometric ground truth")
    logging.info(
        "%-40s %-10s %-10s %-10s %-10s %-10s",
        "comparison", "city", "precision", "recall", "f1", "auroc",
    )
    logging.info("─" * 100)
    for scope in scopes:
        if scope not in results:
            continue
        for comp in comparisons:
            m = results[scope].get(comp, {})
            logging.info(
                "%-40s %-10s %-10.4f %-10.4f %-10.4f %-10s",
                comp,
                scope,
                m.get("precision", float("nan")),
                m.get("recall", float("nan")),
                m.get("f1", float("nan")),
                f"{m.get('auroc', float('nan')):.4f}" if "auroc" in m else "n/a",
            )
    logging.info("─" * 100)


def derive_ps_point_labels(
    ps_df: pd.DataFrame,
    points_df: pd.DataFrame,
    city: str,
    match_radius_m: float,
) -> pd.DataFrame:
    """Project PS crowdsourced labels onto v2 lock points by nearest-neighbour.

    For each lock point, returns:
        ps_sidewalk_present: 1 if any PS positive sidewalk label within radius,
                             0 if any PS negative (NoSidewalk) within radius and no positive,
                             NaN otherwise (no signal — exclude from metrics).
        ps_crosswalk_present: 1 if any Crosswalk label within radius, 0 otherwise.
    """
    from pyproj import Transformer

    pts_city = points_df[points_df["city"] == city].copy()
    ps_city = ps_df[ps_df["city"] == city].copy()
    if ps_city.empty or pts_city.empty:
        out = pts_city[["point_id", "city"]].copy()
        out["ps_sidewalk_present"] = float("nan")
        out["ps_crosswalk_present"] = float("nan")
        return out

    epsg = CITY_UTM_EPSG[city]
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts_x, pts_y = tf.transform(pts_city["lon"].to_numpy(), pts_city["lat"].to_numpy())
    ps_x, ps_y = tf.transform(ps_city["lon"].to_numpy(), ps_city["lat"].to_numpy())

    # KDTree for fast radius queries
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([ps_x, ps_y], axis=1))
    pts_xy = np.stack([pts_x, pts_y], axis=1)
    nearby = tree.query_ball_point(pts_xy, r=match_radius_m)

    types = ps_city["label_type"].to_numpy()
    sw_pres, sw_abs, cw_pres = [], [], []
    for idxs in nearby:
        if not idxs:
            sw_pres.append(False)
            sw_abs.append(False)
            cw_pres.append(False)
            continue
        local_types = set(types[idxs])
        sw_pres.append(bool(local_types & PS_SIDEWALK_POSITIVE))
        sw_abs.append(bool(local_types & PS_SIDEWALK_NEGATIVE))
        cw_pres.append(bool(local_types & PS_CROSSWALK_POSITIVE))

    sw_label = np.where(np.array(sw_pres), 1.0, np.where(np.array(sw_abs), 0.0, np.nan))
    cw_label = np.where(np.array(cw_pres), 1.0, 0.0)
    out = pts_city[["point_id", "city"]].copy()
    out["ps_sidewalk_present"] = sw_label
    out["ps_crosswalk_present"] = cw_label
    return out


def calibrate_against_ps(
    df: pd.DataFrame,
    ps_labels: pd.DataFrame,
    sidewalk_thresh: float,
    crosswalk_thresh: float,
    channels: list[str],
) -> dict[str, dict]:
    """Channel-vs-PS metrics. df has SAM2/caption fields; ps_labels has ps_*_present."""
    merged = df.merge(ps_labels, on=["point_id", "city"], how="inner")
    out: dict[str, dict] = {}

    sw_mask = ~np.isnan(merged["ps_sidewalk_present"].to_numpy(dtype=np.float32))
    sw_sub = merged.loc[sw_mask].copy()
    if len(sw_sub) > 0:
        gt_sw = sw_sub["ps_sidewalk_present"].astype(int).to_numpy()
        if "sam2" in channels:
            sam2_score = sw_sub["sidewalk_frac_max"].fillna(0).to_numpy(dtype=np.float32)
            sam2_pred = (sam2_score > sidewalk_thresh).astype(int)
            out["sam2_vs_ps_sidewalk"] = binary_metrics(gt_sw, sam2_pred, sam2_score)
        if "caption" in channels:
            cap_raw = sw_sub["ca_sidewalk"].fillna(0)
            if cap_raw.dtype == object:
                cap_pred = cap_raw.isin({"one_side", "both_sides"}).astype(int).to_numpy()
                out["caption_vs_ps_sidewalk"] = binary_metrics(gt_sw, cap_pred, None)
            else:
                cap_score = cap_raw.to_numpy(dtype=np.float32)
                out["caption_vs_ps_sidewalk"] = binary_metrics(gt_sw, (cap_score > 0).astype(int), cap_score)

    cw_mask = ~np.isnan(merged["ps_crosswalk_present"].to_numpy(dtype=np.float32))
    cw_sub = merged.loc[cw_mask].copy()
    if len(cw_sub) > 0:
        gt_cw = cw_sub["ps_crosswalk_present"].astype(int).to_numpy()
        if "sam2" in channels:
            sam2_score = cw_sub["crosswalk_frac_max"].fillna(0).to_numpy(dtype=np.float32)
            sam2_pred = (sam2_score > crosswalk_thresh).astype(int)
            out["sam2_vs_ps_crosswalk"] = binary_metrics(gt_cw, sam2_pred, sam2_score)
        if "caption" in channels:
            cap_raw = cw_sub["ca_crosswalk"].fillna(0)
            if cap_raw.dtype == object:
                cap_pred = (cap_raw == "present").astype(int).to_numpy()
                out["caption_vs_ps_crosswalk"] = binary_metrics(gt_cw, cap_pred, None)
            else:
                cap_score = cap_raw.to_numpy(dtype=np.float32)
                out["caption_vs_ps_crosswalk"] = binary_metrics(gt_cw, (cap_score > 0).astype(int), cap_score)

    out["n_points"] = int(len(merged))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate SAM2/caption channels against external GT")
    p.add_argument("--sam2-sidewalk-threshold", type=float, default=DEFAULT_SIDEWALK_THRESH)
    p.add_argument("--sam2-crosswalk-threshold", type=float, default=DEFAULT_CROSSWALK_THRESH)
    p.add_argument("--references", nargs="+", choices=["overture", "project_sidewalk", "ape"],
                   default=["overture"])
    p.add_argument("--channels", nargs="+", choices=["caption", "sam2"], default=["caption", "sam2"])
    p.add_argument("--cities", nargs="+", choices=["msp", "seattle", "dc"],
                   default=["msp", "seattle", "dc"])
    p.add_argument("--ps-match-radius-m", type=float, default=25.0,
                   help="PS/APE label spatial join radius in metres (default 25)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in (OVERTURE_CSV, LABELS_CSV, LOCK_FILE):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            if path == OVERTURE_CSV:
                logging.error(
                    "Run first: python project/scripts/derive_overture_targets.py --city both"
                )
            return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lock_ids = {int(x) for x in LOCK_FILE.read_text().strip().splitlines()}

    overture = pd.read_csv(OVERTURE_CSV)
    overture["point_id"] = overture["point_id"].astype(int)
    overture = overture[overture["point_id"].isin(lock_ids)]

    labels = pd.read_csv(LABELS_CSV)
    labels["point_id"] = labels["point_id"].astype(int)
    labels = labels[labels["point_id"].isin(lock_ids)]

    # Join Overture targets onto labels using city-aware key to avoid cross-city point_id collisions.
    overture_cols = ["point_id", "sidewalk_present", "crosswalk_present"]
    merge_keys = ["point_id"]
    if "city" in labels.columns and "city" in overture.columns:
        overture_cols.insert(1, "city")
        merge_keys = ["point_id", "city"]
    merged = labels.merge(
        overture[overture_cols].rename(columns={
            "sidewalk_present": "overture_sidewalk_present",
            "crosswalk_present": "overture_crosswalk_present",
        }),
        on=merge_keys,
        how="inner",
    )
    logging.info(
        "Merged rows: %d (labels=%d, overture=%d)",
        len(merged), len(labels), len(overture),
    )

    results: dict[str, dict] = {}
    for scope in ["msp", "seattle", "dc", None]:
        key = scope if scope else "combined"
        results[key] = run_calibration(
            merged, scope,
            sidewalk_thresh=args.sam2_sidewalk_threshold,
            crosswalk_thresh=args.sam2_crosswalk_threshold,
        )
        logging.info(
            "  %s: n=%d", key, results[key].get("n_points", 0)
        )

    print_calibration_table(results)

    out_path = REPORT_DIR / "calibration_audit_results.json"
    out_path.write_text(json.dumps(
        {
            "thresholds": {
                "sam2_sidewalk": args.sam2_sidewalk_threshold,
                "sam2_crosswalk": args.sam2_crosswalk_threshold,
            },
            "results": results,
        },
        indent=2,
    ))
    logging.info("Saved: %s", out_path)

    # Project Sidewalk + APE arms (only if requested).
    ps_or_ape = [r for r in args.references if r in ("project_sidewalk", "ape")]
    if ps_or_ape:
        if not APE_CSV.exists():
            logging.error(
                "PS/APE labels not found at %s. Run: python project/scripts/fetch_ape_dataset.py "
                "--cities seattle dc", APE_CSV,
            )
            return 1
        ape = pd.read_csv(APE_CSV)
        if "project_sidewalk" not in ps_or_ape:
            ape = ape[ape["source"] != "project_sidewalk"]
        if "ape" not in ps_or_ape:
            ape = ape[ape["source"] != "ape"]

        # Need lat/lon per lock point — load points csv per city and concat.
        points_frames = []
        for city in args.cities:
            pts_path = CITY_POINTS_CSV.get(city)
            if pts_path is None or not pts_path.exists():
                logging.warning("No points CSV for %s, skipping", city)
                continue
            df_pts = pd.read_csv(pts_path)
            df_pts["city"] = city
            df_pts["point_id"] = df_pts["point_id"].astype(int)
            points_frames.append(df_pts[["point_id", "city", "lat", "lon"]])
        if not points_frames:
            logging.error("No points loaded — PS/APE arm aborted.")
            return 1
        all_points = pd.concat(points_frames, ignore_index=True)

        ext_results: dict[str, dict] = {}
        for city in args.cities:
            ps_lab = derive_ps_point_labels(ape, all_points, city, args.ps_match_radius_m)
            sub = merged[merged["city"] == city] if "city" in merged.columns else merged
            ext_results[city] = calibrate_against_ps(
                sub, ps_lab,
                sidewalk_thresh=args.sam2_sidewalk_threshold,
                crosswalk_thresh=args.sam2_crosswalk_threshold,
                channels=args.channels,
            )
            logging.info("[%s] PS/APE arm: %s", city,
                         {k: v for k, v in ext_results[city].items() if k != "n_points"})

        ext_path = REPORT_DIR / "calibration_table.json"
        ext_path.write_text(json.dumps({
            "references": args.references,
            "channels": args.channels,
            "ps_match_radius_m": args.ps_match_radius_m,
            "results_by_city": ext_results,
        }, indent=2))
        logging.info("Saved: %s", ext_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
