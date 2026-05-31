"""
Label-noise denominator audit (S5).

Reports the agreement between Overture-derived binary labels (used as
training/eval ground truth) and Project Sidewalk crowdsourced labels (used
as a higher-quality external reference) at the same lock points. The ceiling
of model performance against Overture is bounded by Overture's agreement
with PS — a model that scores AUROC=0.85 against Overture when Overture
itself only agrees with PS at AUROC=0.70 is mostly fitting Overture-specific
label noise.

For each (city, target ∈ {sidewalk_present, crosswalk_present}) computes:

  - Cohen's κ between Overture and PS (chance-corrected agreement)
  - AUROC of Overture-as-score against PS-as-truth (and vice versa)
  - Confusion matrix
  - Per-cell prevalence

Only Seattle has full PS coverage. DC has partial PS in the
sidewalk-equity-study clone; report counts and skip if too sparse. MSP has
no PS coverage and is reported as "not applicable".

Run:
    python project/scripts/audit_label_noise.py
    python project/scripts/audit_label_noise.py --match-radius-m 25 \\
        --cities seattle dc

Output:
    project/artifacts/reports/label_noise_audit.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERTURE_CSV = REPO_ROOT / "project/data/processed/labels/overture_targets.csv"
POINTS_DIR   = REPO_ROOT / "project/data/processed/points"
PS_DIR       = REPO_ROOT / "project/data/external/sidewalk-equity-study"
REPORT       = REPO_ROOT / "project/artifacts/reports"

CITY_UTM_EPSG = {"msp": 26915, "seattle": 32610, "dc": 32618, "pittsburgh": 32617}
PS_SIDEWALK_POSITIVE = {"CurbRamp", "Crosswalk"}
PS_SIDEWALK_NEGATIVE = {"NoSidewalk", "NoCurbRamp"}
PS_CROSSWALK_POSITIVE = {"Crosswalk"}


def load_ps(city: str) -> pd.DataFrame:
    """Load PS labels for one city. Schema: lat, lon, label_type, city."""
    # The sidewalk-equity-study layout puts files like
    # seattle/datasets/01-project-sidewalk-labels/attributes_YYYYMMDD.json
    # so we search the per-city subtree, not the filename.
    city_root = PS_DIR / city
    if not city_root.exists():
        # fallback: filename-contains-city
        candidates = (list(PS_DIR.rglob(f"*{city}*.csv"))
                      + list(PS_DIR.rglob(f"*{city}*.geojson"))
                      + list(PS_DIR.rglob(f"*{city}*.json")))
    else:
        candidates = (list(city_root.rglob("attributes_*.json"))
                      + list(city_root.rglob("attributes_*.geojson"))
                      + list(city_root.rglob("attributes_*.csv")))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return pd.DataFrame(columns=["lat", "lon", "label_type", "city"])
    out = []
    for p in candidates:
        if p.suffix == ".geojson":
            import geopandas as gpd
            gdf = gpd.read_file(p)
            gdf = gdf.to_crs("EPSG:4326")
            df = pd.DataFrame({
                "lon": gdf.geometry.x,
                "lat": gdf.geometry.y,
                "label_type": gdf.get("label_type", gdf.get("labelType", "")),
            })
        elif p.suffix == ".json":
            # PS attributes JSON is GeoJSON FeatureCollection
            try:
                import geopandas as gpd
                gdf = gpd.read_file(p)
                if gdf.geometry.iloc[0] is not None:
                    gdf = gdf.to_crs("EPSG:4326")
                    df = pd.DataFrame({
                        "lon": gdf.geometry.x, "lat": gdf.geometry.y,
                        "label_type": gdf.get("label_type", gdf.get("labelType",
                                       gdf.get("attribute_type", ""))),
                    })
                else:
                    raise ValueError("no geometry")
            except Exception:
                raw = json.loads(p.read_text(encoding="utf-8"))
                feats = raw.get("features", raw if isinstance(raw, list) else [])
                rows = []
                for f in feats:
                    coords = f.get("geometry", {}).get("coordinates")
                    props = f.get("properties", f)
                    if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        rows.append({
                            "lon": coords[0], "lat": coords[1],
                            "label_type": props.get("label_type",
                                                   props.get("labelType",
                                                             props.get("attribute_type", ""))),
                        })
                df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(p)
        df["city"] = city
        out.append(df)
    return pd.concat(out, ignore_index=True)


def derive_ps_point_labels(
    ps_df: pd.DataFrame,
    points_df: pd.DataFrame,
    city: str,
    match_radius_m: float,
) -> pd.DataFrame:
    from pyproj import Transformer
    from scipy.spatial import cKDTree

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

    tree = cKDTree(np.stack([ps_x, ps_y], axis=1))
    nearby = tree.query_ball_point(np.stack([pts_x, pts_y], axis=1), r=match_radius_m)

    types = ps_city["label_type"].astype(str).to_numpy()
    sw_pres, sw_abs, cw_pres = [], [], []
    for idxs in nearby:
        if not idxs:
            sw_pres.append(False); sw_abs.append(False); cw_pres.append(False); continue
        local = set(types[idxs])
        sw_pres.append(bool(local & PS_SIDEWALK_POSITIVE))
        sw_abs.append(bool(local & PS_SIDEWALK_NEGATIVE))
        cw_pres.append(bool(local & PS_CROSSWALK_POSITIVE))

    sw_label = np.where(np.array(sw_pres), 1.0,
                        np.where(np.array(sw_abs), 0.0, np.nan))
    cw_label = np.where(np.array(cw_pres), 1.0, 0.0)

    out = pts_city[["point_id", "city"]].copy()
    out["ps_sidewalk_present"] = sw_label
    out["ps_crosswalk_present"] = cw_label
    return out


def agreement(overture: np.ndarray, ps: np.ndarray) -> dict:
    valid = np.isfinite(overture) & np.isfinite(ps)
    o = overture[valid].astype(int)
    p = ps[valid].astype(int)
    n = int(valid.sum())
    if n < 30 or len(np.unique(o)) < 2 or len(np.unique(p)) < 2:
        return {"n": n, "kappa": float("nan"), "auroc_overture_vs_ps": float("nan"),
                "auroc_ps_vs_overture": float("nan"),
                "confusion": [[0, 0], [0, 0]],
                "overture_prevalence": float(o.mean()) if n else float("nan"),
                "ps_prevalence":       float(p.mean()) if n else float("nan")}
    cm = confusion_matrix(p, o, labels=[0, 1]).tolist()
    return {
        "n": n,
        "kappa": float(cohen_kappa_score(p, o)),
        "auroc_overture_vs_ps": float(roc_auc_score(p, o)),
        "auroc_ps_vs_overture": float(roc_auc_score(o, p)),
        "confusion": cm,
        "overture_prevalence": float(o.mean()),
        "ps_prevalence":       float(p.mean()),
        "raw_agreement":       float((o == p).mean()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overture vs Project Sidewalk label agreement audit.")
    p.add_argument("--cities", nargs="+", default=["seattle", "dc"])
    p.add_argument("--match-radius-m", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    REPORT.mkdir(parents=True, exist_ok=True)

    overture = pd.read_csv(OVERTURE_CSV)
    overture["point_id"] = overture["point_id"].astype(int)

    out: dict = {"match_radius_m": args.match_radius_m, "by_city": {}}
    for city in args.cities:
        pts_csv = POINTS_DIR / f"{city}_points_v2_final_lock.csv"
        if not pts_csv.exists():
            logging.warning("Missing points CSV for %s: %s — skipping.", city, pts_csv)
            continue
        pts = pd.read_csv(pts_csv)
        pts["city"] = city
        pts["point_id"] = pts["point_id"].astype(int)
        ps = load_ps(city)
        if ps.empty:
            logging.warning("No PS labels found for %s in %s — skipping.", city, PS_DIR)
            out["by_city"][city] = {"n_ps_labels": 0, "skipped": True}
            continue
        logging.info("[%s] PS labels loaded: %d", city, len(ps))
        ps_pts = derive_ps_point_labels(ps, pts, city, args.match_radius_m)
        joined = overture[overture["city"] == city].merge(
            ps_pts, on=["point_id", "city"], how="inner")
        logging.info("[%s] joined points: %d", city, len(joined))

        sw_col = "overture_sidewalk_present" if "overture_sidewalk_present" in joined.columns else "sidewalk_present"
        cw_col = "overture_crosswalk_present" if "overture_crosswalk_present" in joined.columns else "crosswalk_present"
        sw = agreement(joined[sw_col].to_numpy(dtype=np.float32),
                       joined["ps_sidewalk_present"].to_numpy(dtype=np.float32))
        cw = agreement(joined[cw_col].to_numpy(dtype=np.float32),
                       joined["ps_crosswalk_present"].to_numpy(dtype=np.float32))
        out["by_city"][city] = {
            "n_ps_labels_total": int(len(ps)),
            "n_lock_points":     int(len(joined)),
            "sidewalk_present":  sw,
            "crosswalk_present": cw,
        }
        logging.info("[%s] sidewalk: κ=%.3f AUROC(Overture vs PS)=%.3f n=%d",
                     city, sw["kappa"] if np.isfinite(sw["kappa"]) else float("nan"),
                     sw["auroc_overture_vs_ps"] if np.isfinite(sw["auroc_overture_vs_ps"]) else float("nan"),
                     sw["n"])
        logging.info("[%s] crosswalk: κ=%.3f AUROC(Overture vs PS)=%.3f n=%d",
                     city, cw["kappa"] if np.isfinite(cw["kappa"]) else float("nan"),
                     cw["auroc_overture_vs_ps"] if np.isfinite(cw["auroc_overture_vs_ps"]) else float("nan"),
                     cw["n"])

    out_path = REPORT / "label_noise_audit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
