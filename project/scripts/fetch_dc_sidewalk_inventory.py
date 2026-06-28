"""
DC sidewalk ground-truth from the official Open Data DC "Sidewalks" inventory.

Why this instead of DC Project Sidewalk
---------------------------------------
DC Project Sidewalk is unrecoverable for our purposes: the live API is 503, the
six-city Dataverse export excludes DC, and PS's label schema has no usable sidewalk-
presence signal here. DC's *government* planimetric Sidewalks layer (Open Data DC,
DC GIS) is public, authoritative, and a one-click GeoJSON -- a better sidewalk
ground truth than crowdsourced PS. This un-sticks DC for the sidewalk label audit
(previously "DC permanently 503").

What it does
------------
1. Reads the DC Sidewalks planimetric polygons -- from --local-geojson (recommended,
   one click) or an ArcGIS REST --service-url.
2. Matches each DC lock point to the nearest sidewalk polygon in DC UTM (EPSG:32618);
   present=1 if within --match-radius-m (default 25 m, matching audit_label_noise.py).
3. Compares dc_official_sidewalk_present vs overture_sidewalk_present (Cohen's kappa,
   AUROC, confusion, prevalence).

No API key required.

How to get the GeoJSON (recommended path)
-----------------------------------------
Open https://opendata.dc.gov/datasets/DCGIS::sidewalks -> Download -> GeoJSON,
save it, then pass the path with --local-geojson.

Outputs
-------
  project/data/processed/labels/dc_official_sidewalk_labels.csv   (point_id, city, dc_official_sidewalk_present)
  project/artifacts/reports/dc_official_sidewalk_audit.json       (vs-Overture agreement)

Run (Windows, repo root)
------------------------
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_dc_sidewalk_inventory.py --local-geojson C:\\path\\to\\Sidewalks.geojson
    # or, if you have a working REST endpoint:
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_dc_sidewalk_inventory.py --service-url "https://.../MapServer/<id>/query" --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
RAW_OUT = REPO_ROOT / "project/data/external/dc_opendata/sidewalks.geojson"
LABELS_OUT = REPO_ROOT / "project/data/processed/labels/dc_official_sidewalk_labels.csv"
REPORT_OUT = REPO_ROOT / "project/artifacts/reports/dc_official_sidewalk_audit.json"

DC_UTM_EPSG = 32618
PAGE_SIZE = 1000
USER_AGENT = "WalkBench-fetch-dc-sidewalks/1.0"


def download_geojson(service_url: str, page_size: int) -> dict:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": "1=1", "outFields": "*", "outSR": "4326", "f": "geojson",
            "resultOffset": offset, "resultRecordCount": page_size, "returnGeometry": "true",
        }
        resp = session.get(service_url, params=params, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        page = data.get("features", []) if isinstance(data, dict) else []
        features.extend(page)
        logging.info("Fetched %d features (offset=%d, total=%d)", len(page), offset, len(features))
        if len(page) < page_size:
            break
        offset += page_size
    return {"type": "FeatureCollection", "features": features}


def derive_sidewalk_labels(sidewalks, points: pd.DataFrame, match_radius_m: float) -> pd.DataFrame:
    import geopandas as gpd

    sidewalks = sidewalks[sidewalks.geometry.notna() & ~sidewalks.geometry.is_empty]
    if sidewalks.empty:
        raise ValueError("No sidewalk geometries parsed from the input.")
    if sidewalks.crs is None:
        sidewalks = sidewalks.set_crs("EPSG:4326")
    sidewalks = sidewalks.to_crs(epsg=DC_UTM_EPSG)[["geometry"]]

    pts = gpd.GeoDataFrame(
        points.copy(), geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=DC_UTM_EPSG)

    joined = gpd.sjoin_nearest(pts, sidewalks, how="left", max_distance=match_radius_m, distance_col="dist_m")
    joined = joined[~joined.index.duplicated(keep="first")]

    out = points[["point_id", "city"]].copy()
    out["dc_official_sidewalk_present"] = joined["index_right"].notna().astype(int).to_numpy()
    out["dist_to_sidewalk_m"] = joined["dist_m"].to_numpy()
    return out


def agreement(reference: np.ndarray, derived: np.ndarray) -> dict:
    valid = np.isfinite(reference) & np.isfinite(derived)
    a = reference[valid].astype(int)
    b = derived[valid].astype(int)
    n = int(valid.sum())
    if n < 30 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return {"n": n, "kappa": float("nan"), "auroc_overture_vs_official": float("nan"),
                "overture_prevalence": float(a.mean()) if n else float("nan"),
                "official_prevalence": float(b.mean()) if n else float("nan"),
                "note": "degenerate (n<30 or single class -- expected if both are ~all-sidewalk)"}
    return {
        "n": n,
        "kappa": round(float(cohen_kappa_score(b, a)), 4),
        "auroc_overture_vs_official": round(float(roc_auc_score(b, a)), 4),
        "raw_agreement": round(float((a == b).mean()), 4),
        "confusion_official_rows_overture_cols": confusion_matrix(b, a, labels=[0, 1]).tolist(),
        "overture_prevalence": round(float(a.mean()), 4),
        "official_prevalence": round(float(b.mean()), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DC official sidewalk inventory -> per-point label, vs Overture.")
    p.add_argument("--local-geojson", type=Path, help="DC Sidewalks GeoJSON downloaded from Open Data DC (recommended).")
    p.add_argument("--service-url", help="ArcGIS REST query endpoint for the DC Sidewalks layer (alternative to --local-geojson).")
    p.add_argument("--match-radius-m", type=float, default=25.0)
    p.add_argument("--page-size", type=int, default=PAGE_SIZE)
    p.add_argument("--check-only", action="store_true", help="With --service-url: fetch one page, report fields/geometry, exit.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    if args.check_only:
        if not args.service_url:
            logging.error("--check-only needs --service-url.")
            return 1
        gj = download_geojson(args.service_url, page_size=5)
        feats = gj["features"]
        logging.info("OK -- %d sample features. First geometry type: %s",
                     len(feats), feats[0]["geometry"]["type"] if feats else "none")
        if feats:
            logging.info("First feature properties: %s", list(feats[0].get("properties", {}).keys()))
        return 0

    import geopandas as gpd

    if args.local_geojson:
        logging.info("Reading local GeoJSON: %s", args.local_geojson)
        sidewalks = gpd.read_file(args.local_geojson)
    elif args.service_url:
        geojson = download_geojson(args.service_url, args.page_size)
        sidewalks = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
        RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
        sidewalks.to_file(RAW_OUT, driver="GeoJSON")
        logging.info("Wrote raw sidewalks: %s (%d features)", RAW_OUT, len(sidewalks))
    else:
        logging.error(
            "Provide --local-geojson (recommended) or --service-url.\n"
            "  Download: https://opendata.dc.gov/datasets/DCGIS::sidewalks -> Download -> GeoJSON."
        )
        return 1

    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "lat", "lon", "overture_sidewalk_present"])
    dc = fla[fla["city"] == "dc"].copy()
    dc["point_id"] = dc["point_id"].astype(int)
    logging.info("DC lock points: %d", len(dc))

    labels = derive_sidewalk_labels(sidewalks, dc, args.match_radius_m)
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_OUT, index=False)
    logging.info("Wrote labels: %s  (official sidewalk prevalence=%.3f)",
                 LABELS_OUT, labels["dc_official_sidewalk_present"].mean())

    merged = dc.merge(labels, on=["point_id", "city"], how="inner")
    stats = agreement(
        merged["overture_sidewalk_present"].to_numpy(dtype=np.float32),
        merged["dc_official_sidewalk_present"].to_numpy(dtype=np.float32),
    )
    report = {"match_radius_m": args.match_radius_m, "n_dc_points": int(len(merged)), "overture_vs_official": stats}
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote audit: %s", REPORT_OUT)
    logging.info("=" * 64)
    logging.info("Overture vs DC-official sidewalk: kappa=%.3f  AUROC=%.3f  n=%d",
                 stats["kappa"] if isinstance(stats["kappa"], float) else float("nan"),
                 stats.get("auroc_overture_vs_official", float("nan")), stats["n"])
    logging.info("Prevalence: Overture=%.3f  Official=%.3f",
                 stats.get("overture_prevalence", float("nan")), stats.get("official_prevalence", float("nan")))
    logging.info("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
