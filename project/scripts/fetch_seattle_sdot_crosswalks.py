"""
Pull Seattle SDOT "Marked Crosswalks" (authoritative city inventory) and derive a
per-point crosswalk ground-truth for the Seattle WalkBench lock points.

Why
---
The crosswalk label ceiling (Seattle Overture-vs-PS kappa=0.049) was measured
against Project Sidewalk, which is a curb-ramp/sidewalk dataset, not a crosswalk
source. SDOT publishes an authoritative marked-crosswalk inventory. If Overture
agrees well with SDOT, the "label ceiling" was a PS artifact and crosswalk may be
revivable as a target.

Access notes
------------
The SDOT ArcGIS service (gisdata.seattle.gov) unloads when idle and reports
"Service not started" until the first request wakes it (~10-30 s). This script
uses f=json (the older server does NOT support f=geojson) with returnIdsOnly
paging and wake-retries. If the service stays down, download the GeoJSON manually
(see --local-geojson) from the Seattle GeoData portal.

No API key required.

Outputs
-------
  project/data/processed/labels/seattle_sdot_crosswalk_labels.csv   (point_id, city, sdot_crosswalk_present)
  project/artifacts/reports/seattle_sdot_crosswalk_audit.json       (vs-Overture agreement)

Run (Windows, repo ROOT = C:\\Users\\kvars\\Desktop\\WalkCLIP)
-------------------------------------------------------------
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_seattle_sdot_crosswalks.py --check-only
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_seattle_sdot_crosswalks.py
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_seattle_sdot_crosswalks.py --local-geojson "C:\\path\\to\\Marked_Crosswalks.geojson"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
RAW_OUT = REPO_ROOT / "project/data/external/seattle_sdot/marked_crosswalks.geojson"
LABELS_OUT = REPO_ROOT / "project/data/processed/labels/seattle_sdot_crosswalk_labels.csv"
REPORT_OUT = REPO_ROOT / "project/artifacts/reports/seattle_sdot_crosswalk_audit.json"

SEATTLE_UTM_EPSG = 32610
SDOT_LAYER_URL = "https://gisdata.seattle.gov/server/rest/services/SDOT/SDOT_Pedestrian/MapServer/1"
ID_CHUNK = 400
USER_AGENT = "WalkBench-fetch-seattle-sdot-crosswalks/1.0"


def _get_json(session: requests.Session, url: str, params: dict, retries: int, wait_s: float) -> dict:
    """GET ArcGIS JSON, retrying through 'Service not started' / 5xx / non-JSON wake-up."""
    last = ""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params={**params, "f": "json"}, timeout=120)
            data = resp.json()  # raises if HTML (service waking) -> caught below
            if isinstance(data, dict) and "error" in data:
                last = str(data["error"])
                raise ValueError(last)
            return data
        except Exception as exc:
            last = str(exc)
            logging.warning("ArcGIS attempt %d/%d not ready (%s) -- wait %.0fs", attempt, retries, last[:80], wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"ArcGIS endpoint not ready after {retries} tries: {last}")


def download_features(layer_url: str, retries: int, wait_s: float) -> list[dict]:
    """Return Esri-JSON point features for all marked crosswalks via returnIdsOnly paging."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    ids = _get_json(session, f"{layer_url}/query", {"where": "1=1", "returnIdsOnly": "true"}, retries, wait_s)
    oid_field = ids.get("objectIdFieldName", "OBJECTID")
    oids = ids.get("objectIds") or []
    logging.info("Marked-crosswalk OIDs: %d (oid field=%s)", len(oids), oid_field)
    feats: list[dict] = []
    for i in range(0, len(oids), ID_CHUNK):
        chunk = oids[i:i + ID_CHUNK]
        data = _get_json(
            session, f"{layer_url}/query",
            {"where": f"{oid_field} IN ({','.join(map(str, chunk))})",
             "outFields": "*", "outSR": "4326", "returnGeometry": "true"},
            retries, wait_s,
        )
        feats.extend(data.get("features", []))
        logging.info("Fetched %d/%d features", len(feats), len(oids))
    return feats


def esri_points_to_gdf(features: list[dict]):
    import geopandas as gpd
    from shapely.geometry import Point

    geoms, attrs = [], []
    for f in features:
        g = f.get("geometry") or {}
        x, y = g.get("x"), g.get("y")
        if x is not None and y is not None:
            geoms.append(Point(x, y))
            attrs.append(f.get("attributes", {}))
    if not geoms:
        raise ValueError("No point geometries in the SDOT response (unexpected geometry type?).")
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs="EPSG:4326")


def load_crosswalks_gdf(args):
    import geopandas as gpd

    if args.local_geojson:
        logging.info("Reading local GeoJSON: %s", args.local_geojson)
        return gpd.read_file(args.local_geojson)
    feats = download_features(args.layer_url, args.retries, args.wait_s)
    gdf = esri_points_to_gdf(feats)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(RAW_OUT, driver="GeoJSON")
    logging.info("Wrote raw crosswalks: %s (%d features)", RAW_OUT, len(gdf))
    return gdf


def derive_crosswalk_labels(crosswalks, points: pd.DataFrame, match_radius_m: float) -> pd.DataFrame:
    import geopandas as gpd

    cw = crosswalks[crosswalks.geometry.notna() & ~crosswalks.geometry.is_empty]
    if cw.crs is None:
        cw = cw.set_crs("EPSG:4326")
    cw = cw.to_crs(epsg=SEATTLE_UTM_EPSG)[["geometry"]]

    pts = gpd.GeoDataFrame(
        points.copy(), geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=SEATTLE_UTM_EPSG)

    joined = gpd.sjoin_nearest(pts, cw, how="left", max_distance=match_radius_m, distance_col="dist_m")
    joined = joined[~joined.index.duplicated(keep="first")]

    out = points[["point_id", "city"]].copy()
    out["sdot_crosswalk_present"] = joined["index_right"].notna().astype(int).to_numpy()
    out["dist_to_crosswalk_m"] = joined["dist_m"].to_numpy()
    return out


def agreement(reference: np.ndarray, derived: np.ndarray) -> dict:
    valid = np.isfinite(reference) & np.isfinite(derived)
    a, b = reference[valid].astype(int), derived[valid].astype(int)
    n = int(valid.sum())
    if n < 30 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return {"n": n, "kappa": float("nan"), "auroc_overture_vs_sdot": float("nan"),
                "overture_prevalence": float(a.mean()) if n else float("nan"),
                "sdot_prevalence": float(b.mean()) if n else float("nan"),
                "note": "degenerate (n<30 or single class)"}
    return {
        "n": n,
        "kappa": round(float(cohen_kappa_score(b, a)), 4),
        "auroc_overture_vs_sdot": round(float(roc_auc_score(b, a)), 4),
        "raw_agreement": round(float((a == b).mean()), 4),
        "confusion_sdot_rows_overture_cols": confusion_matrix(b, a, labels=[0, 1]).tolist(),
        "overture_prevalence": round(float(a.mean()), 4),
        "sdot_prevalence": round(float(b.mean()), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull Seattle SDOT marked crosswalks and audit vs Overture.")
    p.add_argument("--layer-url", default=SDOT_LAYER_URL, help="ArcGIS REST layer URL (no /query).")
    p.add_argument("--local-geojson", type=Path, help="Use a manually downloaded GeoJSON instead of the REST download.")
    p.add_argument("--match-radius-m", type=float, default=25.0)
    p.add_argument("--retries", type=int, default=6, help="Retries to wake an idle ArcGIS service.")
    p.add_argument("--wait-s", type=float, default=15.0, help="Seconds between wake retries.")
    p.add_argument("--check-only", action="store_true", help="Report the crosswalk feature count and exit.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    if args.check_only and not args.local_geojson:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        ids = _get_json(session, f"{args.layer_url}/query", {"where": "1=1", "returnIdsOnly": "true"},
                        args.retries, args.wait_s)
        logging.info("OK -- %d marked crosswalks available (oid field=%s).",
                     len(ids.get("objectIds") or []), ids.get("objectIdFieldName"))
        return 0

    crosswalks = load_crosswalks_gdf(args)

    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "lat", "lon", "overture_crosswalk_present"])
    seattle = fla[fla["city"] == "seattle"].copy()
    seattle["point_id"] = seattle["point_id"].astype(int)
    logging.info("Seattle lock points: %d", len(seattle))

    labels = derive_crosswalk_labels(crosswalks, seattle, args.match_radius_m)
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_OUT, index=False)
    logging.info("Wrote labels: %s  (SDOT crosswalk prevalence=%.3f)",
                 LABELS_OUT, labels["sdot_crosswalk_present"].mean())

    merged = seattle.merge(labels, on=["point_id", "city"], how="inner")
    stats = agreement(
        merged["overture_crosswalk_present"].to_numpy(dtype=np.float32),
        merged["sdot_crosswalk_present"].to_numpy(dtype=np.float32),
    )
    report = {
        "match_radius_m": args.match_radius_m,
        "n_sdot_crosswalk_features": int(len(crosswalks)),
        "n_seattle_points": int(len(merged)),
        "overture_vs_sdot": stats,
        "prior_overture_vs_ps_seattle": {"kappa": 0.049, "auroc": 0.607, "ps_prevalence": 0.024},
        "interpretation": "If overture_vs_sdot kappa >> 0.049, the crosswalk label ceiling was a PS artifact.",
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("=" * 64)
    logging.info("Overture vs SDOT crosswalk: kappa=%s  AUROC=%s  n=%d",
                 stats["kappa"], stats.get("auroc_overture_vs_sdot"), stats["n"])
    logging.info("Prevalence: Overture=%s  SDOT=%s  (prior PS=0.024)",
                 stats.get("overture_prevalence"), stats.get("sdot_prevalence"))
    logging.info("Wrote audit: %s", REPORT_OUT)
    logging.info("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
