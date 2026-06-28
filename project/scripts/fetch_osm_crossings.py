"""
OSM pedestrian-crossing reference labels (fallback crosswalk ground truth).

Why
---
SDOT's marked-crosswalk service is down, and Project Sidewalk is not a crosswalk
source. OpenStreetMap maps pedestrian crossings as nodes (highway=crossing) and
ways (footway=crossing), including a "marked" subset. This is a real, human-mapped
crosswalk reference available for every city via Overpass. Caveat: Overture's
crosswalk layer partly derives from OSM, so OSM-vs-Overture agreement is not fully
independent -- but it is far better than Project Sidewalk (PS Seattle crosswalk
prevalence was 2.4%, i.e. essentially absent) for asking "does Overture crosswalk
capture real crossings, or was the label ceiling a PS artifact?".

What it does (per city)
-----------------------
1. Computes the city bbox from the lock points and queries Overpass for crossings.
2. Derives two per-point binary labels within --match-radius-m (default 25 m):
     osm_crossing_present          (any mapped crossing nearby)
     osm_marked_crossing_present   (crossing tagged marked / zebra / has markings)
3. Compares each against overture_crosswalk_present (Cohen's kappa, AUROC, prevalence)
   using the same agreement() logic as audit_label_noise.py.

No API key required.

Outputs
-------
  project/data/processed/labels/osm_crossing_labels.csv          (point_id, city, osm_crossing_present, osm_marked_crossing_present)
  project/artifacts/reports/osm_crossing_audit.json              (vs-Overture per city)

Run (Windows, repo ROOT)
------------------------
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_osm_crossings.py --cities seattle
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_osm_crossings.py            # all 4 cities
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
LABELS_OUT = REPO_ROOT / "project/data/processed/labels/osm_crossing_labels.csv"
REPORT_OUT = REPO_ROOT / "project/artifacts/reports/osm_crossing_audit.json"

CITY_UTM_EPSG = {"msp": 26915, "seattle": 32610, "dc": 32618, "pittsburgh": 32617}
BBOX_MARGIN_DEG = 0.005
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MARKED_VALUES = {"marked", "zebra", "tiger", "ladder", "lines", "dashes", "dots"}
USER_AGENT = "WalkBench-fetch-osm-crossings/1.0"


def query_overpass(bbox: tuple[float, float, float, float], url: str, retries: int) -> list[dict]:
    """bbox = (south, west, north, east). Returns elements with lat/lon + tags."""
    south, west, north, east = bbox
    q = (
        "[out:json][timeout:180];"
        f'(node["highway"="crossing"]({south},{west},{north},{east});'
        f'way["footway"="crossing"]({south},{west},{north},{east}););'
        "out tags center;"
    )
    last = ""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=240)
            if resp.status_code in (429, 504):
                raise requests.HTTPError(f"{resp.status_code} (Overpass busy)")
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as exc:
            last = str(exc)
            wait = min(60.0, 10.0 * attempt)
            logging.warning("Overpass attempt %d/%d failed (%s) -- wait %.0fs", attempt, retries, last[:80], wait)
            time.sleep(wait)
    raise RuntimeError(f"Overpass failed after {retries} tries: {last}")


def elements_to_points(elements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_crossings_lonlat, marked_crossings_lonlat) as Nx2 arrays."""
    all_ll, marked_ll = [], []
    for el in elements:
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if lat is None or lon is None:
            continue
        all_ll.append((lon, lat))
        tags = el.get("tags", {})
        markings = tags.get("crossing:markings", "")
        is_marked = (
            tags.get("crossing") in MARKED_VALUES
            or tags.get("crossing_ref") == "zebra"
            or (markings and markings != "no")
        )
        if is_marked:
            marked_ll.append((lon, lat))
    return (np.array(all_ll, dtype=float).reshape(-1, 2),
            np.array(marked_ll, dtype=float).reshape(-1, 2))


def present_within(points: pd.DataFrame, crossings_ll: np.ndarray, epsg: int, radius_m: float) -> np.ndarray:
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    if len(crossings_ll) == 0:
        return np.zeros(len(points), dtype=int)
    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    px, py = tf.transform(points["lon"].to_numpy(), points["lat"].to_numpy())
    cx, cy = tf.transform(crossings_ll[:, 0], crossings_ll[:, 1])
    tree = cKDTree(np.stack([cx, cy], axis=1))
    nearby = tree.query_ball_point(np.stack([px, py], axis=1), r=radius_m)
    return np.array([1 if idxs else 0 for idxs in nearby], dtype=int)


def agreement(reference: np.ndarray, derived: np.ndarray) -> dict:
    valid = np.isfinite(reference) & np.isfinite(derived)
    a, b = reference[valid].astype(int), derived[valid].astype(int)
    n = int(valid.sum())
    if n < 30 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return {"n": n, "kappa": float("nan"), "auroc_overture_vs_osm": float("nan"),
                "overture_prevalence": float(a.mean()) if n else float("nan"),
                "osm_prevalence": float(b.mean()) if n else float("nan"),
                "note": "degenerate (n<30 or single class)"}
    return {
        "n": n,
        "kappa": round(float(cohen_kappa_score(b, a)), 4),
        "auroc_overture_vs_osm": round(float(roc_auc_score(b, a)), 4),
        "raw_agreement": round(float((a == b).mean()), 4),
        "confusion_osm_rows_overture_cols": confusion_matrix(b, a, labels=[0, 1]).tolist(),
        "overture_prevalence": round(float(a.mean()), 4),
        "osm_prevalence": round(float(b.mean()), 4),
    }


def city_bbox(points: pd.DataFrame) -> tuple[float, float, float, float]:
    return (points["lat"].min() - BBOX_MARGIN_DEG, points["lon"].min() - BBOX_MARGIN_DEG,
            points["lat"].max() + BBOX_MARGIN_DEG, points["lon"].max() + BBOX_MARGIN_DEG)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OSM pedestrian-crossing reference labels, vs Overture crosswalk.")
    p.add_argument("--cities", nargs="+", default=list(CITY_UTM_EPSG), choices=list(CITY_UTM_EPSG))
    p.add_argument("--match-radius-m", type=float, default=25.0)
    p.add_argument("--overpass-url", default=OVERPASS_URL, help="Override (e.g. https://overpass.kumi.systems/api/interpreter).")
    p.add_argument("--retries", type=int, default=4)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "lat", "lon", "overture_crosswalk_present"])
    fla["point_id"] = fla["point_id"].astype(int)

    all_labels: list[pd.DataFrame] = []
    report: dict = {"match_radius_m": args.match_radius_m, "by_city": {}}
    for city in args.cities:
        pts = fla[fla["city"] == city].copy()
        if pts.empty:
            logging.warning("No points for %s -- skipping.", city)
            continue
        elements = query_overpass(city_bbox(pts), args.overpass_url, args.retries)
        all_ll, marked_ll = elements_to_points(elements)
        logging.info("[%s] OSM crossings: %d total, %d marked", city, len(all_ll), len(marked_ll))

        epsg = CITY_UTM_EPSG[city]
        any_pres = present_within(pts, all_ll, epsg, args.match_radius_m)
        marked_pres = present_within(pts, marked_ll, epsg, args.match_radius_m)

        labels = pts[["point_id", "city"]].copy()
        labels["osm_crossing_present"] = any_pres
        labels["osm_marked_crossing_present"] = marked_pres
        all_labels.append(labels)

        ovt = pts["overture_crosswalk_present"].to_numpy(dtype=np.float32)
        report["by_city"][city] = {
            "n_osm_crossings": int(len(all_ll)),
            "n_osm_marked": int(len(marked_ll)),
            "overture_vs_osm_any": agreement(ovt, any_pres.astype(np.float32)),
            "overture_vs_osm_marked": agreement(ovt, marked_pres.astype(np.float32)),
        }
        m = report["by_city"][city]["overture_vs_osm_marked"]
        logging.info("[%s] Overture vs OSM-marked crosswalk: kappa=%s AUROC=%s  (Overture prev=%s, OSM-marked prev=%s)",
                     city, m["kappa"], m.get("auroc_overture_vs_osm"),
                     m.get("overture_prevalence"), m.get("osm_prevalence"))

    if all_labels:
        pd.concat(all_labels, ignore_index=True).to_csv(LABELS_OUT, index=False)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote labels: %s  and audit: %s", LABELS_OUT, REPORT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
