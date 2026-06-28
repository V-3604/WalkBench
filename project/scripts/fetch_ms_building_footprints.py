"""
Independent building-footprint labels from Microsoft US Building Footprints.

Why
---
overture_building_footprint_frac_100m is the strongest imagery target (cross-city
rho 0.82). This script recomputes the same quantity from a fully independent source
(Microsoft's 129M computer-generated US footprints, ODbL) so we can confirm the
target is not an Overture artifact: high Spearman rho between the two means the
label is real, not source-specific.

What it does (per city)
-----------------------
1. Downloads the relevant US state footprints zip (public Azure blob) -- or uses a
   manually downloaded zip via --local-zip-dir.
2. Streams it with a bounding-box filter around the city's lock points (no full
   extraction, bounded memory).
3. For each lock point: buffers 100 m in the city UTM and computes the fraction of
   the buffer covered by building polygons (union-clipped), matching the Overture
   target definition.
4. Compares ms_building_footprint_frac_100m vs overture_building_footprint_frac_100m
   (Spearman rho, MAE) per city.

No API key required (public Azure blob / ODbL).

State files are large (DC ~tens of MB; MN/WA/PA ~hundreds of MB each; ~1-1.5 GB
total). Use --cities to do one at a time and --limit to smoke-test.

Outputs
-------
  project/data/external/ms_buildings/{State}.geojson.zip                (raw, cached)
  project/data/processed/labels/ms_building_footprint.csv              (point_id, city, ms_building_footprint_frac_100m)
  project/artifacts/reports/ms_building_footprint_audit.json           (vs-Overture per city)

Run (Windows, repo root)
------------------------
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_ms_building_footprints.py --cities dc --limit 200   # smoke test
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_ms_building_footprints.py                            # all 4 cities
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
RAW_DIR = REPO_ROOT / "project/data/external/ms_buildings"
LABELS_OUT = REPO_ROOT / "project/data/processed/labels/ms_building_footprint.csv"
REPORT_OUT = REPO_ROOT / "project/artifacts/reports/ms_building_footprint_audit.json"

CITY_UTM_EPSG = {"msp": 26915, "seattle": 32610, "dc": 32618, "pittsburgh": 32617}
CITY_STATE = {"msp": "Minnesota", "seattle": "Washington", "dc": "DistrictofColumbia", "pittsburgh": "Pennsylvania"}
BLOB_TEMPLATE = "https://usbuildingdata.blob.core.windows.net/usbuildings-v2/{state}.geojson.zip"
BUFFER_M = 100.0
BBOX_MARGIN_DEG = 0.003  # ~330 m padding so 100 m buffers near the city edge are fully covered
USER_AGENT = "WalkBench-fetch-ms-buildings/1.0"


def download_state_zip(state: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        logging.info("Cached: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return
    url = BLOB_TEMPLATE.format(state=state)
    logging.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600, headers={"User-Agent": USER_AGENT}) as resp:
        if resp.status_code == 404:
            raise FileNotFoundError(
                f"404 for {url}. The download URL pattern may have changed -- check the current "
                f"links at https://github.com/microsoft/USBuildingFootprints and use --local-zip-dir."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:6.1f} / {total/1e6:6.1f} MB", end="", flush=True)
        print()
    logging.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def city_bbox(points: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        points["lon"].min() - BBOX_MARGIN_DEG, points["lat"].min() - BBOX_MARGIN_DEG,
        points["lon"].max() + BBOX_MARGIN_DEG, points["lat"].max() + BBOX_MARGIN_DEG,
    )


def load_buildings_bbox(zip_path: Path, bbox: tuple[float, float, float, float]):
    import geopandas as gpd

    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".geojson")]
    if not members:
        raise ValueError(f"No .geojson member inside {zip_path}.")
    vsizip = f"/vsizip/{zip_path.as_posix()}/{members[0]}"
    logging.info("Reading %s with bbox filter %s", members[0], tuple(round(v, 4) for v in bbox))
    gdf = gpd.read_file(vsizip, bbox=bbox)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    logging.info("Buildings in bbox: %d", len(gdf))
    return gdf


def compute_fraction(points: pd.DataFrame, buildings, city: str, limit: int | None) -> pd.DataFrame:
    import geopandas as gpd
    from shapely.ops import unary_union

    epsg = CITY_UTM_EPSG[city]
    buildings = buildings.to_crs(epsg=epsg)
    pts = gpd.GeoDataFrame(
        points.copy(), geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=epsg)
    if limit:
        pts = pts.iloc[:limit].copy()

    sindex = buildings.sindex
    buffer_area = np.pi * BUFFER_M ** 2
    fracs: list[float] = []
    for geom in tqdm(pts.geometry, desc=f"{city} 100m buffers", unit="pt"):
        buf = geom.buffer(BUFFER_M)
        cand_idx = list(sindex.query(buf, predicate="intersects"))
        if not cand_idx:
            fracs.append(0.0)
            continue
        covered = unary_union(buildings.geometry.iloc[cand_idx].tolist()).intersection(buf).area
        fracs.append(float(min(covered / buffer_area, 1.0)))

    out = pts[["point_id", "city"]].copy()
    out["ms_building_footprint_frac_100m"] = fracs
    return out


def compare(merged: pd.DataFrame) -> dict:
    o = merged["overture_building_footprint_frac_100m"].to_numpy(dtype=float)
    m = merged["ms_building_footprint_frac_100m"].to_numpy(dtype=float)
    valid = np.isfinite(o) & np.isfinite(m)
    o, m = o[valid], m[valid]
    return {
        "n": int(valid.sum()),
        "spearman_rho": round(float(scipy.stats.spearmanr(o, m)[0]), 4),
        "pearson_r": round(float(scipy.stats.pearsonr(o, m)[0]), 4),
        "mae": round(float(np.mean(np.abs(o - m))), 4),
        "overture_mean": round(float(o.mean()), 4),
        "ms_mean": round(float(m.mean()), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microsoft US Building Footprints -> 100m frac, vs Overture.")
    p.add_argument("--cities", nargs="+", default=list(CITY_STATE), choices=list(CITY_STATE))
    p.add_argument("--local-zip-dir", type=Path, help="Directory holding manually downloaded {State}.geojson.zip files.")
    p.add_argument("--limit", type=int, help="Only process the first N points per city (smoke test).")
    p.add_argument("--dry-run", action="store_true", help="Print the download plan and exit.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "lat", "lon", "overture_building_footprint_frac_100m"])
    fla["point_id"] = fla["point_id"].astype(int)

    if args.dry_run:
        for city in args.cities:
            zip_name = f"{CITY_STATE[city]}.geojson.zip"
            src = (args.local_zip_dir / zip_name) if args.local_zip_dir else BLOB_TEMPLATE.format(state=CITY_STATE[city])
            logging.info("%-11s state=%-17s points=%d  source=%s",
                         city, CITY_STATE[city], int((fla["city"] == city).sum()), src)
        return 0

    per_city_labels: list[pd.DataFrame] = []
    report: dict = {"buffer_m": BUFFER_M, "by_city": {}}
    for city in args.cities:
        state = CITY_STATE[city]
        zip_path = (args.local_zip_dir / f"{state}.geojson.zip") if args.local_zip_dir else RAW_DIR / f"{state}.geojson.zip"
        if args.local_zip_dir and not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} not found in --local-zip-dir.")
        if not args.local_zip_dir:
            download_state_zip(state, zip_path)

        pts = fla[fla["city"] == city].copy()
        buildings = load_buildings_bbox(zip_path, city_bbox(pts))
        labels = compute_fraction(pts, buildings, city, args.limit)
        per_city_labels.append(labels)

        merged = pts.merge(labels, on=["point_id", "city"], how="inner")
        stats = compare(merged)
        report["by_city"][city] = stats
        logging.info("[%s] ms-vs-overture building frac: rho=%.3f  MAE=%.3f  (n=%d)",
                     city, stats["spearman_rho"], stats["mae"], stats["n"])

    all_labels = pd.concat(per_city_labels, ignore_index=True)
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    all_labels.to_csv(LABELS_OUT, index=False)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote labels: %s  and audit: %s", LABELS_OUT, REPORT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
