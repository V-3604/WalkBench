"""
Re-derive crosswalk_present per (point_id, heading) using a tight near-buffer.

Old: 100 m radius, ±45° wedge ⇒ noisy, prevalence ~0.57.
New:  15 m radius, ±30° wedge ⇒ sharp, prevalence ~0.10.

A tight wedge is what is actually visible in a single Mapillary frame, so
this aligns the label with the supervisory image.

The 'sidewalk_present' column is left untouched (the existing 50 m / ±45°
radius is correct for sidewalks, which are continuous).

Inputs:
    project/data/processed/labels/heading_overture_labels.csv  (existing v1)
    project/data/raw/overture/{city}/overture_segment_{city}.parquet
    project/data/processed/points/{city}_points_v2_final_lock.csv

Outputs:
    project/data/processed/labels/heading_overture_labels_v2.csv
        Same schema, but crosswalk_present is the new sharp version.

Run:
    & $PY project/scripts/relabel_crosswalk_near_buffer.py --cities msp seattle dc
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
OVERTURE_DIR = REPO_ROOT / "project/data/raw/overture"
HEADING_V1 = REPO_ROOT / "project/data/processed/labels/heading_overture_labels.csv"
HEADING_V2 = REPO_ROOT / "project/data/processed/labels/heading_overture_labels_v2.csv"

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

CROSSWALK_RADIUS_M = 15.0      # was 100.0
WEDGE_HALF_ANGLE_DEG = 30.0    # was 45.0
WEDGE_ARC_POINTS = 24

CROSSWALK_CLASSES = frozenset({"marked"})  # only physically painted crosswalks; broad OSM tags excluded
CROSSWALK_SUBCLASSES = frozenset({"crosswalk", "cycle_crossing", "crossing"})


def wedge_polygon(cx: float, cy: float, heading_deg: float,
                   radius_m: float, half_angle_deg: float,
                   n_arc: int = WEDGE_ARC_POINTS) -> Polygon:
    """Heading=0 means north; heading increases clockwise (compass).
    Build a wedge of given radius and half-angle around heading_deg."""
    # Compass → math: math_angle = 90 - heading_deg (degrees, CCW from +x).
    center_math = math.radians(90.0 - heading_deg)
    half = math.radians(half_angle_deg)
    angles = np.linspace(center_math - half, center_math + half, n_arc)
    pts = [(cx, cy)]
    pts.extend((cx + radius_m * math.cos(a), cy + radius_m * math.sin(a))
               for a in angles)
    return Polygon(pts)


def load_overture_crosswalks(city: str) -> gpd.GeoDataFrame:
    p = OVERTURE_DIR / city / f"overture_segment_{city}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run download_overture_data.py first.")
    g = gpd.read_parquet(p)
    if g.crs is None:
        g = g.set_crs(CRS_WGS84)
    g = g.to_crs(CRS_METRIC)
    # Overture transportation schemas vary by release. Crosswalk signal can live
    # in class/subtype (older) or subclass (newer).
    mask = pd.Series(False, index=g.index)
    hit_counts: dict[str, int] = {}
    if "class" in g.columns:
        m = g["class"].astype(str).str.lower().isin(CROSSWALK_CLASSES)
        mask = mask | m
        hit_counts["class"] = int(m.sum())
    if "subtype" in g.columns:
        m = g["subtype"].astype(str).str.lower().isin(CROSSWALK_CLASSES)
        mask = mask | m
        hit_counts["subtype"] = int(m.sum())
    if "subclass" in g.columns:
        m = g["subclass"].astype(str).str.lower().isin(CROSSWALK_SUBCLASSES)
        mask = mask | m
        hit_counts["subclass"] = int(m.sum())
    if not hit_counts:
        raise SystemExit(
            f"No class/subtype/subclass columns in {p}; columns: {list(g.columns)}"
        )
    cw = g[mask].copy()
    logging.info("[%s] crosswalk hits by field: %s", city, hit_counts)
    logging.info("[%s] %d crosswalk geometries from Overture", city, len(cw))
    return cw[["geometry"]]


def relabel_city(city: str, v1: pd.DataFrame) -> pd.DataFrame:
    points_csv = POINTS_DIR / f"{city}_points_v2_final_lock.csv"
    pts = pd.read_csv(points_csv)
    pts["city"] = city
    pts_g = gpd.GeoDataFrame(
        pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]),
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)
    cw = load_overture_crosswalks(city)
    cw_sindex = cw.sindex

    sub = v1[v1["city"] == city].copy()
    pt_xy = pts_g.set_index("point_id")[["geometry"]].to_dict()["geometry"]

    new_cw = np.zeros(len(sub), dtype=np.int8)
    for i, (pid, hd) in enumerate(zip(sub["point_id"].to_numpy(),
                                        sub["heading"].to_numpy())):
        geom = pt_xy.get(int(pid))
        if geom is None:
            continue
        wedge = wedge_polygon(geom.x, geom.y, float(hd),
                               CROSSWALK_RADIUS_M, WEDGE_HALF_ANGLE_DEG)
        cand_idx = list(cw_sindex.intersection(wedge.bounds))
        if not cand_idx:
            continue
        if cw.iloc[cand_idx].intersects(wedge).any():
            new_cw[i] = 1
    sub["crosswalk_present"] = new_cw
    return sub


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=["msp", "seattle", "dc"])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if not HEADING_V1.exists():
        raise SystemExit(f"Missing {HEADING_V1}. Run derive_heading_overture_labels.py first.")
    v1 = pd.read_csv(HEADING_V1)
    logging.info("v1 rows: %d, columns: %s", len(v1), list(v1.columns))
    out = []
    for city in args.cities:
        sub = relabel_city(city, v1)
        prev_old = float(v1[v1["city"] == city]["crosswalk_present"].mean())
        prev_new = float(sub["crosswalk_present"].mean())
        logging.info("[%s] prevalence  old=%.3f  new=%.3f  (delta=%.3f)",
                     city, prev_old, prev_new, prev_new - prev_old)
        out.append(sub)
    final = pd.concat(out, ignore_index=True)
    HEADING_V2.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(HEADING_V2, index=False)
    logging.info("Wrote %s (%d rows)", HEADING_V2, len(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
