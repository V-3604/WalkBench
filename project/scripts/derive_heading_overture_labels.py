"""
Derive per-(point_id, heading) Overture walkability labels for WalkCLIP v2.

For each point × heading pair, builds a 60-degree wedge from the point in
the heading direction and intersects it with Overture footway/crosswalk
geometry. This enables heading-conditioned supervision in the trainer.

Wedge geometry:
  - Sidewalk: radius 15 m, heading ± 30°
  - Crosswalk: radius 15 m, heading ± 30°
  - Coordinate system: EPSG:3857 (Web Mercator, metres)

Outputs:
  project/data/processed/labels/heading_overture_labels_v2.csv
  Columns: point_id, city, heading, sidewalk_present, crosswalk_present

Run:
    & ".venv\Scripts\python.exe" project/scripts/derive_heading_overture_labels.py --version v2 --cities msp seattle dc
    & ".venv\Scripts\python.exe" project/scripts/derive_heading_overture_labels.py --version v2 --cities pittsburgh
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = REPO_ROOT / "project/data/processed/locks"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
OVERTURE_DIR = REPO_ROOT / "project/data/raw/overture"
OUT_CSV = REPO_ROOT / "project/data/processed/labels/heading_overture_labels_v2.csv"

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

TARGET_HEADINGS = (0, 90, 180, 270)
SIDEWALK_RADIUS_M = 15.0
CROSSWALK_RADIUS_M = 15.0
WEDGE_HALF_ANGLE_DEG = 30.0
WEDGE_ARC_POINTS = 32  # polygon resolution

SIDEWALK_CLASSES = frozenset({"footway", "sidewalk", "pedestrian", "path", "steps"})
CROSSWALK_CLASSES = frozenset({"marked"})  # only physically painted crosswalks; broad OSM tags excluded

CITY_LOCK_FILES: dict[str, str] = {
    "msp":         "msp_v2_final_lock_ids.txt",
    "seattle":     "seattle_v2_final_lock_ids.txt",
    "dc":          "dc_v2_final_lock_ids.txt",
    "pittsburgh":  "pittsburgh_v2_final_lock_ids.txt",
}
CITY_LOCK_FALLBACK = "v2_final_lock_ids.txt"
CITY_POINTS_CSV: dict[str, Path] = {
    "msp":         POINTS_DIR / "msp_points_v2_final_lock.csv",
    "seattle":     POINTS_DIR / "seattle_points_v2_final_lock.csv",
    "dc":          POINTS_DIR / "dc_points_v2_final_lock.csv",
    "pittsburgh":  POINTS_DIR / "pittsburgh_points_v2_final_lock.csv",
}


def load_lock_ids(city: str) -> set[int]:
    primary = LOCKS_DIR / CITY_LOCK_FILES[city]
    fallback = LOCKS_DIR / CITY_LOCK_FALLBACK
    for p in [primary, fallback]:
        if p.exists():
            return {int(x) for x in p.read_text().strip().splitlines()}
    raise FileNotFoundError(f"No lock file for {city}. Run build_v2_final_lock.py first.")


def build_wedge(cx: float, cy: float, heading_deg: float, radius_m: float) -> Polygon:
    """Pie-slice wedge polygon centered at (cx, cy) in EPSG:3857 metres.

    heading_deg: degrees clockwise from North.
    In EPSG:3857 East=+x, North=+y, so:
      x_offset = sin(angle), y_offset = cos(angle)
    """
    start_rad = math.radians(heading_deg - WEDGE_HALF_ANGLE_DEG)
    end_rad = math.radians(heading_deg + WEDGE_HALF_ANGLE_DEG)
    angles = np.linspace(start_rad, end_rad, WEDGE_ARC_POINTS)
    arc = [
        (cx + radius_m * math.sin(a), cy + radius_m * math.cos(a))
        for a in angles
    ]
    coords = [(cx, cy)] + arc + [(cx, cy)]
    return Polygon(coords)


def load_overture_segments(city: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (sidewalk_gdf, crosswalk_gdf) in EPSG:3857."""
    seg_path = OVERTURE_DIR / city / f"overture_segment_{city}.parquet"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Overture segments not found: {seg_path}\n"
            "Run: python project/scripts/download_overture_data.py --city {city}"
        )
    gdf = gpd.read_parquet(seg_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_WGS84)
    gdf = gdf.to_crs(CRS_METRIC)

    class_col = next(
        (c for c in ("class", "subtype", "road_class", "highway") if c in gdf.columns), None
    )
    if class_col is None:
        logging.warning("%s: no class column in Overture segments — all labels will be 0", city)
        empty = gdf.iloc[:0].copy()
        return empty, empty

    classes = gdf[class_col].fillna("").str.lower()
    sw_gdf = gdf[classes.isin(SIDEWALK_CLASSES)].copy()
    cw_gdf = gdf[classes.isin(CROSSWALK_CLASSES)].copy()
    logging.info("%s: sidewalk=%d  crosswalk=%d", city, len(sw_gdf), len(cw_gdf))
    return sw_gdf, cw_gdf


CITY_BBOX: dict[str, tuple[float, float, float, float]] = {
    # (west, south, east, north)
    "msp":         (-93.38, 44.84, -93.01, 45.11),
    "seattle":     (-122.48, 47.43, -122.19, 47.78),
    "dc":          (-77.1198, 38.7916, -76.9094, 38.9959),
    "pittsburgh":  (-80.10, 40.36, -79.85, 40.50),
}


def load_osm_crossing_nodes(city: str) -> gpd.GeoDataFrame:
    """Return OSM crossing=marked point features in EPSG:3857.

    Uses crossing=marked (physically painted crosswalks) to align with
    Project Sidewalk's 'Crosswalk' label, which requires a visible painted
    marking. Generic highway=crossing tags (uncontrolled, traffic_signals,
    bare crossing) are excluded because they fire at ~10× PS prevalence.

    Primary: features_from_bbox(tags={"crossing": "marked"}) — direct query,
    faster than downloading the full walk graph.
    Fallback: graph-based highway=crossing nodes filtered to crossing=marked
    attribute, for cities where OSM editors put the subtype on graph nodes
    rather than as standalone point features.
    """
    west, south, east, north = CITY_BBOX[city]
    pad = 0.01
    logging.info("%s: querying OSM crossing=marked features...", city)
    ox.settings.log_console = False

    crossings: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[], crs=CRS_METRIC)
    try:
        feats = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={"crossing": "marked"},
        )
        pts = feats[feats.geom_type == "Point"][["geometry"]].copy()
        if len(pts) > 0:
            crossings = pts.set_crs(CRS_WGS84, allow_override=True).to_crs(CRS_METRIC)
    except Exception as exc:
        logging.warning("%s: features_from_bbox crossing=marked query failed: %s", city, exc)

    if len(crossings) == 0:
        logging.info(
            "%s: no crossing=marked point features; trying graph-based fallback...", city
        )
        bbox_padded = (west - pad, south - pad, east + pad, north + pad)
        try:
            G = ox.graph_from_bbox(bbox=bbox_padded, network_type="walk", simplify=True)
            nodes, _ = ox.graph_to_gdfs(G)
            nodes_clipped = nodes.cx[west:east, south:north].copy()
            if "highway" in nodes_clipped.columns:
                def _is_crossing(val: object) -> bool:
                    if isinstance(val, list):
                        return "crossing" in val
                    return str(val) == "crossing" if pd.notna(val) else False
                crossing_mask = nodes_clipped["highway"].apply(_is_crossing)
                crossing_nodes = nodes_clipped[crossing_mask].copy()
                if "crossing" in crossing_nodes.columns:
                    def _is_marked(val: object) -> bool:
                        if isinstance(val, list):
                            return "marked" in val
                        return str(val) == "marked" if pd.notna(val) else False
                    marked_mask = crossing_nodes["crossing"].apply(_is_marked)
                    crossings = crossing_nodes[marked_mask][["geometry"]].to_crs(CRS_METRIC)
        except Exception as exc:
            logging.warning(
                "%s: graph-based crossing=marked fallback failed: %s", city, exc
            )

    logging.info("%s: OSM crossing=marked nodes found: %d", city, len(crossings))
    return crossings


def process_city(
    city: str,
    limit: int | None,
) -> pd.DataFrame:
    logging.info("=== %s ===", city.upper())

    lock_ids = load_lock_ids(city)
    points_csv = CITY_POINTS_CSV[city]
    if not points_csv.exists():
        raise FileNotFoundError(f"Points CSV not found: {points_csv}")

    pts = pd.read_csv(points_csv)
    pts["point_id"] = pts["point_id"].astype(int)
    pts = pts[pts["point_id"].isin(lock_ids)].reset_index(drop=True)
    if limit:
        pts = pts.head(limit)
    logging.info("Points: %d", len(pts))

    sw_gdf, cw_gdf = load_overture_segments(city)
    sw_sindex = sw_gdf.sindex if len(sw_gdf) > 0 else None

    # Overture segments have no crosswalk class values; use OSM crossing nodes instead
    crossing_gdf = load_osm_crossing_nodes(city)
    cw_gdf = crossing_gdf  # point GDF, one row per crossing node
    cw_sindex = cw_gdf.sindex if len(cw_gdf) > 0 else None

    # Project points to metric CRS
    pts_gdf = gpd.GeoDataFrame(
        pts[["point_id", "lat", "lon"]].copy(),
        geometry=[Point(row.lon, row.lat) for _, row in pts.iterrows()],
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)
    pts_gdf["x"] = pts_gdf.geometry.x
    pts_gdf["y"] = pts_gdf.geometry.y

    records: list[dict] = []
    for _, row in pts_gdf.iterrows():
        cx, cy, pid = float(row["x"]), float(row["y"]), int(row["point_id"])
        for heading in TARGET_HEADINGS:
            sw_wedge = build_wedge(cx, cy, heading, SIDEWALK_RADIUS_M)
            cw_wedge = build_wedge(cx, cy, heading, CROSSWALK_RADIUS_M)

            # Sidewalk: check any intersection
            sw_present = 0
            if sw_sindex is not None and len(sw_gdf) > 0:
                candidates = list(sw_sindex.intersection(sw_wedge.bounds))
                if candidates:
                    subset = sw_gdf.iloc[candidates]
                    sw_present = int(subset.intersects(sw_wedge).any())

            # Crosswalk: check any intersection
            cw_present = 0
            if cw_sindex is not None and len(cw_gdf) > 0:
                candidates = list(cw_sindex.intersection(cw_wedge.bounds))
                if candidates:
                    subset = cw_gdf.iloc[candidates]
                    cw_present = int(subset.intersects(cw_wedge).any())

            records.append({
                "point_id": pid,
                "city": city,
                "heading": heading,
                "sidewalk_present": sw_present,
                "crosswalk_present": cw_present,
            })

    df = pd.DataFrame(records)
    logging.info(
        "%s heading labels: %d rows  sw_present=%.1f%%  cw_present=%.1f%%",
        city, len(df),
        df["sidewalk_present"].mean() * 100,
        df["crosswalk_present"].mean() * 100,
    )
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Derive per-(point_id,heading) Overture wedge labels")
    ap.add_argument("--version", choices=["v2"], default="v2")
    ap.add_argument("--cities", nargs="+", choices=["msp", "seattle", "dc", "pittsburgh"],
                    default=["msp", "seattle", "dc"])
    ap.add_argument("--output", default=str(OUT_CSV))
    ap.add_argument("--limit", type=int, default=None, help="First N points per city (testing)")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output with only the requested cities instead of merging into existing rows.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    out_path = Path(args.output)
    dfs: list[pd.DataFrame] = []
    for city in args.cities:
        df = process_city(city, limit=args.limit)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if args.limit is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            existing = pd.read_csv(out_path)
            existing = existing[~existing["city"].isin(args.cities)]
            combined = pd.concat([existing, combined], ignore_index=True)
            logging.info("Merged requested cities into existing output: %s", ", ".join(args.cities))
        combined.to_csv(out_path, index=False)
        logging.info("Saved: %s (%d rows)", out_path, len(combined))
    else:
        logging.info("[limit mode] not saving — remove --limit to write full output")
        print(combined.head(20).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
