"""
Derive Overture Maps Tier 1 walkability targets for all locked points.

Uses vectorized geopandas sjoin (spatial index) — fast (~3–5 min for both cities).

Inputs:
    project/data/raw/overture/{city}/overture_segment_{city}.parquet
    project/data/raw/overture/{city}/overture_building_{city}.parquet
    project/data/processed/points/{city}_points_v2_final_lock.csv (both cities)
    project/data/processed/locks/v2_final_lock_ids.txt

Outputs:
    project/data/processed/labels/overture_targets.csv
    Columns: point_id, city, sidewalk_present, crosswalk_present,
             intersection_count_200m, footway_density_200m,
             building_footprint_frac_100m

Notes:
  - sidewalk/crosswalk: Overture segment class field
  - crosswalk fallback: OSMnx highway=crossing features when Overture has 0 crossings
  - intersection_count_200m: OSMnx walk network, degree >= 3 nodes
  - All spatial ops in EPSG:3857 (Web Mercator, meters)

Run:
    python project/scripts/derive_overture_targets.py --city both
    python project/scripts/derive_overture_targets.py --cities pittsburgh
    python project/scripts/derive_overture_targets.py --city msp --limit 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTS_DIR = REPO_ROOT / "project" / "data" / "processed" / "points"
OVERTURE_DIR = REPO_ROOT / "project" / "data" / "raw" / "overture"
LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"
LOCK_DIR  = REPO_ROOT / "project" / "data" / "processed" / "locks"
OUT_CSV   = REPO_ROOT / "project" / "data" / "processed" / "labels" / "overture_targets.csv"

SIDEWALK_BUFFER_M = 25
CROSSWALK_BUFFER_M = 50
INTERSECTION_BUFFER_M = 200
BUILDING_BUFFER_M = 100

SIDEWALK_CLASSES = {"footway", "sidewalk", "pedestrian", "path", "steps"}
CROSSWALK_CLASSES = {"marked"}  # only physically painted crosswalks; broad OSM tags excluded

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:3857"

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (west, south, east, north)
    "msp":         (-93.38, 44.84, -93.01, 45.11),
    "seattle":     (-122.48, 47.43, -122.19, 47.78),
    "dc":          (-77.1198, 38.7916, -76.9094, 38.9959),
    "pittsburgh":  (-80.10, 40.36, -79.85, 40.50),
}


def load_overture_segments(city: str) -> gpd.GeoDataFrame:
    path = OVERTURE_DIR / city / f"overture_segment_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Overture segment file not found: {path}\n"
            "Run: python project/scripts/download_overture_data.py --city both"
        )
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_WGS84)
    return gdf


def load_overture_buildings(city: str) -> gpd.GeoDataFrame:
    path = OVERTURE_DIR / city / f"overture_building_{city}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Overture building file not found: {path}\n"
            "Run: python project/scripts/download_overture_data.py --city both"
        )
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_WGS84)
    return gdf


def get_class_column(gdf: gpd.GeoDataFrame) -> str | None:
    for candidate in ("class", "subtype", "road_class", "highway"):
        if candidate in gdf.columns:
            return candidate
    return None


def classify_overture_segments(
    segments: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    class_col = get_class_column(segments)
    if class_col is None:
        logging.warning("No class column in Overture segments — returning empty sets.")
        empty = segments.iloc[:0].copy()
        return empty, empty

    classes = segments[class_col].fillna("").str.lower()
    sw_mask = classes.isin(SIDEWALK_CLASSES)
    cw_mask = classes.isin(CROSSWALK_CLASSES)
    logging.info(
        "  Segments total=%d  sidewalk=%d  crosswalk=%d  (col=%s)",
        len(segments), sw_mask.sum(), cw_mask.sum(), class_col,
    )
    logging.info(
        "  All class values: %s",
        dict(classes.value_counts().head(20)),
    )
    return segments[sw_mask].copy(), segments[cw_mask].copy()


def osm_network_data_from_bbox(
    city: str,
    bbox: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download OSMnx walk network; return (intersections, crossings) in EPSG:3857.

    Intersections: nodes with undirected degree >= 3.
    Crossings: nodes tagged highway=crossing from the graph, with fallback to
               features_from_bbox for cities where crossing tags are on point
               features rather than intersection nodes (e.g. DC OSM conventions).

    OSMnx 2.x note: graph_from_bbox raises ValueError("Some edges missing nodes")
    when the Overpass response includes way-references to nodes just outside the
    bbox that weren't fetched individually. Fix: download with 0.01° padding so
    all boundary-way nodes are present, then filter to original bbox post-hoc.
    """
    import osmnx as ox

    west, south, east, north = bbox
    # 0.01° ≈ 1 km — ensures way-referenced boundary nodes are fetched.
    pad = 0.01
    bbox_padded = (west - pad, south - pad, east + pad, north + pad)
    logging.info("  Downloading OSMnx walk network for %s (padded bbox)...", city)
    G = ox.graph_from_bbox(bbox=bbox_padded, network_type="walk", simplify=True)

    nodes, _ = ox.graph_to_gdfs(G)

    # Filter nodes to the original (unpadded) bbox before degree computation so
    # counts reflect the actual study area, not the padded buffer.
    nodes_clipped = nodes.cx[west:east, south:north].copy()

    G_undir = G.to_undirected()
    undirected_degrees = dict(G_undir.degree())
    nodes_clipped["undirected_degree"] = (
        nodes_clipped.index.map(undirected_degrees).fillna(0).astype(int)
    )
    intersections = nodes_clipped[nodes_clipped["undirected_degree"] >= 3][["geometry"]].to_crs(CRS_METRIC)
    logging.info(
        "  OSMnx nodes clipped=%d  intersections(degree>=3)=%d",
        len(nodes_clipped), len(intersections),
    )

    # Primary: highway=crossing graph nodes filtered to crossing=marked subtype.
    crossings: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[], crs=CRS_METRIC)
    if "highway" in nodes_clipped.columns:
        def _is_crossing(val: object) -> bool:
            if isinstance(val, list):
                return "crossing" in val
            return str(val) == "crossing" if pd.notna(val) else False

        crossing_mask = nodes_clipped["highway"].apply(_is_crossing)
        crossing_nodes = nodes_clipped[crossing_mask].copy()

        # Filter to crossing=marked when the attribute is present. Absent ≠ not-marked:
        # the fallback below handles nodes where OSM editors omitted the subtype tag.
        if "crossing" in crossing_nodes.columns:
            def _is_marked(val: object) -> bool:
                if isinstance(val, list):
                    return "marked" in val
                return str(val) == "marked" if pd.notna(val) else False
            marked_mask = crossing_nodes["crossing"].apply(_is_marked)
            crossings = crossing_nodes[marked_mask][["geometry"]].to_crs(CRS_METRIC)
            logging.info(
                "  Graph crossing=marked: %d (of %d highway=crossing)",
                len(crossings), len(crossing_nodes),
            )

    # Fallback: features_from_bbox with crossing=marked for cities where the
    # crossing subtype lives on point features rather than graph nodes (e.g., DC).
    if len(crossings) == 0:
        logging.info("  No marked crossings in graph nodes; querying features_from_bbox...")
        try:
            feats = ox.features_from_bbox(
                bbox=(west, south, east, north),
                tags={"crossing": "marked"},
            )
            pts = feats[feats.geom_type == "Point"][["geometry"]].copy()
            if len(pts) > 0:
                crossings = pts.set_crs(CRS_WGS84, allow_override=True).to_crs(CRS_METRIC)
                logging.info("  features_from_bbox crossing=marked fallback: %d points", len(crossings))
        except Exception as exc:
            logging.warning("  features_from_bbox crossing=marked query failed: %s", exc)

    logging.info("  OSMnx crossing nodes: %d", len(crossings))
    return intersections, crossings


def make_buffer_gdf(pts_metric: gpd.GeoDataFrame, radius_m: float) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of circular buffers around each point."""
    buf = pts_metric.copy()
    buf["geometry"] = pts_metric.geometry.buffer(radius_m)
    return buf


def sjoin_count(
    buffers: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    point_id_col: str = "point_id",
) -> pd.Series:
    """Count how many features intersect each buffer. Returns Series indexed by point_id."""
    if len(features) == 0:
        return pd.Series(0, index=buffers[point_id_col], dtype=int)
    features_idx = features[["geometry"]].reset_index(drop=True).copy()
    features_idx["_rid"] = np.arange(len(features_idx), dtype=int)
    joined = gpd.sjoin(
        buffers[[point_id_col, "geometry"]],
        features_idx[["_rid", "geometry"]],
        how="left",
        predicate="intersects",
    )
    counts = joined.groupby(point_id_col)["_rid"].count()
    return counts.reindex(buffers[point_id_col], fill_value=0)


def sjoin_any(
    buffers: gpd.GeoDataFrame,
    features: gpd.GeoDataFrame,
    point_id_col: str = "point_id",
) -> pd.Series:
    """Boolean: does any feature intersect each buffer? Returns Series indexed by point_id."""
    counts = sjoin_count(buffers, features, point_id_col)
    return (counts > 0).astype(int)


def compute_footway_density(
    pts_metric: gpd.GeoDataFrame,
    sidewalk_metric: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Total sidewalk length / buffer area for each point."""
    buf_area = np.pi * radius_m**2
    if len(sidewalk_metric) == 0:
        return pd.Series(0.0, index=pts_metric["point_id"])

    buffers = make_buffer_gdf(pts_metric, radius_m)
    sidewalk_idx = sidewalk_metric[["geometry"]].reset_index(drop=True).copy()
    sidewalk_idx["_rid"] = np.arange(len(sidewalk_idx), dtype=int)
    joined = gpd.sjoin(
        buffers[["point_id", "geometry"]],
        sidewalk_idx[["_rid", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # For each buffer-segment pair, compute the clipped segment length
    # Merge back the sidewalk geometries
    joined = joined.reset_index(drop=True)
    valid_mask = joined["_rid"].notna()
    if valid_mask.sum() == 0:
        return pd.Series(0.0, index=pts_metric["point_id"])

    buf_geom = buffers.set_index("point_id").loc[joined.loc[valid_mask, "point_id"], "geometry"].values
    seg_geom = sidewalk_idx.iloc[
        joined.loc[valid_mask, "_rid"].astype(int).values
    ]["geometry"].values

    lengths = np.array([
        b.intersection(s).length if b.intersects(s) else 0.0
        for b, s in zip(buf_geom, seg_geom)
    ])
    density_df = pd.DataFrame({
        "point_id": joined.loc[valid_mask, "point_id"].values,
        "length": lengths,
    })
    total_lengths = density_df.groupby("point_id")["length"].sum()
    density = total_lengths / buf_area
    return density.reindex(pts_metric["point_id"], fill_value=0.0)


def compute_building_frac(
    pts_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    radius_m: float,
) -> pd.Series:
    """Fraction of buffer area covered by building footprints."""
    buf_area = np.pi * radius_m**2
    if len(buildings_metric) == 0:
        return pd.Series(0.0, index=pts_metric["point_id"])

    buffers = make_buffer_gdf(pts_metric, radius_m)
    buildings_idx = buildings_metric[["geometry"]].reset_index(drop=True).copy()
    buildings_idx["_rid"] = np.arange(len(buildings_idx), dtype=int)
    joined = gpd.sjoin(
        buffers[["point_id", "geometry"]],
        buildings_idx[["_rid", "geometry"]],
        how="left",
        predicate="intersects",
    )
    valid_mask = joined["_rid"].notna()
    if valid_mask.sum() == 0:
        return pd.Series(0.0, index=pts_metric["point_id"])

    buf_geom = buffers.set_index("point_id").loc[joined.loc[valid_mask, "point_id"], "geometry"].values
    bld_geom = buildings_idx.iloc[
        joined.loc[valid_mask, "_rid"].astype(int).values
    ]["geometry"].values

    areas = np.array([
        b.intersection(s).area if b.intersects(s) else 0.0
        for b, s in zip(buf_geom, bld_geom)
    ])
    area_df = pd.DataFrame({
        "point_id": joined.loc[valid_mask, "point_id"].values,
        "area": areas,
    })
    total_areas = area_df.groupby("point_id")["area"].sum()
    frac = total_areas / buf_area
    return frac.reindex(pts_metric["point_id"], fill_value=0.0)


def process_city(
    city: str,
    lock_ids: set[int],
    limit: int | None,
) -> pd.DataFrame:
    logging.info("=== Processing %s ===", city.upper())

    # Accept both final-lock and raw points CSV (DC may not have final-lock yet)
    points_path = POINTS_DIR / f"{city}_points_v2_final_lock.csv"
    if not points_path.exists():
        points_path = POINTS_DIR / f"{city}_points_v2.csv"
    points = pd.read_csv(points_path)
    points["point_id"] = points["point_id"].astype(int)
    points = points[points["point_id"].isin(lock_ids)].reset_index(drop=True)
    if limit:
        points = points.head(limit)
    logging.info("  Points: %d", len(points))

    logging.info("  Loading Overture segments...")
    segments = load_overture_segments(city)
    logging.info("  Loading Overture buildings...")
    buildings = load_overture_buildings(city)

    sidewalk_segs, crosswalk_segs_overture = classify_overture_segments(segments)

    sidewalk_metric = sidewalk_segs.to_crs(CRS_METRIC) if len(sidewalk_segs) > 0 else sidewalk_segs
    buildings_metric = buildings.to_crs(CRS_METRIC)

    # OSMnx walk network: intersections + crossing nodes (single download)
    intersections_metric, osm_crossings = osm_network_data_from_bbox(city, CITY_BBOXES[city])

    if len(crosswalk_segs_overture) > 0:
        crosswalk_metric = crosswalk_segs_overture.to_crs(CRS_METRIC)
        crosswalk_source = "overture"
    elif len(osm_crossings) > 0:
        crosswalk_metric = osm_crossings
        crosswalk_source = "osm_nodes"
    else:
        crosswalk_metric = gpd.GeoDataFrame(geometry=[], crs=CRS_METRIC)
        crosswalk_source = "none"

    logging.info("  Crosswalk source: %s (%d features)", crosswalk_source, len(crosswalk_metric))

    # Build metric point GeoDataFrame
    pts_gdf = gpd.GeoDataFrame(
        points[["point_id", "lat", "lon"]].copy(),
        geometry=[Point(row.lon, row.lat) for _, row in points.iterrows()],
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    logging.info("  Running vectorized spatial joins...")

    # Sidewalk presence (25m buffer)
    sw_buffers = make_buffer_gdf(pts_gdf, SIDEWALK_BUFFER_M)
    sidewalk_present = sjoin_any(sw_buffers, sidewalk_metric)

    # Crosswalk presence (50m buffer)
    cw_buffers = make_buffer_gdf(pts_gdf, CROSSWALK_BUFFER_M)
    crosswalk_present = sjoin_any(cw_buffers, crosswalk_metric)

    # Intersection count (200m buffer)
    int_buffers = make_buffer_gdf(pts_gdf, INTERSECTION_BUFFER_M)
    intersection_count = sjoin_count(int_buffers, intersections_metric)

    # Footway density (200m buffer)
    logging.info("  Computing footway density...")
    footway_density = compute_footway_density(pts_gdf, sidewalk_metric, INTERSECTION_BUFFER_M)

    # Building footprint fraction (100m buffer)
    logging.info("  Computing building footprint fraction...")
    building_frac = compute_building_frac(pts_gdf, buildings_metric, BUILDING_BUFFER_M)

    result = pd.DataFrame({
        "point_id": pts_gdf["point_id"].values,
        "city": city,
        "sidewalk_present": sidewalk_present.values,
        "crosswalk_present": crosswalk_present.values,
        "crosswalk_source": crosswalk_source,
        "intersection_count_200m": intersection_count.values,
        "footway_density_200m": footway_density.values.round(6),
        "building_footprint_frac_100m": building_frac.values.round(4),
    })

    logging.info(
        "  Summary: sidewalk_present=%.1f%%  crosswalk_present=%.1f%%  "
        "mean_intersections=%.1f  mean_building_frac=%.3f",
        result["sidewalk_present"].mean() * 100,
        result["crosswalk_present"].mean() * 100,
        result["intersection_count_200m"].mean(),
        result["building_footprint_frac_100m"].mean(),
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive Overture Tier 1 targets for locked points")
    p.add_argument("--city", choices=["msp", "seattle", "dc", "pittsburgh", "both", "all"], default=None)
    p.add_argument("--cities", nargs="+", choices=["msp", "seattle", "dc", "pittsburgh"], default=None)
    p.add_argument("--limit", type=int, default=None, help="First N points only (testing)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output with only the requested cities instead of merging into existing rows.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in (LOCK_FILE,):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            return 1

    shared_lock_ids = {int(x) for x in LOCK_FILE.read_text().strip().splitlines()}

    if args.cities:
        cities = args.cities
    elif args.city in (None, "both", "all"):
        cities = ["msp", "seattle", "dc"]
    else:
        cities = [args.city]

    dfs: list[pd.DataFrame] = []
    for city in cities:
        # Use per-city lock file when available so each city uses its own complete set.
        # DC has 4,994 complete points; the shared MSP/Seattle lock only covers IDs 0–4823.
        per_city_lock = LOCK_DIR / f"{city}_v2_final_lock_ids.txt"
        if per_city_lock.exists():
            lock_ids = {int(x) for x in per_city_lock.read_text().strip().splitlines()}
            logging.info("Using per-city lock for %s: %d IDs", city, len(lock_ids))
        else:
            lock_ids = shared_lock_ids
        df = process_city(city, lock_ids, limit=args.limit)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if args.limit is None:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        if OUT_CSV.exists() and not args.overwrite:
            existing = pd.read_csv(OUT_CSV)
            existing = existing[~existing["city"].isin(cities)]
            combined = pd.concat([existing, combined], ignore_index=True)
            logging.info("Merged requested cities into existing output: %s", ", ".join(cities))
        combined.to_csv(OUT_CSV, index=False)
        logging.info("Saved: %s (%d rows)", OUT_CSV, len(combined))
    else:
        logging.info("[limit mode] Not saving — remove --limit to write full output")
        print(combined.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
