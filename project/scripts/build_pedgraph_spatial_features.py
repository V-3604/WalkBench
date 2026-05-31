"""
Build Overture pedestrian-graph kNN spatial context features for WalkCLIP v2.

Replaces Euclidean kNN with walk-network graph-distance kNN:
  - Download OSMnx walk graph for each city (projected to UTM)
  - Each sample point's osmid corresponds to a graph node (points were
    selected at street intersections, so osmid is in the graph)
  - For each point, run single_source_dijkstra_path_length(cutoff=400m)
  - Find the k=8 nearest OTHER sample points by graph distance
  - Aggregate their observable features (inverse-distance weighted)

Output schema matches spatial_context_features.csv so --spatial-features pedgraph
can hot-swap the file in the trainer.

Outputs:
  project/data/processed/spatial_context/pedgraph_features.csv

Run:
    & ".venv\Scripts\python.exe" project/scripts/build_pedgraph_spatial_features.py ^
        --cities msp seattle dc --k 8 --max-graph-distance-m 400
    & ".venv\Scripts\python.exe" project/scripts/build_pedgraph_spatial_features.py ^
        --cities dc --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = REPO_ROOT / "project/data/processed/locks"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
CAPTION_CSV = REPO_ROOT / "project/data/processed/text/caption_features_final_lock.csv"
OUT_DIR = REPO_ROOT / "project/data/processed/spatial_context"
OUT_CSV = OUT_DIR / "pedgraph_features.csv"
OUT_JSON = OUT_DIR / "pedgraph_features_summary.json"

K_NEIGHBORS = 8
MAX_GRAPH_DISTANCE_M = 400.0

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "msp":         (-93.38, 44.84, -93.01, 45.11),
    "seattle":     (-122.48, 47.43, -122.19, 47.78),
    "dc":          (-77.1198, 38.7916, -76.9094, 38.9959),
    "pittsburgh":  (-80.10, 40.36, -79.85, 40.50),
}
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

# Features aggregated from neighbors — must match BASE_COLS in build_spatial_context_features.py
BASE_COLS = [
    "sidewalk_frac_mean", "crosswalk_frac_mean", "vegetation_frac_mean",
    "sky_frac_mean", "building_frac_mean", "sidewalk_width_px",
    "street_mask_coverage_mean", "agreement_score",
    "cap_sidewalk_ord", "cap_crosswalk_ord", "cap_traffic_volume_ord",
    "cap_tree_canopy_ord", "cap_transit_stop_ord", "cap_lighting_ord",
    "cap_maintenance_ord", "cap_unknown_count",
]


def load_lock_ids(city: str) -> set[int]:
    primary = LOCKS_DIR / CITY_LOCK_FILES[city]
    fallback = LOCKS_DIR / CITY_LOCK_FALLBACK
    for p in [primary, fallback]:
        if p.exists():
            return {int(x) for x in p.read_text().strip().splitlines()}
    raise FileNotFoundError(f"No lock file for {city}. Run build_v2_final_lock.py first.")


def load_city_features(city: str, lock_ids: set[int]) -> pd.DataFrame:
    """Load merged labels + captions for a city, filling missing columns with 0."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    labels["point_id"] = labels["point_id"].astype(int)
    labels = labels[labels["city"] == city].copy()
    labels = labels[labels["point_id"].isin(lock_ids)]

    if CAPTION_CSV.exists():
        caps = pd.read_csv(CAPTION_CSV)
        caps["point_id"] = caps["point_id"].astype(int)
        labels = labels.merge(caps, on=["point_id", "city"], how="left")

    for col in BASE_COLS:
        if col not in labels.columns:
            labels[col] = 0.0
        else:
            median = float(labels[col].median()) if labels[col].notna().any() else 0.0
            labels[col] = labels[col].fillna(median)

    return labels.set_index("point_id")


def build_projected_graph(city: str) -> nx.MultiDiGraph:
    """Download and project OSMnx walk graph for the city bbox."""
    bbox = CITY_BBOXES[city]
    logging.info("[%s] Downloading OSMnx walk graph …", city)
    west, south, east, north = bbox
    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type="walk")
    except TypeError:
        G = ox.graph_from_bbox(north, south, east, west, network_type="walk")
    G = ox.project_graph(G)  # UTM, adds 'length' in metres to edges
    logging.info("[%s] Graph: %d nodes, %d edges", city, G.number_of_nodes(), G.number_of_edges())
    return G


def snap_to_graph(
    G: nx.MultiDiGraph,
    osmid: int,
    lat: float | None = None,
    lon: float | None = None,
) -> tuple[int, bool]:
    """Return (graph_node_id, was_fallback). If ``osmid`` is in the graph, returns it
    directly. Otherwise falls back to the geographically nearest node via
    ``ox.distance.nearest_nodes`` (requires lon/lat — the projected graph carries
    metric coordinates, but ``nearest_nodes`` accepts unprojected lon/lat with the
    underlying graph's CRS handled by OSMnx).

    The pre-2026-04-26 implementation returned ``next(iter(G.nodes()))`` — an
    *arbitrary* node — when the osmid was missing. All such points then anchored to
    the same arbitrary node and produced spurious kNN aggregates. The audit script
    ``audit_pedgraph_osmid_coverage.py`` reports per-city fallback counts so the
    locked features file can be re-checked.
    """
    if G.has_node(osmid):
        return int(osmid), False
    if lat is None or lon is None:
        # No coords available; signal fallback but return a stable node so callers
        # can decide to skip. We return -1 to make missing coords explicit.
        return -1, True
    try:
        nearest = ox.distance.nearest_nodes(G, X=float(lon), Y=float(lat))
        return int(nearest), True
    except Exception as exc:
        logging.warning("nearest_nodes failed for (%.6f,%.6f): %s", lat, lon, exc)
        return -1, True


def build_city_pedgraph_features(
    city: str,
    k: int,
    max_graph_dist_m: float,
    limit: int | None,
) -> pd.DataFrame:
    lock_ids = load_lock_ids(city)
    features_df = load_city_features(city, lock_ids)
    base_arr = features_df[BASE_COLS].to_numpy(dtype=np.float32)
    point_ids = features_df.index.to_numpy(dtype=np.int64)
    pid_to_idx: dict[int, int] = {int(pid): idx for idx, pid in enumerate(point_ids)}

    # Load points CSV for osmid and coordinates
    points_csv = CITY_POINTS_CSV[city]
    if not points_csv.exists():
        raise FileNotFoundError(f"Points CSV not found: {points_csv}")
    pts = pd.read_csv(points_csv)
    pts["point_id"] = pts["point_id"].astype(int)
    pts = pts[pts["point_id"].isin(lock_ids)].set_index("point_id")

    if limit:
        point_ids = point_ids[:limit]

    G = build_projected_graph(city)

    # Build osmid → arr_idx lookup ONCE before the loop. Previously this was inside
    # an `if i == 0` block, which would fail with NameError if the first point was
    # skipped by the `if pid not in pts.index: continue` guard. (Locked DC features
    # were correct only because the first DC point happened to be present.)
    osmid_to_arridx: dict[int, int] = {}
    if "osmid" in pts.columns:
        for pid2, row2 in pts.iterrows():
            if int(pid2) in pid_to_idx:
                osmid_to_arridx[int(row2["osmid"])] = pid_to_idx[int(pid2)]

    out_rows: list[dict] = []
    summary_dists: list[float] = []
    n_fallback = 0
    n_unanchored = 0

    for i, pid in enumerate(point_ids):
        if pid not in pts.index:
            continue
        row = pts.loc[pid]
        osmid = int(row["osmid"]) if "osmid" in pts.columns else -1
        plat = float(row["lat"]) if "lat" in pts.columns else None
        plon = float(row["lon"]) if "lon" in pts.columns else None

        if osmid > 0:
            source_node, was_fallback = snap_to_graph(G, osmid, lat=plat, lon=plon)
        else:
            source_node, was_fallback = -1, True
        if was_fallback:
            n_fallback += 1
        if source_node < 0:
            n_unanchored += 1
            # Skip kNN; emit a row with self-feature defaults so the schema stays aligned
            this_idx = pid_to_idx.get(int(pid), -1)
            knn_wmean = base_arr[this_idx] if this_idx >= 0 else np.zeros(len(BASE_COLS), dtype=np.float32)
            out_row = {
                "point_id": int(pid),
                "city": city,
                "spatial_knn_mean_dist_m": float(max_graph_dist_m),
                "spatial_knn_max_dist_m": float(max_graph_dist_m),
                "spatial_neighbor_count_400m": 0,
                "pedgraph_unanchored": 1,
            }
            for j, col in enumerate(BASE_COLS):
                out_row[f"spatial_knn_wmean__{col}"] = float(knn_wmean[j])
            out_rows.append(out_row)
            continue

        # Graph distances from this node (in metres) to all reachable nodes within cutoff
        lengths = nx.single_source_dijkstra_path_length(G, source_node, cutoff=max_graph_dist_m, weight="length")

        neighbor_entries: list[tuple[float, int, int]] = []
        for graph_node, dist_m in lengths.items():
            if graph_node == source_node:
                continue
            arr_idx = osmid_to_arridx.get(graph_node)
            if arr_idx is not None and arr_idx != pid_to_idx.get(int(pid), -1):
                neighbor_entries.append((dist_m, graph_node, arr_idx))

        # Sort by graph distance, take k nearest
        neighbor_entries.sort(key=lambda x: x[0])
        k_nearest = neighbor_entries[:k]

        this_idx = pid_to_idx.get(int(pid), -1)
        if not k_nearest:
            # No neighbors within cutoff: use self values
            knn_wmean = base_arr[this_idx] if this_idx >= 0 else np.zeros(len(BASE_COLS), dtype=np.float32)
            mean_dist = max_graph_dist_m
        else:
            dists = np.array([e[0] for e in k_nearest], dtype=np.float32)
            idxs = np.array([e[2] for e in k_nearest], dtype=np.int64)
            weights = 1.0 / np.clip(dists, 1.0, None)
            weights /= weights.sum()
            knn_wmean = (base_arr[idxs] * weights[:, None]).sum(axis=0)
            mean_dist = float(dists.mean())
            summary_dists.append(mean_dist)

        out_row: dict = {
            "point_id": int(pid),
            "city": city,
            "spatial_knn_mean_dist_m": mean_dist,
            "spatial_knn_max_dist_m": float(k_nearest[-1][0]) if k_nearest else max_graph_dist_m,
            "spatial_neighbor_count_400m": len(neighbor_entries),
            "pedgraph_unanchored": 0,
        }
        for j, col in enumerate(BASE_COLS):
            out_row[f"spatial_knn_wmean__{col}"] = float(knn_wmean[j])

        out_rows.append(out_row)

        if i % 500 == 0:
            logging.info("[%s] %d/%d points processed", city, i, len(point_ids))

    df = pd.DataFrame(out_rows)
    logging.info(
        "[%s] pedgraph features: %d rows, mean_knn_dist=%.1fm, "
        "fallback_snap=%d, unanchored=%d",
        city, len(df),
        float(np.mean(summary_dists)) if summary_dists else 0.0,
        n_fallback, n_unanchored,
    )
    df.attrs["n_fallback"] = n_fallback
    df.attrs["n_unanchored"] = n_unanchored
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build pedestrian-graph kNN spatial features")
    ap.add_argument("--cities", nargs="+", choices=["msp", "seattle", "dc", "pittsburgh"],
                    default=["msp", "seattle", "dc"])
    ap.add_argument("--k", type=int, default=K_NEIGHBORS)
    ap.add_argument("--max-graph-distance-m", type=float, default=MAX_GRAPH_DISTANCE_M)
    ap.add_argument("--output", default=str(OUT_CSV))
    ap.add_argument("--limit", type=int, default=None, help="First N points per city (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="Add new cities to existing output without regenerating all cities. "
                         "Existing rows for the specified --cities are replaced.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite and not args.append and args.limit is None:
        logging.error(
            "Output exists: %s — use --append to add new cities or --overwrite to regenerate all",
            out_path,
        )
        return 1

    dfs: list[pd.DataFrame] = []
    summary: dict[str, object] = {}

    for city in args.cities:
        logging.info("=== %s ===", city.upper())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_city_pedgraph_features(city, k=args.k,
                                              max_graph_dist_m=args.max_graph_distance_m,
                                              limit=args.limit)
        dfs.append(df)
        summary[city] = {
            "n_points": len(df),
            "mean_knn_dist_m": round(float(df["spatial_knn_mean_dist_m"].mean()), 2),
            "mean_neighbors_400m": round(float(df["spatial_neighbor_count_400m"].mean()), 2),
            "n_snap_fallback": int(df.attrs.get("n_fallback", 0)),
            "n_unanchored": int(df.attrs.get("n_unanchored", 0)),
        }

    combined = pd.concat(dfs, ignore_index=True).sort_values(["city", "point_id"])

    if args.limit is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.append and out_path.exists():
            existing = pd.read_csv(out_path, low_memory=False)
            existing = existing[~existing["city"].isin(args.cities)]
            combined = pd.concat([existing, combined], ignore_index=True).sort_values(
                ["city", "point_id"]
            )
            logging.info("Appended; total rows now: %d", len(combined))
        combined.to_csv(out_path, index=False)
        logging.info("Saved: %s (%d rows × %d cols)", out_path, len(combined), len(combined.columns))
        OUT_JSON.write_text(json.dumps(summary, indent=2))
        logging.info("Summary: %s", OUT_JSON)
    else:
        logging.info("[limit mode] not saving — remove --limit for full output")
        print(combined.head(20).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
