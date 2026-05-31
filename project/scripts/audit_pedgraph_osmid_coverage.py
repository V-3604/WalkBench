"""Audit per-city OSMnx graph coverage of locked-point osmids.

Finding G in ``docs/v2_CODE_AUDIT_2026-04-26.md``: the previous
``snap_to_graph`` fallback returned an arbitrary node when a point's
osmid was missing from the OSMnx walk graph, contaminating the kNN
features for any such point. This script counts, per city, how many of
the locked points are present in the freshly downloaded OSMnx walk graph
versus how many would hit the fallback.

If fallback rate is <1% per city, the locked
``pedgraph_features.csv`` is materially clean and does not require a
rebuild. If higher, rerun ``build_pedgraph_spatial_features.py`` (which
now uses ``ox.distance.nearest_nodes``) and overwrite.

Run:
    & ".venv\\Scripts\\python.exe" project/scripts/audit_pedgraph_osmid_coverage.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import osmnx as ox
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = REPO_ROOT / "project/data/processed/locks"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
OUT_JSON = REPO_ROOT / "project/artifacts/reports/pedgraph_osmid_coverage_audit.json"

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "msp":     (-93.38, 44.84, -93.01, 45.11),
    "seattle": (-122.48, 47.43, -122.19, 47.78),
    "dc":      (-77.1198, 38.7916, -76.9094, 38.9959),
}
CITY_LOCK_FILES: dict[str, str] = {
    "msp":     "msp_v2_final_lock_ids.txt",
    "seattle": "seattle_v2_final_lock_ids.txt",
    "dc":      "dc_v2_final_lock_ids.txt",
}
CITY_POINTS_CSV: dict[str, Path] = {
    "msp":     POINTS_DIR / "msp_points_v2_final_lock.csv",
    "seattle": POINTS_DIR / "seattle_points_v2_final_lock.csv",
    "dc":      POINTS_DIR / "dc_points_v2_final_lock.csv",
}


def audit_city(city: str) -> dict[str, object]:
    lock_path = LOCKS_DIR / CITY_LOCK_FILES[city]
    lock_ids = {int(x) for x in lock_path.read_text().strip().splitlines()}
    pts = pd.read_csv(CITY_POINTS_CSV[city])
    pts["point_id"] = pts["point_id"].astype(int)
    pts = pts[pts["point_id"].isin(lock_ids)].copy()

    bbox = CITY_BBOXES[city]
    west, south, east, north = bbox
    logging.info("[%s] downloading OSMnx walk graph …", city)
    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type="walk")
    except TypeError:
        G = ox.graph_from_bbox(north, south, east, west, network_type="walk")

    graph_node_set = set(G.nodes())
    n_total = len(pts)
    n_in_graph = int(pts["osmid"].astype(int).isin(graph_node_set).sum())
    n_fallback = n_total - n_in_graph
    fallback_pct = (n_fallback / n_total * 100.0) if n_total else 0.0

    logging.info(
        "[%s] %d/%d points in graph (%.2f%% fallback)",
        city, n_in_graph, n_total, fallback_pct,
    )

    return {
        "city": city,
        "n_locked_points": n_total,
        "n_in_graph": n_in_graph,
        "n_fallback": n_fallback,
        "fallback_pct": round(fallback_pct, 3),
        "graph_nodes": len(graph_node_set),
        "rebuild_recommended": fallback_pct >= 1.0,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {"cities": {}}
    any_rebuild = False
    for city in ("msp", "seattle", "dc"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = audit_city(city)
        results["cities"][city] = r
        any_rebuild = any_rebuild or r["rebuild_recommended"]

    results["rebuild_required_for_any_city"] = any_rebuild
    OUT_JSON.write_text(json.dumps(results, indent=2))
    logging.info("Wrote %s", OUT_JSON)
    if any_rebuild:
        logging.warning(
            "At least one city has ≥1%% fallback. Rerun "
            "build_pedgraph_spatial_features.py with --overwrite to incorporate "
            "the nearest_nodes fix."
        )
    else:
        logging.info("All cities <1%% fallback — locked pedgraph_features.csv is materially clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
