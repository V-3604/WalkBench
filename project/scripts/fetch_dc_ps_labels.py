"""
Download Project Sidewalk attribute labels for Washington DC.

The live DC deployment (sidewalk-dc.cs.washington.edu) often returns 503 when
overloaded or in maintenance. This script tiles the DC bbox into smaller
requests (same pattern as the Seattle equity-study snapshot) and merges them
into one GeoJSON FeatureCollection for audit_label_noise.py.

Output (default):
    project/data/external/sidewalk-equity-study/dc/datasets/
        01-project-sidewalk-labels/attributes_dc.json

Run (Windows, repo root):
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_dc_ps_labels.py
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_dc_ps_labels.py --check-only
    & ".venv\\Scripts\\python.exe" project\\scripts\\fetch_dc_ps_labels.py --tile-deg 0.04 --retries 5

After a successful download:
    & ".venv\\Scripts\\python.exe" project\\scripts\\audit_label_noise.py --cities dc
    & ".venv\\Scripts\\python.exe" project\\scripts\\eval_model_vs_ps.py --test-city dc
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
# Match select_dc_points.py
DC_BBOX = (-77.1198, 38.7916, -76.9094, 38.9959)  # west, south, east, north
DEFAULT_OUT = (
    REPO_ROOT
    / "project/data/external/sidewalk-equity-study/dc/datasets"
    / "01-project-sidewalk-labels/attributes_dc.json"
)
API_TEMPLATE = (
    "https://sidewalk-dc.cs.washington.edu/v2/access/attributes"
    "?lat1={lat1:.6f}&lng1={lng1:.6f}&lat2={lat2:.6f}&lng2={lng2:.6f}"
    "&filetype=geojson"
)
USER_AGENT = "WalkCLIP-fetch-dc-ps-labels/1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch DC Project Sidewalk labels via tiled API.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--tile-deg",
        type=float,
        default=0.04,
        help="Tile size in degrees (~4 km lat). Smaller tiles if server times out.",
    )
    p.add_argument("--retries", type=int, default=4, help="Retries per tile on 5xx/timeout.")
    p.add_argument("--pause-s", type=float, default=1.0, help="Pause between tile requests.")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Probe one small tile and exit (no download).",
    )
    p.add_argument("--bbox", nargs=4, type=float, default=list(DC_BBOX),
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    return p.parse_args()


def tile_bbox(
    west: float, south: float, east: float, north: float, step: float
) -> list[tuple[float, float, float, float]]:
    tiles: list[tuple[float, float, float, float]] = []
    lat = south
    while lat < north:
        lat2 = min(lat + step, north)
        lon = west
        while lon < east:
            lon2 = min(lon + step, east)
            tiles.append((lon, lat, lon2, lat2))
            lon = lon2
        lat = lat2
    return tiles


def fetch_tile(
    session: requests.Session,
    *,
    lng1: float,
    lat1: float,
    lng2: float,
    lat2: float,
    retries: int,
) -> list[dict]:
    url = API_TEMPLATE.format(lat1=lat1, lng1=lng1, lat2=lat2, lng2=lng2)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=180)
            if resp.status_code == 503:
                raise requests.HTTPError(f"503 Service Unavailable (tile {lat1:.3f},{lng1:.3f})")
            resp.raise_for_status()
            data = resp.json()
            feats = data.get("features", []) if isinstance(data, dict) else []
            return feats
        except Exception as exc:
            last_err = exc
            wait = min(30.0, 2.0 ** attempt)
            logging.warning("Tile attempt %d/%d failed: %s — retry in %.0fs", attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Tile fetch failed after {retries} attempts: {last_err}") from last_err


def feature_key(feat: dict) -> str:
    props = feat.get("properties") or {}
    for key in ("attribute_id", "label_id", "id"):
        val = props.get(key)
        if val is not None:
            return f"{key}:{val}"
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or []
    if len(coords) >= 2:
        return f"coord:{coords[0]:.6f},{coords[1]:.6f}"
    return json.dumps(feat, sort_keys=True, default=str)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    west, south, east, north = args.bbox
    tiles = tile_bbox(west, south, east, north, args.tile_deg)
    logging.info("DC bbox tiles: %d (step=%.3f deg)", len(tiles), args.tile_deg)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    if args.check_only:
        lng1, lat1, lng2, lat2 = tiles[len(tiles) // 2]
        url = API_TEMPLATE.format(lat1=lat1, lng1=lng1, lat2=lat2, lng2=lng2)
        logging.info("Probe: %s", url)
        resp = session.get(url, timeout=60)
        logging.info("HTTP %s (%d bytes)", resp.status_code, len(resp.content))
        if resp.status_code == 200:
            data = resp.json()
            n = len(data.get("features", [])) if isinstance(data, dict) else 0
            logging.info("OK — sample tile has %d features", n)
            return 0
        logging.error(
            "DC API not reachable (HTTP %s). Server may be down — retry later or contact "
            "the Project Sidewalk / Makeability Lab team.",
            resp.status_code,
        )
        return 1

    merged: dict[str, dict] = {}
    for i, (lng1, lat1, lng2, lat2) in enumerate(tiles, start=1):
        logging.info("[%d/%d] Fetch tile lat %.3f–%.3f, lon %.3f–%.3f", i, len(tiles), lat1, lat2, lng1, lng2)
        feats = fetch_tile(
            session,
            lng1=lng1,
            lat1=lat1,
            lng2=lng2,
            lat2=lat2,
            retries=args.retries,
        )
        for feat in feats:
            merged[feature_key(feat)] = feat
        logging.info("[%d/%d] tile features=%d merged_total=%d", i, len(tiles), len(feats), len(merged))
        if i < len(tiles):
            time.sleep(args.pause_s)

    if not merged:
        logging.error("No features downloaded — DC API may be down (503) or bbox empty.")
        return 1

    out = {
        "type": "FeatureCollection",
        "features": list(merged.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out), encoding="utf-8")
    logging.info("Wrote %s (%d features)", args.output, len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
