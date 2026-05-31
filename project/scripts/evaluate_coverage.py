"""
Probe Mapillary coverage density for a city bounding box.

Fetches all zoom-14 vector tiles that intersect the bbox, counts unique images
within the exact lon/lat bbox, computes bbox area, and reports img/km².

Run:
    & ".venv\Scripts\python.exe" project/scripts/evaluate_coverage.py --city DC --token "$MAPILLARY_TOKEN"
    & ".venv\Scripts\python.exe" project/scripts/evaluate_coverage.py --city Boston --token "$MAPILLARY_TOKEN"

Decision gate (from v2_RUNBOOK_2026.md):
    density >= 200 img/km2  → city is viable, continue
    density <  200 img/km2  → pivot to next candidate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from vt2geojson.tools import vt_bytes_to_geojson

CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (west, south, east, north)
    "DC":      (-77.1198, 38.7916, -76.9094, 38.9959),
    "Boston":  (-71.1912, 42.2279, -70.9866, 42.3960),
    "MSP":     (-93.3290, 44.8900, -93.1850, 45.0520),
    "Seattle": (-122.4360, 47.4810, -122.2350, 47.7340),
}

MAPILLARY_TILE_URL = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
)
USER_AGENT = "WalkCLIP-evaluate-coverage/1.0"
ZOOM = 14
DENSITY_THRESHOLD = 200.0  # img/km²

_EARTH_R = 6_371_008.8  # metres


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "project").exists():
            return p
    raise RuntimeError("Could not locate repo root.")


def load_dotenv_if_present(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (1 << zoom))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    return int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * (1 << zoom)
    )


def bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """Approximate bbox area via Haversine on the four corners."""
    def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dl = math.radians(lon2 - lon1)
        dp = math.radians(lat2 - lat1)
        a = math.sin(dp / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
        return 2 * _EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    width_m = haversine(west, (south + north) / 2, east, (south + north) / 2)
    height_m = haversine((west + east) / 2, south, (west + east) / 2, north)
    return width_m * height_m / 1_000_000.0


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self._lock = threading.Lock()
        self._interval = 1.0 / rps if rps > 0 else 0.0
        self._next = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = time.perf_counter() + self._interval


def fetch_tile(
    session: requests.Session,
    limiter: RateLimiter,
    token: str,
    x: int,
    y: int,
    zoom: int,
) -> list[dict[str, Any]]:
    limiter.acquire()
    url = MAPILLARY_TILE_URL.format(z=zoom, x=x, y=y, token=token)
    resp = session.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 404 or not resp.content:
        return []
    resp.raise_for_status()
    try:
        geo = vt_bytes_to_geojson(resp.content, x, y, zoom, layer="image")
        return geo.get("features") or []
    except Exception:
        return []


def probe_city(
    city: str,
    bbox: tuple[float, float, float, float],
    token: str,
    rps: float,
    workers: int,
) -> dict[str, Any]:
    west, south, east, north = bbox
    tx_min = lon_to_tile_x(west, ZOOM)
    tx_max = lon_to_tile_x(east, ZOOM)
    ty_min = lat_to_tile_y(north, ZOOM)  # north → smaller tile-y
    ty_max = lat_to_tile_y(south, ZOOM)  # south → larger tile-y

    tiles: list[tuple[int, int]] = []
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tiles.append((tx, ty))

    print(f"[{city}] Tiles to probe (z{ZOOM}): {len(tiles)}", flush=True)

    area_km2 = bbox_area_km2(*bbox)
    limiter = RateLimiter(rps)
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "s"):
            thread_local.s = requests.Session()
        return thread_local.s

    unique_ids: set[str] = set()
    completed = 0

    def process_tile(tile: tuple[int, int]) -> list[dict[str, Any]]:
        return fetch_tile(get_session(), limiter, token, tile[0], tile[1], ZOOM)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_tile, t): t for t in tiles}
        for fut in as_completed(futures):
            features = fut.result()
            for feat in features:
                props = feat.get("properties") or {}
                img_id = props.get("id")
                if img_id is None:
                    continue
                coords = (feat.get("geometry") or {}).get("coordinates") or []
                if len(coords) < 2:
                    continue
                lon_img, lat_img = float(coords[0]), float(coords[1])
                if west <= lon_img <= east and south <= lat_img <= north:
                    unique_ids.add(str(img_id))
            completed += 1
            if completed % 20 == 0:
                print(f"[{city}] {completed}/{len(tiles)} tiles done, {len(unique_ids)} images so far", flush=True)

    n_images = len(unique_ids)
    density = n_images / area_km2 if area_km2 > 0 else 0.0
    go = density >= DENSITY_THRESHOLD

    return {
        "city": city,
        "bbox": {"west": west, "south": south, "east": east, "north": north},
        "bbox_area_km2": round(area_km2, 1),
        "tiles_fetched": len(tiles),
        "n_images": n_images,
        "density_img_per_km2": round(density, 1),
        "threshold": DENSITY_THRESHOLD,
        "go": go,
        "verdict": "GO" if go else "NO-GO (pivot to next city)",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe Mapillary coverage density for a city bbox")
    ap.add_argument("--city", required=True, choices=list(CITY_BBOXES), help="City to probe")
    ap.add_argument("--token", default="", help="Mapillary access token (or set MAPILLARY_ACCESS_TOKEN)")
    ap.add_argument("--rps", type=float, default=8.0, help="Tile request rate (default: 8 rps)")
    ap.add_argument("--workers", type=int, default=16, help="Parallel fetch workers (default: 16)")
    ap.add_argument("--output-json", default="", help="Write result JSON to this path (optional)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(repo_root() / ".env")

    token = args.token or os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing Mapillary token. Pass --token or set MAPILLARY_ACCESS_TOKEN.")

    bbox = CITY_BBOXES[args.city]
    result = probe_city(args.city, bbox, token, rps=args.rps, workers=args.workers)

    print()
    print("=" * 55)
    print(f"  City:            {result['city']}")
    print(f"  Bbox area:       {result['bbox_area_km2']:.1f} km²")
    print(f"  Images in bbox:  {result['n_images']:,}")
    print(f"  Density:         {result['density_img_per_km2']:.1f} img/km²")
    print(f"  Threshold:       {result['threshold']:.0f} img/km²")
    print(f"  Verdict:         {result['verdict']}")
    print("=" * 55)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Saved: {out}")

    return 0 if result["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
