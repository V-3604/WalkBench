"""
Select research-grade Minneapolis-St Paul points for WalkCLIP v2.

Mirrors select_seattle_points.py exactly:
- Build a Minneapolis-St Paul walk-network graph with OSMnx.
- Keep only intersection-like nodes (street_count >= 3 by default).
- Coverage-gate those nodes with Mapillary so we only keep points with
  usable free street imagery nearby.
- For each kept point, assign one Mapillary image for each of the four
  target headings: 0 / 90 / 180 / 270 degrees.
- Spatially stratify the final set so downtown does not dominate.
- Save the frozen point table to CSV.

Run from repo root:
  & ".venv\\Scripts\\python.exe" project\\scripts\\select_msp_points.py

Smoke test:
  & ".venv\\Scripts\\python.exe" project\\scripts\\select_msp_points.py --target-points 20
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import osmnx as ox
import requests
from vt2geojson.tools import vt_bytes_to_geojson


# Minneapolis + Saint Paul combined urban core.
# Similar geographic extent to Seattle (~22 km wide x 19 km tall).
MSP_BBOX = (-93.3290, 44.8900, -93.0100, 45.0600)  # west, south, east, north

TARGET_HEADINGS = (0, 90, 180, 270)
USER_AGENT = "WalkCLIP-select-msp-points/1.0"
MAPILLARY_TILE_URL = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
)

_DEFAULT_WORKERS = 16


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for path in [cwd, *cwd.parents]:
        if (path / "project").exists():
            return path
    raise RuntimeError("Could not locate repo root (expected a 'project/' folder).")


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


class RateLimiter:
    def __init__(self, rps: float) -> None:
        self._lock = threading.Lock()
        self._min_interval = 0.0 if rps <= 0 else 1.0 / rps
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = time.perf_counter() + self._min_interval


@dataclass
class CandidatePoint:
    osmid: Any
    lat: float
    lon: float
    street_count: int
    tile_x: int
    tile_y: int


@dataclass
class SelectedPoint:
    point_id: int
    osmid: Any
    lat: float
    lon: float
    street_count: int
    tile_x: int
    tile_y: int
    image_id_0: str
    image_id_90: str
    image_id_180: str
    image_id_270: str
    dist_m_0: float
    dist_m_90: float
    dist_m_180: float
    dist_m_270: float
    compass_angle_0: float
    compass_angle_90: float
    compass_angle_180: float
    compass_angle_270: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select and save research-grade MSP points for WalkCLIP v2."
    )
    ap.add_argument(
        "--target-points",
        type=int,
        default=4874,
        help="Number of MSP points to freeze. Default: 4,874 (matches Seattle).",
    )
    ap.add_argument(
        "--points-csv",
        default="project/data/processed/points/msp_points_v2.csv",
        help="Output MSP point table.",
    )
    ap.add_argument(
        "--zoom",
        type=int,
        default=14,
        help="Mapillary vector tile zoom for coverage gating. Default: 14.",
    )
    ap.add_argument(
        "--min-street-count",
        type=int,
        default=3,
        help="Minimum OSMnx street_count to treat a node as an intersection. Default: 3.",
    )
    ap.add_argument(
        "--max-radius-m",
        type=float,
        default=35.0,
        help="Maximum allowed distance between selected point and each chosen Mapillary image.",
    )
    ap.add_argument(
        "--max-heading-error-deg",
        type=float,
        default=60.0,
        help="Maximum compass-angle error allowed when assigning a target heading.",
    )
    ap.add_argument(
        "--min-image-year",
        type=int,
        default=2019,
        help="Ignore Mapillary images older than this year. Default: 2019 (MSP coverage is slightly older than Seattle).",
    )
    ap.add_argument(
        "--grid-km",
        type=float,
        default=1.0,
        help="Approximate stratification grid cell size in kilometers. Default: 1.0.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic point selection.",
    )
    ap.add_argument(
        "--mapillary-rps",
        type=float,
        default=16.0,
        help="Global Mapillary request rate for tile HTTP during point selection. Default: 16 req/s.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Number of parallel HTTP workers for Mapillary tile fetching. Default: {_DEFAULT_WORKERS}.",
    )
    return ap.parse_args()


def lon_to_tile_x(lon: float, zoom: int, _pow2: int | None = None) -> int:
    n = _pow2 if _pow2 is not None else (1 << zoom)
    return int((lon + 180.0) / 360.0 * n)


def lat_to_tile_y(lat: float, zoom: int, _pow2: int | None = None) -> int:
    n = _pow2 if _pow2 is not None else (1 << zoom)
    lat_rad = math.radians(lat)
    return int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2
    ) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def heading_error_deg(target: float, actual: float) -> float:
    return abs((actual - target + 180.0) % 360.0 - 180.0)


_LAT_M_PER_DEG = 111_320.0


def grid_cell(
    lat: float, lon: float, *, ref_lat: float, ref_lon: float, grid_km: float
) -> tuple[int, int]:
    lon_m_per_deg = _LAT_M_PER_DEG * math.cos(math.radians(ref_lat))
    x_km = (lon - ref_lon) * lon_m_per_deg / 1000.0
    y_km = (lat - ref_lat) * _LAT_M_PER_DEG / 1000.0
    return (math.floor(x_km / grid_km), math.floor(y_km / grid_km))


def build_candidate_points(
    *,
    bbox: tuple[float, float, float, float],
    zoom: int,
    min_street_count: int,
) -> list[CandidatePoint]:
    print("[OSMnx] Downloading MSP walk graph...", flush=True)
    ox.settings.log_console = True
    ox.settings.log_level = logging.INFO
    graph = ox.graph_from_bbox(bbox=bbox, network_type="walk")
    print(
        f"[OSMnx] Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.",
        flush=True,
    )
    pow2 = 1 << zoom
    candidates: list[CandidatePoint] = []
    for osmid, data in graph.nodes(data=True):
        street_count = int(data.get("street_count") or 0)
        if street_count < min_street_count:
            continue
        lat = float(data["y"])
        lon = float(data["x"])
        candidates.append(
            CandidatePoint(
                osmid=osmid,
                lat=lat,
                lon=lon,
                street_count=street_count,
                tile_x=lon_to_tile_x(lon, zoom, pow2),
                tile_y=lat_to_tile_y(lat, zoom, pow2),
            )
        )
    print(
        f"[OSMnx] Candidate MSP intersections with street_count>={min_street_count}: {len(candidates)}",
        flush=True,
    )
    return candidates


_SENTINEL: list[Any] = []


def fetch_tile_features(
    session: requests.Session,
    limiter: RateLimiter,
    *,
    token: str,
    zoom: int,
    tile_x: int,
    tile_y: int,
    cache: dict[tuple[int, int], list[dict[str, Any]]],
    cache_lock: threading.Lock,
) -> list[dict[str, Any]]:
    key = (tile_x, tile_y)
    if key in cache:
        return cache[key]
    with cache_lock:
        if key in cache:
            return cache[key]
        cache[key] = _SENTINEL  # type: ignore[assignment]

    limiter.acquire()
    url = MAPILLARY_TILE_URL.format(z=zoom, x=tile_x, y=tile_y, token=token)
    resp = session.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 404 or not resp.content:
        result: list[dict[str, Any]] = []
    else:
        resp.raise_for_status()
        try:
            geojson = vt_bytes_to_geojson(resp.content, tile_x, tile_y, zoom, layer="image")
            result = geojson.get("features") or []
        except Exception:
            result = []

    with cache_lock:
        cache[key] = result
    return result


def _wait_for_tile(
    cache: dict[tuple[int, int], list[dict[str, Any]]],
    key: tuple[int, int],
) -> list[dict[str, Any]]:
    while True:
        val = cache.get(key)
        if val is not _SENTINEL:
            return val or []
        time.sleep(0.001)


def best_heading_candidates(
    *,
    point: CandidatePoint,
    features: list[dict[str, Any]],
    max_radius_m: float,
    max_heading_error_deg: float,
    min_image_year: int,
) -> dict[int, dict[str, Any]] | None:
    import calendar
    min_ts_ms = calendar.timegm((min_image_year, 1, 1, 0, 0, 0, 0, 0, 0)) * 1000
    by_heading: dict[int, dict[str, Any]] = {}

    for feature in features:
        props = feature.get("properties") or {}
        ts = props.get("captured_at")
        compass_angle = props.get("compass_angle")
        image_id = props.get("id")
        coords = (feature.get("geometry") or {}).get("coordinates")
        if ts is None or compass_angle is None or image_id is None or not coords:
            continue
        if ts < min_ts_ms:
            continue
        dist_m = haversine_m(point.lon, point.lat, float(coords[0]), float(coords[1]))
        if dist_m > max_radius_m:
            continue
        compass_angle = float(compass_angle)
        for heading in TARGET_HEADINGS:
            err = heading_error_deg(float(heading), compass_angle)
            if err > max_heading_error_deg:
                continue
            score = dist_m + err * 0.25
            existing = by_heading.get(heading)
            if existing is None or score < existing["score"]:
                by_heading[heading] = {
                    "image_id": str(image_id),
                    "dist_m": dist_m,
                    "compass_angle": compass_angle,
                    "score": score,
                }

    if len(by_heading) != 4:
        return None
    return by_heading


def _process_candidate(
    point: CandidatePoint,
    session: requests.Session,
    limiter: RateLimiter,
    token: str,
    zoom: int,
    max_radius_m: float,
    max_heading_error_deg: float,
    min_image_year: int,
    tile_cache: dict[tuple[int, int], list[dict[str, Any]]],
    cache_lock: threading.Lock,
) -> SelectedPoint | None:
    features: list[dict[str, Any]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (point.tile_x + dx, point.tile_y + dy)
            if tile_cache.get(key) is _SENTINEL:
                features.extend(_wait_for_tile(tile_cache, key))
            else:
                features.extend(
                    fetch_tile_features(
                        session, limiter, token=token, zoom=zoom,
                        tile_x=key[0], tile_y=key[1],
                        cache=tile_cache, cache_lock=cache_lock,
                    )
                )
    chosen = best_heading_candidates(
        point=point,
        features=features,
        max_radius_m=max_radius_m,
        max_heading_error_deg=max_heading_error_deg,
        min_image_year=min_image_year,
    )
    if chosen is None:
        return None
    return SelectedPoint(
        point_id=-1,
        osmid=point.osmid,
        lat=point.lat,
        lon=point.lon,
        street_count=point.street_count,
        tile_x=point.tile_x,
        tile_y=point.tile_y,
        image_id_0=chosen[0]["image_id"],
        image_id_90=chosen[90]["image_id"],
        image_id_180=chosen[180]["image_id"],
        image_id_270=chosen[270]["image_id"],
        dist_m_0=round(chosen[0]["dist_m"], 3),
        dist_m_90=round(chosen[90]["dist_m"], 3),
        dist_m_180=round(chosen[180]["dist_m"], 3),
        dist_m_270=round(chosen[270]["dist_m"], 3),
        compass_angle_0=round(chosen[0]["compass_angle"], 3),
        compass_angle_90=round(chosen[90]["compass_angle"], 3),
        compass_angle_180=round(chosen[180]["compass_angle"], 3),
        compass_angle_270=round(chosen[270]["compass_angle"], 3),
    )


def select_points(
    *,
    candidates: list[CandidatePoint],
    target_points: int,
    token: str,
    zoom: int,
    max_radius_m: float,
    max_heading_error_deg: float,
    min_image_year: int,
    grid_km: float,
    seed: int,
    limiter: RateLimiter,
    workers: int = _DEFAULT_WORKERS,
) -> list[SelectedPoint]:
    tile_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cache_lock = threading.Lock()
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            thread_local.session = s
        return thread_local.session

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)

    overfetch_target = int(target_points * 1.4)
    stop_event = threading.Event()

    covered: list[SelectedPoint] = []
    completed = 0
    covered_lock = threading.Lock()

    def process(point: CandidatePoint) -> SelectedPoint | None:
        if stop_event.is_set():
            return None
        return _process_candidate(
            point, get_session(), limiter, token, zoom,
            max_radius_m, max_heading_error_deg, min_image_year,
            tile_cache, cache_lock,
        )

    print(
        f"[Select] Processing up to {len(shuffled)} candidates with {workers} workers "
        f"(early-stop at {overfetch_target} covered)...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, p): p for p in shuffled}
        for future in as_completed(futures):
            result = future.result()
            with covered_lock:
                completed += 1
                if result is not None:
                    covered.append(result)
                    if len(covered) >= overfetch_target:
                        stop_event.set()
                if completed % 500 == 0:
                    print(
                        f"[Select] Checked {completed}/{len(shuffled)} candidates; covered={len(covered)}",
                        flush=True,
                    )
            if stop_event.is_set():
                for f in futures:
                    f.cancel()
                break

    print(f"[Select] Covered candidates with all 4 headings: {len(covered)}", flush=True)
    if len(covered) < target_points:
        raise SystemExit(
            f"Only {len(covered)} MSP points passed the coverage gate, below target {target_points}. "
            "Try: lower --min-image-year, raise --max-radius-m, lower --min-street-count, "
            "or reduce --target-points."
        )

    rng = random.Random(seed)
    west, south, _, _ = MSP_BBOX
    buckets: dict[tuple[int, int], list[SelectedPoint]] = defaultdict(list)
    for point in covered:
        buckets[
            grid_cell(point.lat, point.lon, ref_lat=south, ref_lon=west, grid_km=grid_km)
        ].append(point)

    ordered_cells = list(buckets.keys())
    rng.shuffle(ordered_cells)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[SelectedPoint] = []
    while len(selected) < target_points:
        progressed = False
        for cell in ordered_cells:
            bucket = buckets[cell]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= target_points:
                break
        if not progressed:
            break

    if len(selected) < target_points:
        raise SystemExit(
            f"Spatial stratification only yielded {len(selected)} points, below target {target_points}."
        )

    selected.sort(key=lambda p: (p.tile_y, p.tile_x, p.lat, p.lon))
    for idx, point in enumerate(selected):
        point.point_id = idx  # type: ignore[misc]
    return selected


def write_points_csv(path: Path, selected: list[SelectedPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_id", "metro", "state", "lat", "lon",
        "osmid", "street_count", "tile_x", "tile_y",
        "image_id_0", "image_id_90", "image_id_180", "image_id_270",
        "dist_m_0", "dist_m_90", "dist_m_180", "dist_m_270",
        "compass_angle_0", "compass_angle_90", "compass_angle_180", "compass_angle_270",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in selected:
            writer.writerow({
                "point_id": p.point_id,
                "metro": "Minneapolis",
                "state": "MN",
                "lat": p.lat,
                "lon": p.lon,
                "osmid": p.osmid,
                "street_count": p.street_count,
                "tile_x": p.tile_x,
                "tile_y": p.tile_y,
                "image_id_0": p.image_id_0,
                "image_id_90": p.image_id_90,
                "image_id_180": p.image_id_180,
                "image_id_270": p.image_id_270,
                "dist_m_0": p.dist_m_0,
                "dist_m_90": p.dist_m_90,
                "dist_m_180": p.dist_m_180,
                "dist_m_270": p.dist_m_270,
                "compass_angle_0": p.compass_angle_0,
                "compass_angle_90": p.compass_angle_90,
                "compass_angle_180": p.compass_angle_180,
                "compass_angle_270": p.compass_angle_270,
            })


def main() -> int:
    args = parse_args()
    root = repo_root()
    load_dotenv_if_present(root / ".env")

    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing MAPILLARY_ACCESS_TOKEN in environment / .env.")

    points_csv = root / args.points_csv
    limiter = RateLimiter(args.mapillary_rps)

    candidates = build_candidate_points(
        bbox=MSP_BBOX,
        zoom=args.zoom,
        min_street_count=args.min_street_count,
    )
    selected = select_points(
        candidates=candidates,
        target_points=args.target_points,
        token=token,
        zoom=args.zoom,
        max_radius_m=args.max_radius_m,
        max_heading_error_deg=args.max_heading_error_deg,
        min_image_year=args.min_image_year,
        grid_km=args.grid_km,
        seed=args.seed,
        limiter=limiter,
        workers=args.workers,
    )
    write_points_csv(points_csv, selected)
    print(f"[Select] Wrote MSP point table: {points_csv}", flush=True)
    print(
        f"[Select] Final selected points: {len(selected)} (target={args.target_points})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
