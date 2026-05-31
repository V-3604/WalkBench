"""
Select research-grade Washington DC points for WalkCLIP v2.

- Build DC walk-network graph with OSMnx.
- Keep intersection-like nodes (street_count >= 3).
- Coverage-gate with Mapillary; keep only points with images for all 4 headings.
- Stratify spatially so downtown DC does not dominate.
- Save the frozen point table to CSV.

Run from repo root:
    & ".venv\Scripts\python.exe" project/scripts/select_dc_points.py

Smoke test:
    & ".venv\Scripts\python.exe" project/scripts/select_dc_points.py --target-n 20

Full run per runbook:
    & ".venv\Scripts\python.exe" project/scripts/select_dc_points.py ^
        --bbox -77.1198 38.7916 -76.9094 38.9959 ^
        --target-n 5000 ^
        --stratify-by grid ^
        --output project/data/processed/points/dc_points_v2.csv
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

DC_BBOX = (-77.1198, 38.7916, -76.9094, 38.9959)  # west, south, east, north
TARGET_HEADINGS = (0, 90, 180, 270)
USER_AGENT = "WalkCLIP-select-dc-points/1.0"
MAPILLARY_TILE_URL = (
    "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={token}"
)
_DEFAULT_WORKERS = 16


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
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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


def lon_to_tile_x(lon: float, zoom: int, _pow2: int | None = None) -> int:
    n = _pow2 if _pow2 is not None else (1 << zoom)
    return int((lon + 180.0) / 360.0 * n)


def lat_to_tile_y(lat: float, zoom: int, _pow2: int | None = None) -> int:
    n = _pow2 if _pow2 is not None else (1 << zoom)
    lat_rad = math.radians(lat)
    return int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    )


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def heading_error_deg(target: float, actual: float) -> float:
    return abs((actual - target + 180.0) % 360.0 - 180.0)


_LAT_M_PER_DEG = 111_320.0


def grid_cell(lat: float, lon: float, *, ref_lat: float, ref_lon: float, grid_km: float) -> tuple[int, int]:
    lon_m_per_deg = _LAT_M_PER_DEG * math.cos(math.radians(ref_lat))
    x_km = (lon - ref_lon) * lon_m_per_deg / 1000.0
    y_km = (lat - ref_lat) * _LAT_M_PER_DEG / 1000.0
    return (math.floor(x_km / grid_km), math.floor(y_km / grid_km))


def build_candidate_points(
    bbox: tuple[float, float, float, float],
    zoom: int,
    min_street_count: int,
) -> list[CandidatePoint]:
    print("[OSMnx] Downloading DC walk graph …", flush=True)
    ox.settings.log_console = True
    ox.settings.log_level = logging.INFO
    graph = ox.graph_from_bbox(bbox=bbox, network_type="walk")
    print(f"[OSMnx] {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges", flush=True)

    pow2 = 1 << zoom
    candidates: list[CandidatePoint] = []
    for osmid, data in graph.nodes(data=True):
        sc = int(data.get("street_count") or 0)
        if sc < min_street_count:
            continue
        lat, lon = float(data["y"]), float(data["x"])
        candidates.append(CandidatePoint(
            osmid=osmid, lat=lat, lon=lon, street_count=sc,
            tile_x=lon_to_tile_x(lon, zoom, pow2),
            tile_y=lat_to_tile_y(lat, zoom, pow2),
        ))
    print(f"[OSMnx] Intersection candidates (sc>={min_street_count}): {len(candidates)}", flush=True)
    return candidates


_SENTINEL: list[Any] = []


def fetch_tile_features(
    session: requests.Session,
    limiter: RateLimiter,
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
            geo = vt_bytes_to_geojson(resp.content, tile_x, tile_y, zoom, layer="image")
            result = geo.get("features") or []
        except Exception:
            result = []

    with cache_lock:
        cache[key] = result
    return result


def _wait_for_tile(cache: dict[tuple[int, int], list[dict[str, Any]]], key: tuple[int, int]) -> list[dict[str, Any]]:
    while True:
        val = cache.get(key)
        if val is not _SENTINEL:
            return val or []
        time.sleep(0.001)


def best_heading_candidates(
    point: CandidatePoint,
    features: list[dict[str, Any]],
    max_radius_m: float,
    max_heading_error_deg: float,
    min_image_year: int,
) -> dict[int, dict[str, Any]] | None:
    import calendar
    min_ts_ms = calendar.timegm((min_image_year, 1, 1, 0, 0, 0, 0, 0, 0)) * 1000
    by_heading: dict[int, dict[str, Any]] = {}

    for feat in features:
        props = feat.get("properties") or {}
        ts = props.get("captured_at")
        compass = props.get("compass_angle")
        img_id = props.get("id")
        coords = (feat.get("geometry") or {}).get("coordinates")
        if ts is None or compass is None or img_id is None or not coords:
            continue
        if ts < min_ts_ms:
            continue
        dist_m = haversine_m(point.lon, point.lat, float(coords[0]), float(coords[1]))
        if dist_m > max_radius_m:
            continue
        compass = float(compass)
        for heading in TARGET_HEADINGS:
            err = heading_error_deg(float(heading), compass)
            if err > max_heading_error_deg:
                continue
            score = dist_m + err * 0.25
            existing = by_heading.get(heading)
            if existing is None or score < existing["score"]:
                by_heading[heading] = {
                    "image_id": str(img_id),
                    "dist_m": dist_m,
                    "compass_angle": compass,
                    "score": score,
                }

    return by_heading if len(by_heading) == 4 else None


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
                features.extend(fetch_tile_features(
                    session, limiter, token, zoom, key[0], key[1], tile_cache, cache_lock,
                ))

    chosen = best_heading_candidates(point, features, max_radius_m, max_heading_error_deg, min_image_year)
    if chosen is None:
        return None
    return SelectedPoint(
        point_id=-1, osmid=point.osmid, lat=point.lat, lon=point.lon,
        street_count=point.street_count, tile_x=point.tile_x, tile_y=point.tile_y,
        image_id_0=chosen[0]["image_id"], image_id_90=chosen[90]["image_id"],
        image_id_180=chosen[180]["image_id"], image_id_270=chosen[270]["image_id"],
        dist_m_0=round(chosen[0]["dist_m"], 3), dist_m_90=round(chosen[90]["dist_m"], 3),
        dist_m_180=round(chosen[180]["dist_m"], 3), dist_m_270=round(chosen[270]["dist_m"], 3),
        compass_angle_0=round(chosen[0]["compass_angle"], 3),
        compass_angle_90=round(chosen[90]["compass_angle"], 3),
        compass_angle_180=round(chosen[180]["compass_angle"], 3),
        compass_angle_270=round(chosen[270]["compass_angle"], 3),
    )


def select_points(
    candidates: list[CandidatePoint],
    target_points: int,
    bbox: tuple[float, float, float, float],
    token: str,
    zoom: int,
    max_radius_m: float,
    max_heading_error_deg: float,
    min_image_year: int,
    grid_km: float,
    seed: int,
    limiter: RateLimiter,
    workers: int,
) -> list[SelectedPoint]:
    tile_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cache_lock = threading.Lock()
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "s"):
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            thread_local.s = s
        return thread_local.s

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    overfetch = int(target_points * 1.4)
    stop_event = threading.Event()
    covered: list[SelectedPoint] = []
    completed = 0
    lock = threading.Lock()

    def process(point: CandidatePoint) -> SelectedPoint | None:
        if stop_event.is_set():
            return None
        return _process_candidate(
            point, get_session(), limiter, token, zoom,
            max_radius_m, max_heading_error_deg, min_image_year, tile_cache, cache_lock,
        )

    print(f"[Select] Processing {len(shuffled)} candidates with {workers} workers (stop at {overfetch}) …", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, p): p for p in shuffled}
        for fut in as_completed(futures):
            result = fut.result()
            with lock:
                completed += 1
                if result is not None:
                    covered.append(result)
                    if len(covered) >= overfetch:
                        stop_event.set()
                if completed % 500 == 0:
                    print(f"[Select] {completed}/{len(shuffled)} checked; covered={len(covered)}", flush=True)
            if stop_event.is_set():
                for f in futures:
                    f.cancel()
                break

    print(f"[Select] Points with 4/4 headings: {len(covered)}", flush=True)
    if len(covered) < target_points:
        raise SystemExit(
            f"Only {len(covered)} DC points passed coverage gate, need {target_points}. "
            "Relax --max-radius-m, lower --min-street-count, or check Mapillary coverage first."
        )

    rng = random.Random(seed)
    west, south, _, _ = bbox
    buckets: dict[tuple[int, int], list[SelectedPoint]] = defaultdict(list)
    for p in covered:
        buckets[grid_cell(p.lat, p.lon, ref_lat=south, ref_lon=west, grid_km=grid_km)].append(p)

    cells = list(buckets.keys())
    rng.shuffle(cells)
    for b in buckets.values():
        rng.shuffle(b)

    selected: list[SelectedPoint] = []
    while len(selected) < target_points:
        progressed = False
        for cell in cells:
            if not buckets[cell]:
                continue
            selected.append(buckets[cell].pop())
            progressed = True
            if len(selected) >= target_points:
                break
        if not progressed:
            break

    if len(selected) < target_points:
        raise SystemExit(f"Stratification yielded only {len(selected)} points (need {target_points}).")

    selected.sort(key=lambda p: (p.tile_y, p.tile_x, p.lat, p.lon))
    for idx, p in enumerate(selected):
        p.point_id = idx  # type: ignore[misc]
    return selected


def write_points_csv(path: Path, selected: list[SelectedPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_id", "metro", "state", "lat", "lon", "osmid", "street_count",
        "tile_x", "tile_y",
        "image_id_0", "image_id_90", "image_id_180", "image_id_270",
        "dist_m_0", "dist_m_90", "dist_m_180", "dist_m_270",
        "compass_angle_0", "compass_angle_90", "compass_angle_180", "compass_angle_270",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in selected:
            writer.writerow({
                "point_id": p.point_id, "metro": "Washington DC", "state": "DC",
                "lat": p.lat, "lon": p.lon, "osmid": p.osmid, "street_count": p.street_count,
                "tile_x": p.tile_x, "tile_y": p.tile_y,
                "image_id_0": p.image_id_0, "image_id_90": p.image_id_90,
                "image_id_180": p.image_id_180, "image_id_270": p.image_id_270,
                "dist_m_0": p.dist_m_0, "dist_m_90": p.dist_m_90,
                "dist_m_180": p.dist_m_180, "dist_m_270": p.dist_m_270,
                "compass_angle_0": p.compass_angle_0, "compass_angle_90": p.compass_angle_90,
                "compass_angle_180": p.compass_angle_180, "compass_angle_270": p.compass_angle_270,
            })


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Select DC intersection points for WalkCLIP v2")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                    default=list(DC_BBOX))
    ap.add_argument("--target-n", type=int, default=5000)
    ap.add_argument("--stratify-by", choices=["grid", "tract_density"], default="grid",
                    help="tract_density uses EPA SLD D1B weights; grid is pure spatial stratification")
    ap.add_argument("--output", default="project/data/processed/points/dc_points_v2.csv")
    ap.add_argument("--zoom", type=int, default=14)
    ap.add_argument("--min-street-count", type=int, default=3)
    ap.add_argument("--max-radius-m", type=float, default=35.0)
    ap.add_argument("--max-heading-error-deg", type=float, default=60.0)
    ap.add_argument("--min-image-year", type=int, default=2020)
    ap.add_argument("--grid-km", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mapillary-rps", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    load_dotenv_if_present(root / ".env")

    token = os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing MAPILLARY_ACCESS_TOKEN in environment / .env.")

    if args.stratify_by == "tract_density":
        print("[Info] tract_density stratification requested; using spatial grid (EPA not loaded at point-selection stage).", flush=True)

    bbox = tuple(args.bbox)  # type: ignore[arg-type]
    candidates = build_candidate_points(bbox, args.zoom, args.min_street_count)
    limiter = RateLimiter(args.mapillary_rps)
    selected = select_points(
        candidates=candidates, target_points=args.target_n, bbox=bbox,
        token=token, zoom=args.zoom, max_radius_m=args.max_radius_m,
        max_heading_error_deg=args.max_heading_error_deg, min_image_year=args.min_image_year,
        grid_km=args.grid_km, seed=args.seed, limiter=limiter, workers=args.workers,
    )
    out = root / args.output
    write_points_csv(out, selected)
    print(f"[Select] Wrote {len(selected)} DC points → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
