"""
Download NAIP aerial tiles for any WalkCLIP city.

Reads the city's points CSV, downloads one 200 m × 200 m tile per point
from the USDA NRCS ArcGIS ImageServer. No API key required.

Run:
    & ".venv\Scripts\python.exe" project/scripts/fetch_naip_tiles.py --city dc --year 2024
    & ".venv\Scripts\python.exe" project/scripts/fetch_naip_tiles.py --city msp --year 2022
    & ".venv\Scripts\python.exe" project/scripts/fetch_naip_tiles.py --city dc --year 2024 --limit 5

Outputs:
    project/data/raw/imagery/aerial_naip/{City}/{point_id}.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

NAIP_EXPORT_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
)
USER_AGENT = "WalkCLIP-fetch-naip-tiles/1.0"

_DEFAULT_WORKERS = 16
_DEFAULT_RPS = 6.0

REPO_ROOT = Path(__file__).resolve().parents[2]

CITY_META: dict[str, dict] = {
    "msp": {
        "points_csv": REPO_ROOT / "project/data/processed/points/msp_points_v2.csv",
        "output_dir": REPO_ROOT / "project/data/raw/imagery/aerial_naip/Minneapolis",
    },
    "seattle": {
        "points_csv": REPO_ROOT / "project/data/processed/points/seattle_points_v2.csv",
        "output_dir": REPO_ROOT / "project/data/raw/imagery/aerial_naip/Seattle",
    },
    "dc": {
        "points_csv": REPO_ROOT / "project/data/processed/points/dc_points_v2.csv",
        "output_dir": REPO_ROOT / "project/data/raw/imagery/aerial_naip/DC",
    },
    "pittsburgh": {
        "points_csv": REPO_ROOT / "project/data/processed/points/pittsburgh_points_v2.csv",
        "output_dir": REPO_ROOT / "project/data/raw/imagery/aerial_naip/Pittsburgh",
    },
}


def meters_to_deg_lat(meters: float) -> float:
    return meters / 111_320.0


def meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    return meters / (111_320.0 * math.cos(math.radians(lat_deg)))


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
class TileResult:
    point_id: int
    lat: float
    lon: float
    saved_path: Path
    download_ok: bool
    skipped: bool
    error: str = ""


def _download_one(
    point_id: int,
    lat: float,
    lon: float,
    out_path: Path,
    session: requests.Session,
    limiter: RateLimiter,
    *,
    side_m: float,
    pixel_size: int,
    fmt: str,
    overwrite: bool,
) -> TileResult:
    if not overwrite and out_path.exists():
        return TileResult(point_id=point_id, lat=lat, lon=lon,
                          saved_path=out_path, download_ok=True, skipped=True)

    half_lat = meters_to_deg_lat(side_m / 2.0)
    half_lon = meters_to_deg_lon(side_m / 2.0, lat)
    bbox = (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
    params = {
        "bbox": ",".join(f"{x:.8f}" for x in bbox),
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{pixel_size},{pixel_size}",
        "format": fmt,
        "f": "json",
    }

    try:
        limiter.acquire()
        resp = session.get(NAIP_EXPORT_URL, params=params, timeout=90,
                           headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        href = resp.json().get("href", "")
        if not href:
            return TileResult(point_id=point_id, lat=lat, lon=lon,
                              saved_path=out_path, download_ok=False, skipped=False,
                              error=f"No href: {resp.text[:200]}")

        limiter.acquire()
        img_resp = session.get(href, timeout=120, headers={"User-Agent": USER_AGENT})
        img_resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_resp.content)
        return TileResult(point_id=point_id, lat=lat, lon=lon,
                          saved_path=out_path, download_ok=True, skipped=False)
    except Exception as exc:
        return TileResult(point_id=point_id, lat=lat, lon=lon,
                          saved_path=out_path, download_ok=False, skipped=False,
                          error=str(exc))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download NAIP aerial tiles for a WalkCLIP city")
    ap.add_argument("--city", required=True, choices=list(CITY_META))
    ap.add_argument("--year", type=int, default=2022,
                    help="NAIP vintage year (informational; the USDA endpoint serves latest available).")
    ap.add_argument("--points-csv", default="", help="Override default points CSV path")
    ap.add_argument("--output-dir", default="", help="Override default output directory")
    ap.add_argument("--tile-size-m", type=float, default=200.0)
    ap.add_argument("--pixel-size", type=int, default=333,
                    help="Output image size in pixels (200m / 0.6m/px ≈ 333). Default: 333.")
    ap.add_argument("--format", default="jpgpng", choices=["jpgpng", "png", "jpg"])
    ap.add_argument("--limit", type=int, default=0, help="Download first N points only (0 = all)")
    ap.add_argument("--rps", type=float, default=_DEFAULT_RPS)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    meta = CITY_META[args.city]
    points_csv = Path(args.points_csv) if args.points_csv else meta["points_csv"]
    out_dir = Path(args.output_dir) if args.output_dir else meta["output_dir"]

    if not points_csv.exists():
        raise SystemExit(
            f"Points CSV not found: {points_csv}\n"
            f"Run select_{args.city}_points.py first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    with points_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[NAIP] City={args.city}  year={args.year}  points={len(rows)}  out={out_dir}", flush=True)

    limiter = RateLimiter(args.rps)
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "s"):
            thread_local.s = requests.Session()
        return thread_local.s

    def process(row: dict) -> TileResult:
        pid = int(row["point_id"])
        lat, lon = float(row["lat"]), float(row["lon"])
        out_path = out_dir / f"{pid}.jpg"
        return _download_one(
            pid, lat, lon, out_path, get_session(), limiter,
            side_m=args.tile_size_m, pixel_size=args.pixel_size,
            fmt=args.format, overwrite=args.overwrite,
        )

    results: list[TileResult] = []
    n_ok = n_skip = n_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, r): r for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.skipped:
                n_skip += 1
            elif res.download_ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  [FAIL] point_id={res.point_id}: {res.error}", flush=True)
            if len(results) % args.log_every == 0:
                print(f"  {len(results)}/{len(rows)}  ok={n_ok}  skip={n_skip}  fail={n_fail}", flush=True)

    print(f"\n[NAIP] Done. ok={n_ok}  skipped={n_skip}  failed={n_fail}", flush=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "lat", "lon", "download_ok", "skipped", "path"])
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.point_id):
            writer.writerow({
                "point_id": r.point_id, "lat": r.lat, "lon": r.lon,
                "download_ok": r.download_ok, "skipped": r.skipped,
                "path": str(r.saved_path),
            })
    print(f"[NAIP] Manifest: {manifest_path}", flush=True)

    if n_fail > len(rows) * 0.10:
        print(f"WARNING: {n_fail}/{len(rows)} tiles failed — check USDA endpoint availability.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
