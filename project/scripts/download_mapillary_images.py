"""
Download Mapillary heading images for any WalkCLIP city (unified script).

Reads a lock-IDs file and a points CSV (which has image_id_0/90/180/270),
resolves each image's thumb_1024_url via the Mapillary Graph API, and downloads.

Run:
    & ".venv\Scripts\python.exe" project/scripts/download_mapillary_images.py ^
        --point-ids-file project/data/processed/locks/dc_v2_final_lock_ids.txt ^
        --points-csv project/data/processed/points/dc_points_v2.csv ^
        --headings 0 90 180 270 ^
        --output-dir project/data/raw/imagery/street_view_mapillary/DC ^
        --token "%MAPILLARY_ACCESS_TOKEN%"

Smoke test:
    & ".venv\Scripts\python.exe" project/scripts/download_mapillary_images.py ^
        --point-ids-file project/data/processed/locks/dc_v2_final_lock_ids.txt ^
        --points-csv project/data/processed/points/dc_points_v2.csv ^
        --limit 5
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

GRAPH_IMAGE_URL = "https://graph.mapillary.com/{image_id}"
USER_AGENT = "WalkCLIP-download-mapillary/1.0"
_DEFAULT_WORKERS = 16
_DEFAULT_RPS = 16.0

REPO_ROOT = Path(__file__).resolve().parents[2]


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
class DownloadResult:
    point_id: int
    heading: int
    image_id: str
    saved_path: Path
    ok: bool
    skipped: bool
    error: str = ""


def resolve_thumb_url(
    session: requests.Session,
    limiter: RateLimiter,
    token: str,
    image_id: str,
) -> str | None:
    limiter.acquire()
    resp = session.get(
        GRAPH_IMAGE_URL.format(image_id=image_id),
        params={"fields": "thumb_1024_url", "access_token": token},
        timeout=30,
        headers={"User-Agent": USER_AGENT},
    )
    if resp.status_code != 200:
        return None
    url = resp.json().get("thumb_1024_url")
    return url if isinstance(url, str) and url else None


def download_image(
    session: requests.Session,
    limiter: RateLimiter,
    url: str,
    out_path: Path,
) -> bool:
    limiter.acquire()
    resp = session.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    if resp.status_code != 200:
        return False
    out_path.write_bytes(resp.content)
    return True


def _process_one(
    point_id: int,
    heading: int,
    image_id: str,
    out_dir: Path,
    session: requests.Session,
    limiter: RateLimiter,
    token: str,
    overwrite: bool,
) -> DownloadResult:
    out_path = out_dir / f"{point_id}_{heading}.jpg"
    if not overwrite and out_path.exists():
        return DownloadResult(point_id=point_id, heading=heading, image_id=image_id,
                              saved_path=out_path, ok=True, skipped=True)
    thumb_url = resolve_thumb_url(session, limiter, token, image_id)
    if thumb_url is None:
        return DownloadResult(point_id=point_id, heading=heading, image_id=image_id,
                              saved_path=out_path, ok=False, skipped=False,
                              error="thumb_url not found")
    ok = download_image(session, limiter, thumb_url, out_path)
    return DownloadResult(point_id=point_id, heading=heading, image_id=image_id,
                          saved_path=out_path, ok=ok, skipped=False,
                          error="" if ok else "download failed")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Mapillary heading images for WalkCLIP")
    ap.add_argument("--point-ids-file", required=True,
                    help="Text file with one lock point_id per line")
    ap.add_argument("--points-csv", required=True,
                    help="Points CSV with image_id_0/90/180/270 columns")
    ap.add_argument("--headings", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--output-dir", required=True, help="Directory to write {point_id}_{heading}.jpg")
    ap.add_argument("--token", default="",
                    help="Mapillary access token (or set MAPILLARY_ACCESS_TOKEN)")
    ap.add_argument("--rps", type=float, default=_DEFAULT_RPS)
    ap.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="Process first N points only (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(REPO_ROOT / ".env")

    token = args.token or os.environ.get("MAPILLARY_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing Mapillary token. Pass --token or set MAPILLARY_ACCESS_TOKEN.")

    lock_file = Path(args.point_ids_file)
    if not lock_file.exists():
        raise SystemExit(f"Lock IDs file not found: {lock_file}")
    lock_ids = {int(x) for x in lock_file.read_text().strip().splitlines()}

    points_csv = Path(args.points_csv)
    if not points_csv.exists():
        raise SystemExit(f"Points CSV not found: {points_csv}")

    with points_csv.open("r", newline="", encoding="utf-8") as f:
        all_rows: list[dict[str, Any]] = list(csv.DictReader(f))

    rows = [r for r in all_rows if int(r["point_id"]) in lock_ids]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build flat task list: (point_id, heading, image_id)
    tasks: list[tuple[int, int, str]] = []
    for row in rows:
        pid = int(row["point_id"])
        for h in args.headings:
            img_id = row.get(f"image_id_{h}", "").strip()
            if img_id:
                tasks.append((pid, h, img_id))

    print(
        f"[Mapillary] {len(rows)} points × {len(args.headings)} headings = {len(tasks)} tasks",
        flush=True,
    )

    limiter = RateLimiter(args.rps)
    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "s"):
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            thread_local.s = s
        return thread_local.s

    n_ok = n_skip = n_fail = 0
    completed = 0

    manifest_rows: list[dict] = []

    def process(task: tuple[int, int, str]) -> DownloadResult:
        pid, heading, img_id = task
        return _process_one(pid, heading, img_id, out_dir, get_session(), limiter, token, args.overwrite)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res.skipped:
                n_skip += 1
            elif res.ok:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  [FAIL] {res.point_id}/{res.heading}: {res.error}", flush=True)
            manifest_rows.append({
                "point_id": res.point_id, "heading": res.heading,
                "image_id": res.image_id, "ok": res.ok, "skipped": res.skipped,
                "path": str(res.saved_path),
            })
            if completed % args.log_every == 0:
                print(
                    f"  {completed}/{len(tasks)}  ok={n_ok}  skip={n_skip}  fail={n_fail}",
                    flush=True,
                )

    print(f"\n[Mapillary] Done. ok={n_ok}  skipped={n_skip}  failed={n_fail}", flush=True)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "heading", "image_id", "ok", "skipped", "path"])
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: (r["point_id"], r["heading"])))
    print(f"[Mapillary] Manifest: {manifest_path}", flush=True)

    fail_rate = n_fail / max(len(tasks), 1)
    if fail_rate > 0.10:
        print(f"WARNING: {n_fail}/{len(tasks)} downloads failed ({100*fail_rate:.1f}%).", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
