"""
Build final multimodal data locks for WalkCLIP v2 (3-city version).

For each included city, scans on-disk imagery and finds point_ids with:
  - 4/4 Mapillary headings (0, 90, 180, 270)
  - 1 NAIP tile

Outputs per city:
  project/data/processed/locks/{city}_v2_final_lock_ids.txt
  project/data/processed/points/{city}_points_v2_final_lock.csv

Plus backward-compat combined file (MSP + Seattle shared IDs):
  project/data/processed/locks/v2_final_lock_ids.txt  (union for MSP/Seattle, same IDs 0-4823)

Summary:
  project/data/processed/locks/v2_3city_lock_summary.json

Run:
    & ".venv\Scripts\python.exe" project/scripts/build_v2_final_lock.py --include-cities msp seattle dc
    & ".venv\Scripts\python.exe" project/scripts/build_v2_final_lock.py --include-cities dc
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = REPO_ROOT / "project/data/processed/locks"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
IMG_ROOT = REPO_ROOT / "project/data/raw/imagery"

HEADINGS = frozenset({0, 90, 180, 270})

CITY_STREET_DIRS: dict[str, Path] = {
    "msp":         IMG_ROOT / "street_view_mapillary/Minneapolis",
    "seattle":     IMG_ROOT / "street_view_mapillary/Seattle",
    "dc":          IMG_ROOT / "street_view_mapillary/DC",
    "pittsburgh":  IMG_ROOT / "street_view_mapillary/Pittsburgh",
}
CITY_NAIP_DIRS: dict[str, Path] = {
    "msp":         IMG_ROOT / "aerial_naip/Minneapolis",
    "seattle":     IMG_ROOT / "aerial_naip/Seattle",
    "dc":          IMG_ROOT / "aerial_naip/DC",
    "pittsburgh":  IMG_ROOT / "aerial_naip/Pittsburgh",
}
CITY_POINTS_CSV: dict[str, Path] = {
    "msp":         POINTS_DIR / "msp_points_v2.csv",
    "seattle":     POINTS_DIR / "seattle_points_v2.csv",
    "dc":          POINTS_DIR / "dc_points_v2.csv",
    "pittsburgh":  POINTS_DIR / "pittsburgh_points_v2.csv",
}


def scan_street_dir(folder: Path) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    if not folder.exists():
        return seen
    for img in folder.glob("*.jpg"):
        stem = img.stem
        if "_" not in stem:
            continue
        parts = stem.split("_", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            seen[int(parts[0])].add(int(parts[1]))
    return seen


def scan_naip_dir(folder: Path) -> set[int]:
    if not folder.exists():
        return set()
    return {int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()}


def build_city_lock(city: str) -> dict:
    points_csv = CITY_POINTS_CSV[city]
    if not points_csv.exists():
        return {"error": f"Points CSV not found: {points_csv}"}

    points_df = pd.read_csv(points_csv)
    points_df["point_id"] = points_df["point_id"].astype(int)
    all_ids: set[int] = set(points_df["point_id"])

    street_seen = scan_street_dir(CITY_STREET_DIRS[city])
    naip_seen = scan_naip_dir(CITY_NAIP_DIRS[city])

    complete_street = {pid for pid in all_ids if street_seen.get(pid, set()) == HEADINGS}
    partial_street = {pid for pid in all_ids if 0 < len(street_seen.get(pid, set())) < 4}
    missing_street = {pid for pid in all_ids if not street_seen.get(pid)}

    locked = complete_street & naip_seen
    missing_naip = complete_street - naip_seen

    return {
        "city": city,
        "total_points": len(all_ids),
        "complete_street": len(complete_street),
        "partial_street": len(partial_street),
        "missing_street": len(missing_street),
        "naip_present": len(naip_seen & all_ids),
        "missing_naip_from_complete_street": len(missing_naip),
        "lock_size": len(locked),
        "locked_ids": sorted(locked),
    }


def write_lock_files(city: str, lock_result: dict) -> None:
    if "error" in lock_result:
        print(f"  [{city}] SKIP: {lock_result['error']}", flush=True)
        return

    locked_ids: list[int] = lock_result["locked_ids"]
    lock_path = LOCKS_DIR / f"{city}_v2_final_lock_ids.txt"
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("\n".join(str(i) for i in locked_ids) + "\n")
    print(f"  [{city}] Lock: {len(locked_ids)} IDs -> {lock_path}", flush=True)

    points_csv = CITY_POINTS_CSV[city]
    if points_csv.exists():
        points_df = pd.read_csv(points_csv)
        points_df["point_id"] = points_df["point_id"].astype(int)
        locked_set = set(locked_ids)
        lock_points = points_df[points_df["point_id"].isin(locked_set)].copy()
        out_csv = POINTS_DIR / f"{city}_points_v2_final_lock.csv"
        lock_points.to_csv(out_csv, index=False)
        print(f"  [{city}] Final-lock points CSV: {out_csv} ({len(lock_points)} rows)", flush=True)


def write_legacy_lock(msp_result: dict, seattle_result: dict) -> None:
    """Write the backward-compat v2_final_lock_ids.txt with shared IDs from MSP+Seattle.

    The existing 2-city lock uses IDs 0-4823 that are the same in both cities.
    Only update if either city is newly re-locked (non-empty).
    """
    if "error" in msp_result or "error" in seattle_result:
        return
    msp_ids = set(msp_result["locked_ids"])
    sea_ids = set(seattle_result["locked_ids"])
    shared = sorted(msp_ids & sea_ids)
    if not shared:
        print("  [legacy] No shared MSP/Seattle IDs — legacy lock not updated.", flush=True)
        return
    legacy_path = LOCKS_DIR / "v2_final_lock_ids.txt"
    legacy_path.write_text("\n".join(str(i) for i in shared) + "\n")
    print(f"  [legacy] v2_final_lock_ids.txt: {len(shared)} shared IDs -> {legacy_path}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build final multimodal lock for 1–3 WalkCLIP cities")
    ap.add_argument("--include-cities", nargs="+", choices=["msp", "seattle", "dc", "pittsburgh"],
                    default=["msp", "seattle", "dc"])
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing lock files")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    summary: dict[str, dict] = {}
    for city in args.include_cities:
        print(f"\n[{city.upper()}]", flush=True)
        result = build_city_lock(city)
        summary[city] = {k: v for k, v in result.items() if k != "locked_ids"}
        print(
            f"  total={result.get('total_points', '?')}  "
            f"complete_street={result.get('complete_street', '?')}  "
            f"naip={result.get('naip_present', '?')}  "
            f"LOCK={result.get('lock_size', '?')}",
            flush=True,
        )
        write_lock_files(city, result)

    # Update legacy lock if both MSP and Seattle were processed
    if "msp" in args.include_cities and "seattle" in args.include_cities:
        msp_r = build_city_lock("msp")
        sea_r = build_city_lock("seattle")
        write_legacy_lock(msp_r, sea_r)

    summary_path = LOCKS_DIR / "v2_3city_lock_summary.json"
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_summary.update(summary)
        summary = existing_summary
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Summary] {summary_path}", flush=True)

    for city, s in summary.items():
        if "error" in s:
            print(f"  {city}: ERROR — {s['error']}")
        else:
            print(f"  {city}: lock_size={s.get('lock_size', '?')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
