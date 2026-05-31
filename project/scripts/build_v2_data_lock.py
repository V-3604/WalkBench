"""
Build the final shared WalkCLIP v2 data lock from on-disk imagery.

Research rule:
- keep the raw 4,874-row point tables unchanged
- derive a final lock as the shared `point_id` subset that is complete in BOTH cities
- require 4/4 Mapillary headings and 1 NAIP tile per point in both cities
- avoid arbitrary downsampling of Seattle-only extra complete points

Outputs:
- project/data/processed/locks/v2_final_lock_ids.txt
- project/data/processed/locks/v2_final_lock_manifest.csv
- project/data/processed/locks/v2_final_lock_summary.json
- project/data/processed/points/msp_points_v2_final_lock.csv
- project/data/processed/points/seattle_points_v2_final_lock.csv
- project/data/processed/text/streetview_openai_seattle_final_lock.jsonl
- project/data/processed/text/streetview_openai_seattle_final_lock.csv

Run from repo root:
  & ".venv\\Scripts\\python.exe" project\\scripts\\build_v2_data_lock.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


HEADINGS = {0, 90, 180, 270}


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "project").exists():
            return candidate
    raise RuntimeError("Could not locate repo root (expected a 'project/' folder).")


def load_points(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "point_id" not in df.columns:
        raise ValueError(f"Missing point_id column in {csv_path}")
    df["point_id"] = df["point_id"].astype(int)
    if df["point_id"].duplicated().any():
        raise ValueError(f"Duplicate point_id values in {csv_path}")
    return df.sort_values("point_id").reset_index(drop=True)


def scan_street_dir(folder: Path) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for image_path in folder.glob("*.jpg"):
        stem = image_path.stem
        if "_" not in stem:
            continue
        point_str, heading_str = stem.split("_", 1)
        if not point_str.isdigit() or not heading_str.isdigit():
            continue
        seen[int(point_str)].add(int(heading_str))
    return seen


def scan_naip_dir(folder: Path) -> set[int]:
    return {int(image_path.stem) for image_path in folder.glob("*.jpg") if image_path.stem.isdigit()}


def street_status(expected_ids: set[int], folder: Path) -> tuple[set[int], set[int], set[int], dict[int, set[int]]]:
    seen = scan_street_dir(folder)
    complete = {point_id for point_id in expected_ids if seen.get(point_id, set()) == HEADINGS}
    partial = {point_id for point_id in expected_ids if 0 < len(seen.get(point_id, set())) < len(HEADINGS)}
    none = {point_id for point_id in expected_ids if not seen.get(point_id, set())}
    return complete, partial, none, seen


def parse_image_point_id(image_name: str) -> int | None:
    image_name = str(image_name).strip()
    if not image_name.endswith("_combined.jpg"):
        return None
    point_str = image_name.split("_", 1)[0]
    return int(point_str) if point_str.isdigit() else None


def filter_caption_csv(src: Path, dst: Path, keep_ids: set[int]) -> int:
    if not src.exists():
        return 0
    df = pd.read_csv(src)
    if "image" not in df.columns:
        raise ValueError(f"Expected an image column in {src}")
    df["point_id"] = df["image"].map(parse_image_point_id)
    filtered = (
        df[df["point_id"].isin(keep_ids)]
        .drop(columns=["point_id"])
        .sort_values("image")
        .reset_index(drop=True)
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dst, index=False)
    return len(filtered)


def filter_caption_jsonl(src: Path, dst: Path, keep_ids: set[int]) -> int:
    if not src.exists():
        return 0
    kept_lines: list[str] = []
    with src.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            point_id = record.get("id")
            if isinstance(point_id, int) and point_id in keep_ids:
                kept_lines.append(json.dumps(record, ensure_ascii=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as outfile:
        if kept_lines:
            outfile.write("\n".join(kept_lines) + "\n")
    return len(kept_lines)


def write_point_subset(src_df: pd.DataFrame, dst: Path, keep_ids: set[int]) -> int:
    filtered = src_df[src_df["point_id"].isin(keep_ids)].sort_values("point_id").reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dst, index=False)
    return len(filtered)


def main() -> int:
    root = repo_root()

    msp_points_path = root / "project/data/processed/points/msp_points_v2.csv"
    sea_points_path = root / "project/data/processed/points/seattle_points_v2.csv"
    msp_street_dir = root / "project/data/raw/imagery/street_view_mapillary/Minneapolis"
    sea_street_dir = root / "project/data/raw/imagery/street_view_mapillary/Seattle"
    msp_naip_dir = root / "project/data/raw/imagery/aerial_naip/Minneapolis"
    sea_naip_dir = root / "project/data/raw/imagery/aerial_naip/Seattle"

    locks_dir = root / "project/data/processed/locks"
    lock_ids_path = locks_dir / "v2_final_lock_ids.txt"
    lock_manifest_path = locks_dir / "v2_final_lock_manifest.csv"
    lock_summary_path = locks_dir / "v2_final_lock_summary.json"
    msp_lock_points_path = root / "project/data/processed/points/msp_points_v2_final_lock.csv"
    sea_lock_points_path = root / "project/data/processed/points/seattle_points_v2_final_lock.csv"
    sea_lock_caption_csv = root / "project/data/processed/text/streetview_openai_seattle_final_lock.csv"
    sea_lock_caption_jsonl = root / "project/data/processed/text/streetview_openai_seattle_final_lock.jsonl"

    msp_points = load_points(msp_points_path)
    sea_points = load_points(sea_points_path)
    msp_ids = set(msp_points["point_id"])
    sea_ids = set(sea_points["point_id"])
    if msp_ids != sea_ids:
        raise ValueError("MSP and Seattle point_id universes differ; cannot build a shared lock.")

    point_ids = sorted(msp_ids)

    msp_street_complete, msp_street_partial, msp_street_none, msp_street_seen = street_status(msp_ids, msp_street_dir)
    sea_street_complete, sea_street_partial, sea_street_none, sea_street_seen = street_status(sea_ids, sea_street_dir)

    msp_naip_ids = scan_naip_dir(msp_naip_dir)
    sea_naip_ids = scan_naip_dir(sea_naip_dir)

    msp_multimodal = msp_street_complete & msp_naip_ids
    sea_multimodal = sea_street_complete & sea_naip_ids
    shared_lock_ids = sorted(msp_multimodal & sea_multimodal)
    shared_lock_set = set(shared_lock_ids)

    manifest_rows: list[dict[str, object]] = []
    for point_id in point_ids:
        reasons: list[str] = []
        if point_id not in msp_street_complete:
            reasons.append("msp_street_incomplete")
        if point_id not in msp_naip_ids:
            reasons.append("msp_naip_missing")
        if point_id not in sea_street_complete:
            reasons.append("seattle_street_incomplete")
        if point_id not in sea_naip_ids:
            reasons.append("seattle_naip_missing")

        manifest_rows.append(
            {
                "point_id": point_id,
                "msp_street_heading_count": len(msp_street_seen.get(point_id, set())),
                "msp_street_complete": point_id in msp_street_complete,
                "msp_naip_present": point_id in msp_naip_ids,
                "seattle_street_heading_count": len(sea_street_seen.get(point_id, set())),
                "seattle_street_complete": point_id in sea_street_complete,
                "seattle_naip_present": point_id in sea_naip_ids,
                "in_final_lock": point_id in shared_lock_set,
                "drop_reasons": "|".join(reasons),
            }
        )

    locks_dir.mkdir(parents=True, exist_ok=True)
    with lock_ids_path.open("w", encoding="utf-8", newline="\n") as outfile:
        if shared_lock_ids:
            outfile.write("\n".join(str(point_id) for point_id in shared_lock_ids) + "\n")

    pd.DataFrame(manifest_rows).to_csv(lock_manifest_path, index=False)

    msp_rows_written = write_point_subset(msp_points, msp_lock_points_path, shared_lock_set)
    sea_rows_written = write_point_subset(sea_points, sea_lock_points_path, shared_lock_set)

    seattle_caption_rows_csv = filter_caption_csv(
        root / "project/data/processed/text/streetview_openai_seattle.csv",
        sea_lock_caption_csv,
        shared_lock_set,
    )
    seattle_caption_rows_jsonl = filter_caption_jsonl(
        root / "project/data/processed/text/streetview_openai_seattle.jsonl",
        sea_lock_caption_jsonl,
        shared_lock_set,
    )

    summary = {
        "lock_date": "2026-04-22",
        "lock_strategy": "shared_complete_multimodal_intersection",
        "why_this_strategy": (
            "Maximal non-arbitrary shared subset: every retained point_id has 4/4 Mapillary "
            "headings and a NAIP tile in both cities, without randomly discarding extra Seattle-complete rows."
        ),
        "raw_point_tables": {
            "msp": {"path": str(msp_points_path.relative_to(root)).replace("\\", "/"), "rows": len(msp_points)},
            "seattle": {"path": str(sea_points_path.relative_to(root)).replace("\\", "/"), "rows": len(sea_points)},
        },
        "availability": {
            "msp": {
                "street_complete": len(msp_street_complete),
                "street_partial": len(msp_street_partial),
                "street_none": len(msp_street_none),
                "naip_present": len(msp_naip_ids),
                "multimodal_complete": len(msp_multimodal),
            },
            "seattle": {
                "street_complete": len(sea_street_complete),
                "street_partial": len(sea_street_partial),
                "street_none": len(sea_street_none),
                "naip_present": len(sea_naip_ids),
                "multimodal_complete": len(sea_multimodal),
            },
        },
        "lock_sizes": {
            "shared_final_lock_per_city": len(shared_lock_ids),
            "joint_final_lock_total": len(shared_lock_ids) * 2,
            "equal_n_if_city_subsets_were_independent": min(len(msp_multimodal), len(sea_multimodal)),
        },
        "excluded_ids": {
            "msp_street_incomplete": sorted(msp_ids - msp_street_complete),
            "seattle_street_incomplete": sorted(sea_ids - sea_street_complete),
            "seattle_naip_missing": sorted(sea_ids - sea_naip_ids),
        },
        "outputs": {
            "lock_ids": str(lock_ids_path.relative_to(root)).replace("\\", "/"),
            "lock_manifest": str(lock_manifest_path.relative_to(root)).replace("\\", "/"),
            "lock_summary": str(lock_summary_path.relative_to(root)).replace("\\", "/"),
            "msp_points_final_lock": str(msp_lock_points_path.relative_to(root)).replace("\\", "/"),
            "seattle_points_final_lock": str(sea_lock_points_path.relative_to(root)).replace("\\", "/"),
            "seattle_captions_final_lock_csv": str(sea_lock_caption_csv.relative_to(root)).replace("\\", "/"),
            "seattle_captions_final_lock_jsonl": str(sea_lock_caption_jsonl.relative_to(root)).replace("\\", "/"),
        },
        "caption_provenance": {
            "seattle_existing_captions_filtered_to_lock": {
                "csv_rows": seattle_caption_rows_csv,
                "jsonl_rows": seattle_caption_rows_jsonl,
            },
            "msp_existing_caption_files_reused": False,
            "msp_note": (
                "Do not treat the existing MSP caption files as final v2 lock artifacts after retiring OLD_MSP_data. "
                "Re-run project/scripts/generate_openai_structured_captions.py against the current Minneapolis folder "
                "using project/data/processed/locks/v2_final_lock_ids.txt."
            ),
        },
    }

    with lock_summary_path.open("w", encoding="utf-8", newline="\n") as outfile:
        json.dump(summary, outfile, indent=2)
        outfile.write("\n")

    print("WalkCLIP v2 final shared lock")
    print(f"  MSP multimodal complete:      {len(msp_multimodal)}")
    print(f"  Seattle multimodal complete:  {len(sea_multimodal)}")
    print(f"  Final shared lock per city:   {len(shared_lock_ids)}")
    print(f"  Joint total:                  {len(shared_lock_ids) * 2}")
    print(f"  Equal-N if city subsets were independent: {min(len(msp_multimodal), len(sea_multimodal))}")
    print()
    print("Dropped point_id groups")
    print(f"  MSP street incomplete:        {sorted(msp_ids - msp_street_complete)}")
    print(f"  Seattle street incomplete:    {sorted(sea_ids - sea_street_complete)}")
    print(f"  Seattle NAIP missing:         {sorted(sea_ids - sea_naip_ids)}")
    print()
    print("Wrote")
    print(f"  {lock_ids_path.relative_to(root)}")
    print(f"  {lock_manifest_path.relative_to(root)}")
    print(f"  {lock_summary_path.relative_to(root)}")
    print(f"  {msp_lock_points_path.relative_to(root)}")
    print(f"  {sea_lock_points_path.relative_to(root)}")
    if seattle_caption_rows_csv:
        print(f"  {sea_lock_caption_csv.relative_to(root)}")
    if seattle_caption_rows_jsonl:
        print(f"  {sea_lock_caption_jsonl.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
