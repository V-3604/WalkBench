"""
Audit the WalkCLIP v2 data lock and write a coverage Markdown report.

Reads all per-city lock files, checks imagery counts, caption coverage,
Overture label coverage, and embedding presence. Writes a report per city
plus a combined summary.

Run:
    & ".venv\Scripts\python.exe" project/scripts/audit_v2_lock.py
    & ".venv\Scripts\python.exe" project/scripts/audit_v2_lock.py --cities dc
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKS_DIR = REPO_ROOT / "project/data/processed/locks"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
IMG_ROOT = REPO_ROOT / "project/data/raw/imagery"
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
LABELS_DIR = REPO_ROOT / "project/data/processed/labels"
TEXT_DIR = REPO_ROOT / "project/data/processed/text"
REPORT_DIR = REPO_ROOT / "project/artifacts/reports"

CITY_META = {
    "msp": {
        "label": "Minneapolis–Saint Paul",
        "lock_file": "msp_v2_final_lock_ids.txt",
        "lock_file_fallback": "v2_final_lock_ids.txt",
        "points_lock_csv": "msp_points_v2_final_lock.csv",
        "street_dir": IMG_ROOT / "street_view_mapillary/Minneapolis",
        "naip_dir": IMG_ROOT / "aerial_naip/Minneapolis",
        "street_embed": "siglip_street_msp.npy",
        "naip_embed": "siglip_naip_msp.npy",
        "caption_jsonl": TEXT_DIR / "streetview_openai_msp_final_lock.jsonl",
    },
    "seattle": {
        "label": "Seattle",
        "lock_file": "seattle_v2_final_lock_ids.txt",
        "lock_file_fallback": "v2_final_lock_ids.txt",
        "points_lock_csv": "seattle_points_v2_final_lock.csv",
        "street_dir": IMG_ROOT / "street_view_mapillary/Seattle",
        "naip_dir": IMG_ROOT / "aerial_naip/Seattle",
        "street_embed": "siglip_street_seattle.npy",
        "naip_embed": "siglip_naip_seattle.npy",
        "caption_jsonl": TEXT_DIR / "streetview_openai_seattle_final_lock.jsonl",
    },
    "dc": {
        "label": "Washington DC",
        "lock_file": "dc_v2_final_lock_ids.txt",
        "lock_file_fallback": None,
        "points_lock_csv": "dc_points_v2_final_lock.csv",
        "street_dir": IMG_ROOT / "street_view_mapillary/DC",
        "naip_dir": IMG_ROOT / "aerial_naip/DC",
        "street_embed": "siglip_street_dc.npy",
        "naip_embed": "siglip_naip_dc.npy",
        "caption_jsonl": TEXT_DIR / "streetview_openai_dc_final_lock.jsonl",
    },
    "pittsburgh": {
        "label": "Pittsburgh",
        "lock_file": "pittsburgh_v2_final_lock_ids.txt",
        "lock_file_fallback": None,
        "points_lock_csv": "pittsburgh_points_v2_final_lock.csv",
        "street_dir": IMG_ROOT / "street_view_mapillary/Pittsburgh",
        "naip_dir": IMG_ROOT / "aerial_naip/Pittsburgh",
        "street_embed": "siglip_street_pittsburgh.npy",
        "naip_embed": "siglip_naip_pittsburgh.npy",
        "caption_jsonl": TEXT_DIR / "streetview_openai_pittsburgh_final_lock.jsonl",
    },
}


def load_lock_ids(city: str) -> list[int]:
    meta = CITY_META[city]
    for fname in [meta["lock_file"], meta.get("lock_file_fallback")]:
        if fname is None:
            continue
        p = LOCKS_DIR / fname
        if p.exists():
            return [int(x) for x in p.read_text().strip().splitlines()]
    return []


def audit_city(city: str) -> dict:
    meta = CITY_META[city]
    lock_ids = load_lock_ids(city)
    n_lock = len(lock_ids)
    lock_set = set(lock_ids)

    result: dict = {"city": city, "label": meta["label"], "lock_size": n_lock}

    # Mapillary images
    street_dir: Path = meta["street_dir"]
    if street_dir.exists():
        jpgs = list(street_dir.glob("*.jpg"))
        result["street_images_total"] = len(jpgs)
        # Count complete points (4 headings)
        from collections import defaultdict
        seen: dict[int, set] = defaultdict(set)
        for j in jpgs:
            if "_" in j.stem:
                parts = j.stem.split("_", 1)
                if parts[0].isdigit() and parts[1].isdigit():
                    seen[int(parts[0])].add(int(parts[1]))
        complete = sum(1 for pid in lock_set if seen.get(pid, set()) == {0, 90, 180, 270})
        result["street_complete_points"] = complete
        result["street_complete_pct"] = round(100 * complete / n_lock, 1) if n_lock else 0
    else:
        result["street_images_total"] = 0
        result["street_complete_points"] = 0
        result["street_complete_pct"] = 0.0

    # NAIP
    naip_dir: Path = meta["naip_dir"]
    if naip_dir.exists():
        naip_ids = {int(p.stem) for p in naip_dir.glob("*.jpg") if p.stem.isdigit()}
        naip_covered = len(naip_ids & lock_set)
        result["naip_covered"] = naip_covered
        result["naip_pct"] = round(100 * naip_covered / n_lock, 1) if n_lock else 0
    else:
        result["naip_covered"] = 0
        result["naip_pct"] = 0.0

    # Embeddings
    result["street_embed_exists"] = (EMBED_DIR / meta["street_embed"]).exists()
    result["naip_embed_exists"] = (EMBED_DIR / meta["naip_embed"]).exists()

    if result["street_embed_exists"]:
        arr = np.load(EMBED_DIR / meta["street_embed"])
        result["street_embed_shape"] = list(arr.shape)

    # Captions
    cap_path: Path = meta["caption_jsonl"]
    if cap_path.exists():
        n_captions = sum(1 for _ in cap_path.open("r", encoding="utf-8"))
        result["captions_present"] = n_captions
        result["captions_pct"] = round(100 * n_captions / n_lock, 1) if n_lock else 0
    else:
        result["captions_present"] = 0
        result["captions_pct"] = 0.0

    # Overture labels
    overture_csv = LABELS_DIR / "overture_targets.csv"
    if overture_csv.exists():
        df = pd.read_csv(overture_csv)
        city_rows = df[df["city"] == city]
        covered_overture = len(set(city_rows["point_id"].astype(int)) & lock_set)
        result["overture_labels_covered"] = covered_overture
        result["overture_labels_pct"] = round(100 * covered_overture / n_lock, 1) if n_lock else 0
    else:
        result["overture_labels_covered"] = 0
        result["overture_labels_pct"] = 0.0

    return result


def write_markdown_report(audits: list[dict], out_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# WalkCLIP v2 Lock Audit — {ts}",
        "",
        "## Summary",
        "",
        "| City | Lock size | Street 4/4 | NAIP | Street embed | NAIP embed | Captions | Overture |",
        "|------|----------:|----------:|-----:|:------------:|:----------:|---------:|---------:|",
    ]
    for a in audits:
        lines.append(
            f"| {a['label']} "
            f"| {a['lock_size']:,} "
            f"| {a['street_complete_points']:,} ({a['street_complete_pct']}%) "
            f"| {a['naip_covered']:,} ({a['naip_pct']}%) "
            f"| {'Y' if a['street_embed_exists'] else 'N'} "
            f"| {'Y' if a['naip_embed_exists'] else 'N'} "
            f"| {a['captions_present']:,} ({a['captions_pct']}%) "
            f"| {a['overture_labels_covered']:,} ({a['overture_labels_pct']}%) |"
        )

    lines += ["", "## Per-city detail", ""]
    for a in audits:
        lines += [
            f"### {a['label']} (`{a['city']}`)",
            "",
            f"- Lock size: **{a['lock_size']:,}**",
            f"- Mapillary complete: {a['street_complete_points']:,} ({a['street_complete_pct']}%)",
            f"- NAIP covered: {a['naip_covered']:,} ({a['naip_pct']}%)",
            f"- Street embedding: {'present' if a['street_embed_exists'] else 'MISSING'}",
            f"- NAIP embedding: {'present' if a['naip_embed_exists'] else 'MISSING'}",
            f"- Captions: {a['captions_present']:,} ({a['captions_pct']}%)",
            f"- Overture labels: {a['overture_labels_covered']:,} ({a['overture_labels_pct']}%)",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit WalkCLIP v2 data lock coverage")
    ap.add_argument("--cities", nargs="+", choices=list(CITY_META), default=list(CITY_META))
    ap.add_argument("--report-path", default="",
                    help="Override report output path (default: project/artifacts/reports/dc_v2_coverage.md)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audits: list[dict] = []
    for city in args.cities:
        print(f"\nAuditing {city.upper()} …", flush=True)
        a = audit_city(city)
        audits.append(a)
        print(
            f"  lock={a['lock_size']}  street_complete={a['street_complete_points']}  "
            f"naip={a['naip_covered']}  captions={a['captions_present']}  "
            f"overture={a['overture_labels_covered']}",
            flush=True,
        )

    # Write per-city report
    if len(args.cities) == 1:
        city = args.cities[0]
        default_name = f"{city}_v2_coverage.md"
    else:
        default_name = "v2_3city_coverage.md"

    report_path = Path(args.report_path) if args.report_path else REPORT_DIR / default_name
    write_markdown_report(audits, report_path)

    # JSON dump for CI/scripting
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(audits, indent=2))
    print(f"JSON: {json_path}", flush=True)

    # Exit non-zero if any city is <50% complete on any critical dimension
    critical_fail = any(
        a["lock_size"] == 0
        or a["street_complete_pct"] < 80
        for a in audits
    )
    if critical_fail:
        print("\nWARNING: One or more cities below 80% Mapillary completeness.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
