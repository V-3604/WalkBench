"""
Download Overture Maps 2026-03-18.0 transportation and buildings data for MSP and Seattle.

Downloads GeoParquet files filtered to city bounding boxes.
Requires: pip install overturemaps pyarrow

Run:
    python project/scripts/download_overture_data.py --city both
    python project/scripts/download_overture_data.py --city msp
    python project/scripts/download_overture_data.py --city seattle --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTS_DIR = REPO_ROOT / "project" / "data" / "processed" / "points"
OUT_DIR = REPO_ROOT / "project" / "data" / "raw" / "overture"

# Bboxes computed from the final lock point coordinates + 0.05° padding
CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    # (min_lon, min_lat, max_lon, max_lat)
    "msp": (-93.38, 44.84, -93.01, 45.11),
    "seattle": (-122.48, 47.43, -122.19, 47.78),
    "dc": (-77.17, 38.74, -76.86, 39.05),
    "pittsburgh": (-80.10, 40.36, -79.85, 40.50),
}

# Overture 2026-03-18.0 release types to download
OVERTURE_TYPES = ["segment", "building"]


def download_overture_type(
    overture_type: str,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    dry_run: bool,
) -> None:
    """Download one Overture type for a bbox using the overturemaps CLI."""
    try:
        import overturemaps
    except ImportError:
        raise ImportError(
            "overturemaps package not installed. Run: pip install overturemaps pyarrow"
        )

    if out_path.exists():
        logging.info("  Already exists, skipping: %s", out_path.name)
        return

    if dry_run:
        logging.info("  [dry-run] Would download %s → %s", overture_type, out_path.name)
        return

    logging.info("  Downloading %s for bbox %s ...", overture_type, bbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cli_path = Path(sys.executable).with_name("overturemaps.exe")
    if not cli_path.exists():
        cli_path = Path(sys.executable).with_name("overturemaps")
    if not cli_path.exists():
        raise RuntimeError(
            "Could not find overturemaps CLI next to current Python executable."
        )

    bbox_str = ",".join(str(v) for v in bbox)
    cmd = [
        str(cli_path),
        "download",
        f"--bbox={bbox_str}",
        "-f",
        "geoparquet",
        "--type",
        overture_type,
        "-o",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    import os
    size_mb = os.path.getsize(out_path) / 1e6
    logging.info("  Saved: %s (%.1f MB)", out_path.name, size_mb)


def summarize_file(path: Path) -> None:
    """Print basic stats about a downloaded GeoParquet file."""
    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(path)
        logging.info(
            "  %s: %d rows, columns: %s",
            path.name,
            len(gdf),
            ", ".join(gdf.columns[:8].tolist()),
        )
        if "subtype" in gdf.columns:
            logging.info("  subtype counts: %s", dict(gdf["subtype"].value_counts().head(10)))
        if "class" in gdf.columns:
            logging.info("  class counts: %s", dict(gdf["class"].value_counts().head(15)))
    except Exception as e:
        logging.warning("  Could not summarize %s: %s", path.name, e)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Overture Maps data for MSP and Seattle")
    p.add_argument("--city", choices=["msp", "seattle", "dc", "pittsburgh", "both", "all"], default="both")
    p.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    p.add_argument("--summarize", action="store_true", help="Print schema info for downloaded files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.city in ("both", "all"):
        cities = ["msp", "seattle", "dc"]
    else:
        cities = [args.city]

    for city in cities:
        bbox = CITY_BBOXES[city]
        logging.info("=== %s  bbox=%s ===", city.upper(), bbox)

        for otype in OVERTURE_TYPES:
            out_path = OUT_DIR / city / f"overture_{otype}_{city}.parquet"
            download_overture_type(otype, bbox, out_path, dry_run=args.dry_run)
            if args.summarize and out_path.exists():
                summarize_file(out_path)

    if not args.dry_run:
        logging.info("Done. Files in: %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
