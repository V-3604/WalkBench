"""
Fetch the APE (Annotations for Pedestrian Environment) dataset (W4 D22b).

APE provides crowdsourced sidewalk/curb/obstacle annotations for Seattle and
some additional cities. It complements Project Sidewalk for our calibration
audit (paper §5.5/§6).

Source: github.com/ProjectSidewalk/sidewalk-equity-study mirrors PS labels
including the APE-derived Seattle subset. APE upstream is also at
github.com/ProjectSidewalk/SidewalkAPI (REST endpoint).

What this script does:
    1. Reads PS/APE label CSVs from the locally cloned sidewalk-equity-study
       repo at project/data/external/sidewalk-equity-study/.
    2. Optionally pulls fresh APE labels for Seattle and DC from the
       ProjectSidewalk REST API (--fetch-fresh).
    3. Standardises columns to:
           label_id, lat, lon, city, label_type, severity, source
       where source ∈ {project_sidewalk, ape}.
    4. Writes project/data/external/ape/ape_labels_combined.csv.

Run (Windows PowerShell):
    & ".venv\Scripts\python.exe" project/scripts/fetch_ape_dataset.py --cities seattle dc
    & ".venv\Scripts\python.exe" project/scripts/fetch_ape_dataset.py --cities seattle --fetch-fresh

No GPU; runs in <1 minute for cached files, ~5 min if --fetch-fresh.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
PS_REPO_DIR = REPO_ROOT / "project/data/external/sidewalk-equity-study"
OUT_DIR = REPO_ROOT / "project/data/external/ape"
OUT_CSV = OUT_DIR / "ape_labels_combined.csv"
OUT_JSON = OUT_DIR / "ape_labels_summary.json"

PS_API_BASE = "https://sidewalk-{city}.cs.washington.edu/v3/api/labels/csv"
SUPPORTED_CITIES = {
    "seattle": "seattle",
    "dc":      "washington-dc",
}

PS_LABEL_TYPES = {
    "CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem",
    "Crosswalk", "NoSidewalk", "Other",
}

CANONICAL_COLS = ["label_id", "lat", "lon", "city", "label_type", "severity", "source"]


def find_ps_csvs(repo_dir: Path) -> list[Path]:
    if not repo_dir.exists():
        return []
    csvs: list[Path] = []
    for ext in ("*.csv", "*.CSV"):
        csvs.extend(repo_dir.rglob(ext))
    return [p for p in csvs if "label" in p.name.lower() or "ape" in p.name.lower()]


def find_ps_geojson(repo_dir: Path) -> list[tuple[Path, str]]:
    """Find PS label GeoJSON files in the cloned sidewalk-equity-study repo.

    The repo stores per-city labels under {city}/datasets/01-project-sidewalk-labels/
    as `attributes_*.json` (FeatureCollection of Point features).
    Returns list of (path, city) tuples.
    """
    if not repo_dir.exists():
        return []
    out: list[tuple[Path, str]] = []
    for city_dir in repo_dir.iterdir():
        if not city_dir.is_dir():
            continue
        labels_dir = city_dir / "datasets" / "01-project-sidewalk-labels"
        if not labels_dir.exists():
            continue
        for path in labels_dir.glob("attributes_*.json"):
            out.append((path, city_dir.name.lower()))
    return out


def normalise_ps_geojson(path: Path, city: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", []) if isinstance(data, dict) else []
    rows = []
    for feat in features:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        rows.append({
            "label_id":   props.get("attribute_id") or props.get("label_id"),
            "lat":        float(coords[1]),
            "lon":        float(coords[0]),
            "city":       city,
            "label_type": props.get("label_type") or "Unknown",
            "severity":   props.get("severity"),
            "source":     "project_sidewalk",
        })
    return pd.DataFrame(rows, columns=CANONICAL_COLS)


def normalise_ps_csv(csv_path: Path, default_city: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    lat_col = pick("lat", "latitude")
    lon_col = pick("lng", "lon", "longitude")
    type_col = pick("label_type", "labeltype", "type")
    sev_col = pick("severity", "label_severity")
    city_col = pick("city", "city_name")
    id_col = pick("label_id", "id")
    if lat_col is None or lon_col is None:
        logging.warning("Skipping %s: no lat/lon columns", csv_path.name)
        return pd.DataFrame(columns=CANONICAL_COLS)

    out = pd.DataFrame({
        "label_id":   df[id_col] if id_col else range(len(df)),
        "lat":        df[lat_col],
        "lon":        df[lon_col],
        "city":       df[city_col].str.lower() if city_col else (default_city or "unknown"),
        "label_type": df[type_col] if type_col else "Unknown",
        "severity":   df[sev_col] if sev_col else None,
        "source":     "project_sidewalk" if "ape" not in csv_path.name.lower() else "ape",
    })
    out = out.dropna(subset=["lat", "lon"]).copy()
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    return out


def fetch_fresh_ps(city: str) -> pd.DataFrame:
    """Pull labels CSV directly from the live Project Sidewalk API for a city."""
    if city not in SUPPORTED_CITIES:
        logging.warning("Fresh fetch unsupported for city=%s", city)
        return pd.DataFrame(columns=CANONICAL_COLS)
    slug = SUPPORTED_CITIES[city]
    url = PS_API_BASE.format(city=slug)
    logging.info("[%s] GET %s", city, url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    tmp = OUT_DIR / f"_tmp_ps_{city}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(resp.content)
    df = normalise_ps_csv(tmp, default_city=city)
    df["city"] = city
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch / normalise PS+APE labels")
    p.add_argument("--cities", nargs="+", choices=["seattle", "dc"], required=True)
    p.add_argument("--fetch-fresh", action="store_true",
                   help="Fetch fresh labels from PS REST API instead of repo CSVs")
    p.add_argument("--output", type=Path, default=OUT_CSV)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    if args.fetch_fresh:
        for city in args.cities:
            try:
                frames.append(fetch_fresh_ps(city))
            except Exception as exc:
                logging.exception("[%s] fresh fetch failed: %s", city, exc)
    else:
        for csv in find_ps_csvs(PS_REPO_DIR):
            df = normalise_ps_csv(csv)
            if not df.empty:
                frames.append(df)
        for path, city in find_ps_geojson(PS_REPO_DIR):
            df = normalise_ps_geojson(path, city)
            if not df.empty:
                logging.info("Loaded %s rows from %s (city=%s)", len(df), path.name, city)
                frames.append(df)
        if not frames:
            logging.error(
                "No PS/APE label files (CSV or GeoJSON) found in %s. Either re-clone "
                "the sidewalk-equity-study repo or pass --fetch-fresh to pull from the "
                "live API.", PS_REPO_DIR,
            )
            return 1

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLS)
    combined = combined[combined["city"].isin(args.cities)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    logging.info("Wrote %s (rows=%d)", args.output, len(combined))

    summary = {
        "rows": int(len(combined)),
        "by_city": combined["city"].value_counts().to_dict() if len(combined) else {},
        "by_source": combined["source"].value_counts().to_dict() if len(combined) else {},
        "by_label_type": combined["label_type"].value_counts().head(10).to_dict() if len(combined) else {},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
