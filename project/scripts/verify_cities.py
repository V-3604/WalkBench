"""
Verify EPA SLD v3 / CDC PLACES 2025 / GTFS data availability for WalkCLIP cities.

For each configured city, confirms:
  - EPA SLD v3 has block groups for that MSA (row count + NatWalkInd range)
  - CDC PLACES 2025 has census tracts for that state FIPS prefix
  - GTFS feed directory is non-empty and has required files

Add new cities by extending CITY_CONFIGS.

Run:
    & ".venv\Scripts\python.exe" project/scripts/verify_cities.py
    & ".venv\Scripts\python.exe" project/scripts/verify_cities.py --cities DC
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EPA_GDB = REPO_ROOT / "project/data/raw/labels/SmartLocationDatabase.gdb"
EPA_LAYER = "EPA_SLD_Database_V3"
PLACES_CSV_DIR = REPO_ROOT / "project/data/raw/labels/cdc_places"
GTFS_ROOT = REPO_ROOT / "project/data/raw/labels/gtfs"

REQUIRED_GTFS_FILES = {"stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}


@dataclass
class CityConfig:
    name: str
    epa_cbsa_name_fragment: str  # substring to match CBSA_Name in EPA SLD
    places_state_abbr: str       # 2-letter state abbreviation (or 'DC')
    places_state_fips: str       # 2-digit FIPS prefix for census tract IDs
    gtfs_dir: str                # relative to GTFS_ROOT
    min_block_groups: int = 100  # expected lower bound


CITY_CONFIGS: dict[str, CityConfig] = {
    "MSP": CityConfig(
        name="Minneapolis–Saint Paul",
        epa_cbsa_name_fragment="Minneapolis",
        places_state_abbr="MN",
        places_state_fips="27",
        gtfs_dir="msp",
        min_block_groups=800,
    ),
    "Seattle": CityConfig(
        name="Seattle",
        epa_cbsa_name_fragment="Seattle",
        places_state_abbr="WA",
        places_state_fips="53",
        gtfs_dir="seattle",
        min_block_groups=800,
    ),
    "DC": CityConfig(
        name="Washington DC",
        epa_cbsa_name_fragment="Washington-Arlington-Alexandria",
        places_state_abbr="DC",
        places_state_fips="11",
        gtfs_dir="dc",
        min_block_groups=450,
    ),
    "Boston": CityConfig(
        name="Boston",
        epa_cbsa_name_fragment="Boston",
        places_state_abbr="MA",
        places_state_fips="25",
        gtfs_dir="boston",
        min_block_groups=800,
    ),
}


@dataclass
class VerifyResult:
    city: str
    epa_ok: bool
    epa_rows: int
    epa_natwalkind_range: tuple[float, float]
    epa_note: str
    places_ok: bool
    places_tracts: int
    places_note: str
    gtfs_ok: bool
    gtfs_note: str
    overall: bool = field(init=False)

    def __post_init__(self) -> None:
        self.overall = self.epa_ok and self.places_ok and self.gtfs_ok


def verify_epa(cfg: CityConfig) -> tuple[bool, int, tuple[float, float], str]:
    if not EPA_GDB.exists():
        return False, 0, (0.0, 0.0), f"EPA GDB not found: {EPA_GDB}"
    try:
        df = gpd.read_file(EPA_GDB, layer=EPA_LAYER)
    except Exception as exc:
        return False, 0, (0.0, 0.0), f"Failed to read EPA GDB: {exc}"

    cbsa_col = next((c for c in df.columns if "CBSA" in c.upper() and "NAME" in c.upper()), None)
    if cbsa_col is None:
        cbsa_col = next((c for c in df.columns if "CSA_Name" in c or "CBSA" in c), None)

    if cbsa_col is None:
        # Fall back: count everything with NatWalkInd in range, use GEOID prefix
        fips = cfg.places_state_fips
        geoid_col = next((c for c in df.columns if "GEOID" in c.upper()), None)
        if geoid_col:
            mask = df[geoid_col].astype(str).str.startswith(fips)
        else:
            mask = pd.Series([True] * len(df))
        subset = df[mask]
        note = f"No CBSA_Name column; filtered by GEOID prefix '{fips}'"
    else:
        mask = df[cbsa_col].fillna("").str.contains(cfg.epa_cbsa_name_fragment, case=False)
        subset = df[mask]
        note = f"Matched CBSA_Name fragment '{cfg.epa_cbsa_name_fragment}'"

    n = len(subset)
    if n < cfg.min_block_groups:
        return False, n, (0.0, 0.0), f"{note}; only {n} rows (expected >= {cfg.min_block_groups})"

    if "NatWalkInd" in subset.columns:
        nwi = subset["NatWalkInd"].dropna()
        rng = (float(nwi.min()), float(nwi.max()))
    else:
        rng = (0.0, 0.0)

    return True, n, rng, note


def verify_places(cfg: CityConfig) -> tuple[bool, int, str]:
    csv_files = list(PLACES_CSV_DIR.glob("*.csv")) if PLACES_CSV_DIR.exists() else []
    if not csv_files:
        return False, 0, f"No CDC PLACES CSV found in {PLACES_CSV_DIR}"

    places_csv = csv_files[0]
    try:
        df = pd.read_csv(places_csv, dtype=str, usecols=lambda c: c in (
            "StateAbbr", "StateDesc", "LocationName", "TractFIPS", "CountyFIPS",
        ))
    except Exception as exc:
        return False, 0, f"Failed to read {places_csv.name}: {exc}"

    state_col = next((c for c in ("StateAbbr", "StateDesc") if c in df.columns), None)
    if state_col == "StateAbbr":
        mask = df["StateAbbr"].str.upper() == cfg.places_state_abbr.upper()
    elif "TractFIPS" in df.columns:
        mask = df["TractFIPS"].astype(str).str.startswith(cfg.places_state_fips)
    else:
        return False, 0, "Could not identify state/tract column in PLACES CSV"

    subset = df[mask]
    n = len(subset)
    if n == 0:
        return False, 0, f"No tracts found for state '{cfg.places_state_abbr}' in {places_csv.name}"
    return True, n, f"Found {n} tracts in {places_csv.name}"


def verify_gtfs(cfg: CityConfig) -> tuple[bool, str]:
    gtfs_dir = GTFS_ROOT / cfg.gtfs_dir
    if not gtfs_dir.exists():
        return False, f"GTFS directory not found: {gtfs_dir}\nDownload WMATA/agency GTFS and extract to that path."

    # Check both raw .txt files and a single zip
    files_present: set[str] = set()

    # Direct .txt files
    for f in gtfs_dir.glob("*.txt"):
        files_present.add(f.name.lower())

    # Or inside a zip
    for z in gtfs_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    files_present.add(Path(name).name.lower())
        except Exception:
            pass

    missing = REQUIRED_GTFS_FILES - files_present
    if missing:
        return False, f"Missing GTFS files in {gtfs_dir}: {sorted(missing)}"
    return True, f"All required GTFS files present in {gtfs_dir}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify EPA / PLACES / GTFS data for WalkCLIP cities")
    ap.add_argument(
        "--cities",
        nargs="+",
        choices=list(CITY_CONFIGS),
        default=list(CITY_CONFIGS),
        help="Cities to verify (default: all)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    results: list[VerifyResult] = []
    for city_key in args.cities:
        cfg = CITY_CONFIGS[city_key]
        logging.info("Verifying %s …", cfg.name)

        epa_ok, epa_rows, nwi_range, epa_note = verify_epa(cfg)
        logging.info("  EPA:   %s — %d block groups  NatWalkInd [%.1f, %.1f]  (%s)",
                     "OK" if epa_ok else "FAIL", epa_rows, nwi_range[0], nwi_range[1], epa_note)

        places_ok, places_tracts, places_note = verify_places(cfg)
        logging.info("  PLACES: %s — %d tracts  (%s)",
                     "OK" if places_ok else "FAIL", places_tracts, places_note)

        gtfs_ok, gtfs_note = verify_gtfs(cfg)
        logging.info("  GTFS:  %s — %s", "OK" if gtfs_ok else "FAIL", gtfs_note)

        results.append(VerifyResult(
            city=city_key,
            epa_ok=epa_ok, epa_rows=epa_rows, epa_natwalkind_range=nwi_range, epa_note=epa_note,
            places_ok=places_ok, places_tracts=places_tracts, places_note=places_note,
            gtfs_ok=gtfs_ok, gtfs_note=gtfs_note,
        ))

    print()
    print("=" * 60)
    all_ok = True
    for r in results:
        status = "PASS" if r.overall else "FAIL"
        print(f"  {r.city:10s}  {status}  "
              f"EPA={r.epa_rows:4d}bg  PLACES={r.places_tracts:4d}tr  "
              f"GTFS={'Y' if r.gtfs_ok else 'N'}")
        if not r.overall:
            all_ok = False
    print("=" * 60)
    if not all_ok:
        print("Fix FAIL items above before proceeding to point selection.")
        return 1
    print("All cities verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
