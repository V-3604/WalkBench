"""
WalkCLIP v2 — Stage C: build per-point label tables.

Reads raw sources:
  EPA SLD v3 GDB     → block-group walkability + component variables
  CDC PLACES 2025    → census-tract health / behavior prevalence
  GTFS (MSP + SEA)   → derived per-point transit access features

Spatially joins both cities' raw v2 point tables (4,874 points each) by lat/lon
→ Census geography.
No GEOIDs are pre-attached to the raw point files; this script derives them.

Outputs (project/data/processed/labels/):
  epa_walkability_blockgroup.csv
  places_health_tract.csv
  gtfs_transit_access.csv
  labels_joined_v2.csv

Run:
  python project/scripts/stage_c_build_labels.py
  python project/scripts/stage_c_build_labels.py --dry-run
  python project/scripts/stage_c_build_labels.py --limit 100   # quick test
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTS_MSP = REPO_ROOT / "project/data/processed/points/msp_points_v2.csv"
POINTS_SEA = REPO_ROOT / "project/data/processed/points/seattle_points_v2.csv"
POINTS_DC  = REPO_ROOT / "project/data/processed/points/dc_points_v2.csv"
EPA_GDB    = REPO_ROOT / "project/data/raw/labels/SmartLocationDatabase.gdb"
PLACES_CSV = (
    REPO_ROOT
    / "project/data/raw/labels/cdc_places"
    / "PLACES__Census_Tract_Data_(GIS_Friendly_Format),_2025_release_20260421.csv"
)
GTFS_MSP = REPO_ROOT / "project/data/raw/labels/gtfs/msp"
GTFS_SEA = REPO_ROOT / "project/data/raw/labels/gtfs/seattle"
GTFS_DC  = REPO_ROOT / "project/data/raw/labels/gtfs/dc"
OUT_DIR  = REPO_ROOT / "project/data/processed/labels"

GTFS_PGH = REPO_ROOT / "project/data/raw/labels/gtfs/pittsburgh"

CITY_POINTS_CSV: dict[str, Path] = {
    "msp":         POINTS_MSP,
    "seattle":     POINTS_SEA,
    "dc":          POINTS_DC,
    "pittsburgh":  REPO_ROOT / "project/data/processed/points/pittsburgh_points_v2.csv",
}
CITY_GTFS_DIR: dict[str, Path] = {
    "msp":         GTFS_MSP,
    "seattle":     GTFS_SEA,
    "dc":          GTFS_DC,
    "pittsburgh":  GTFS_PGH,
}
CITY_EPA_MSA_FRAGMENT: dict[str, str] = {
    "msp":         "Minneapolis",
    "seattle":     "Seattle",
    "dc":          "Washington-Arlington-Alexandria",
    "pittsburgh":  "Pittsburgh",
}

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EPA_LAYER = "EPA_SLD_Database_V3"

# Block-group variables to retain; GEOID keys handled separately.
# D-variable reference: https://www.epa.gov/smartgrowth/smart-location-mapping
EPA_VARS = [
    "NatWalkInd",   # composite National Walkability Index (1–20)
    "D2A_Ranked",   # employment mix score (0–6)
    "D2B_Ranked",   # employment + housing mix score (0–6)
    "D3B_Ranked",   # intersection density score (0–6)
    "D4A_Ranked",   # transit proximity score (0–4)
    "D1A",          # gross residential density (HU/acre)
    "D1B",          # gross population density (pop/acre)
    "D2B_E8MIX",    # employment entropy, 8-category
    "D3A",          # road network density (centerline mi/sq mi)
    "D3B",          # intersection density (auto+ped intersections/sq mi)
    "D3APO",        # pedestrian-oriented network density
    "D4A",          # distance to nearest transit stop (m)
    "D4C",          # transit routes within 0.5 mi
    "D4D",          # aggregate transit service frequency within 0.25 mi/hr
]

PLACES_VARS = [
    "LPA_CrudePrev",
    "LPA_Crude95CI",
    "OBESITY_CrudePrev",
    "OBESITY_Crude95CI",
    "TotalPopulation",
]

BUFFER_RADII_M  = [400, 800]
EARTH_RADIUS_M  = 6_371_008.8
SERVICE_START_M = 6 * 60    # 06:00
SERVICE_END_M   = 22 * 60   # 22:00
SERVICE_HOURS   = (SERVICE_END_M - SERVICE_START_M) / 60  # 16 h


# --------------------------------------------------------------------------- #
# 1. Canonical point table
# --------------------------------------------------------------------------- #

_CITY_PREFIX = {"msp": "MSP", "seattle": "SEA", "dc": "DC", "pittsburgh": "PGH"}


def load_points(include_cities: list[str], limit: int | None = None) -> gpd.GeoDataFrame:
    """Load and unify point tables for the specified cities as a GeoDataFrame (EPSG:4326).

    point_id is prefixed with the city code (e.g. MSP_0, SEA_0, DC_0) to prevent
    cross-city collisions in the label join.
    """
    dfs: list[pd.DataFrame] = []
    for city in include_cities:
        csv_path = CITY_POINTS_CSV[city]
        final_lock_path = csv_path.with_name(csv_path.stem + "_final_lock.csv")
        if final_lock_path.exists():
            csv_path = final_lock_path
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{city} points CSV not found: {csv_path}\n"
                f"Run select_{city}_points.py first."
            )
        df = pd.read_csv(csv_path)
        prefix = _CITY_PREFIX[city]
        df["point_id"] = f"{prefix}_" + df["point_id"].astype(str)
        df["city"] = city
        df = df[["point_id", "city", "metro", "state", "lat", "lon"]]
        if limit is not None:
            df = df.head(limit)
        dfs.append(df)
        print(f"  {city}: {len(df):,} points", flush=True)

    combined = pd.concat(dfs, ignore_index=True)
    if combined["point_id"].duplicated().any():
        raise ValueError("Duplicate point_id after city merge.")

    gdf = gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined["lon"], combined["lat"]),
        crs="EPSG:4326",
    )
    if gdf.geometry.isna().sum():
        raise ValueError("Null geometries detected — check lat/lon columns.")

    print(f"  Total points: {len(gdf):,}", flush=True)
    return gdf


# --------------------------------------------------------------------------- #
# 2. EPA SLD v3 join
# --------------------------------------------------------------------------- #

def join_epa(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatial join: points → EPA block groups.

    Returns a DataFrame keyed by point_id with blockgroup_geoid, tract_geoid,
    and all EPA_VARS that exist in the layer.
    """
    # DC metro: DC (11) + MD suburbs (24) + VA suburbs (51). MN=27, WA=53, PA=42.
    EPA_STATE_FIPS = ("'11'", "'24'", "'27'", "'42'", "'51'", "'53'")
    fips_filter = f"STATEFP IN ({', '.join(EPA_STATE_FIPS)})"
    print(f"\n[EPA] Reading layer '{EPA_LAYER}' (filtering MN + WA + DC + MD + PA + VA) ...")

    try:
        epa = gpd.read_file(str(EPA_GDB), layer=EPA_LAYER, where=fips_filter)
    except Exception:
        print("  [fallback] WHERE filter failed — reading all rows, filtering in memory ...")
        epa = gpd.read_file(str(EPA_GDB), layer=EPA_LAYER)
        epa = epa[epa["STATEFP"].astype(str).isin(["11", "24", "27", "42", "51", "53"])].copy()

    print(f"  EPA rows (MN+WA+DC+MD+PA+VA): {len(epa):,}  |  CRS: {epa.crs}")

    missing_vars = [v for v in EPA_VARS if v not in epa.columns]
    if missing_vars:
        print(f"  WARNING: EPA columns not in layer, will be skipped: {missing_vars}")
    present_vars = [v for v in EPA_VARS if v in epa.columns]

    # Prefer GEOID20 (2020 census tracts) over GEOID10 (2010) so that the derived
    # tract_geoid matches CDC PLACES 2025, which uses 2020 census tract definitions.
    # In fast-growing metros (MSP, Seattle, DC), many 2010 tracts were split by 2020;
    # using GEOID10 causes ~25% PLACES join failure due to this vintage mismatch.
    geoid_col = "GEOID20" if "GEOID20" in epa.columns else "GEOID10"
    if geoid_col not in epa.columns:
        raise RuntimeError("EPA layer has neither GEOID10 nor GEOID20 — cannot assign Census IDs.")

    epa = epa[[geoid_col] + present_vars + ["geometry"]].copy()

    # Reproject points to EPA's Albers CRS for spatial join
    pts = points[["point_id", "geometry"]].to_crs(epa.crs)

    # predicate="intersects" is more robust than "within" for points on polygon edges
    joined = gpd.sjoin(pts, epa, how="left", predicate="intersects")
    # Dedup if a point landed on a shared boundary (rare but possible)
    joined = joined[~joined.index.duplicated(keep="first")].reset_index(drop=True)

    # Explicitly preserve NaN for unmatched rows rather than converting to "nan" string.
    # EPA GEOID20 can be either an 11-digit tract ID or a 12-digit block-group ID
    # depending on the source row. Keep 11-digit tract IDs intact; do not zfill to
    # 12 before deriving tract_geoid, or Pennsylvania tracts become invalid.
    def _clean_geoid(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text.zfill(11) if len(text) < 11 else text

    geoid_raw = joined[geoid_col]
    matched = geoid_raw.notna()
    joined["blockgroup_geoid"] = geoid_raw.map(_clean_geoid)
    joined["tract_geoid"]      = joined["blockgroup_geoid"].str[:11]

    n_assigned = matched.sum()
    pct = 100 * n_assigned / len(points)
    print(f"  Assigned to block group: {n_assigned:,} / {len(points):,} ({pct:.1f}%)")
    if pct < 98:
        print(f"  WARNING: <98% assignment — inspect CRS and EPA coverage for sampled bbox.")
        print(f"  Note: residual misses are typically points in water/park areas with no block group.")

    out_cols = ["point_id", "blockgroup_geoid", "tract_geoid"] + present_vars
    return joined[out_cols]


# --------------------------------------------------------------------------- #
# 3. CDC PLACES 2025 join
# --------------------------------------------------------------------------- #

def join_places(epa_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Join PLACES 2025 health measures to points via tract GEOID.

    Nulls are preserved — suppressed/unreliable PLACES estimates appear as NaN.
    Masking per-target is left to Stage E (training).
    """
    print(f"\n[PLACES] Loading {PLACES_CSV.name} ...")
    places = pd.read_csv(PLACES_CSV, dtype={"TractFIPS": str}, low_memory=False)
    places.columns = places.columns.str.strip()

    places["TractFIPS"] = places["TractFIPS"].str.zfill(11)

    # Include MD + VA for DC metro points that fall in suburban Maryland/Virginia, and PA for Pittsburgh.
    PLACES_STATES = ["MN", "WA", "DC", "MD", "PA", "VA"]
    mn_wa_dc = places[places["StateAbbr"].isin(PLACES_STATES)].copy()
    counts = {s: len(mn_wa_dc[mn_wa_dc["StateAbbr"] == s]) for s in PLACES_STATES}
    print(
        f"  PLACES rows ({'+'.join(PLACES_STATES)}): {len(mn_wa_dc):,}  "
        + "  ".join(f"{s}={counts[s]:,}" for s in PLACES_STATES)
    )

    missing_vars = [v for v in PLACES_VARS if v not in mn_wa_dc.columns]
    if missing_vars:
        print(f"  WARNING: PLACES columns not found, will be skipped: {missing_vars}")
    present_vars = [v for v in PLACES_VARS if v in mn_wa_dc.columns]

    places_slim = mn_wa_dc[["TractFIPS"] + present_vars].drop_duplicates("TractFIPS")

    joined = epa_labels[["point_id", "tract_geoid"]].merge(
        places_slim, left_on="tract_geoid", right_on="TractFIPS", how="left"
    )

    for col in ["LPA_CrudePrev", "OBESITY_CrudePrev"]:
        if col in joined.columns:
            n_null = joined[col].isna().sum()
            print(f"  {col}  nulls: {n_null:,} / {len(joined):,}  ({100*n_null/len(joined):.1f}%)")

    out_cols = ["point_id", "tract_geoid"] + [c for c in present_vars]
    return joined[out_cols]


# --------------------------------------------------------------------------- #
# 4. GTFS transit access features
# --------------------------------------------------------------------------- #

def _parse_gtfs_time(t: str) -> int:
    """Parse 'HH:MM:SS' (hours may exceed 23) → minutes from midnight. Returns -1 on error."""
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return -1


def _pick_representative_tuesday(calendar: pd.DataFrame, gtfs_dir: Path) -> int:
    """
    Return a representative Tuesday within the feed's main service window as YYYYMMDD int.

    Prefers feed_info.txt anchor (max feed_start_date across agencies + 7-day buffer)
    so that consolidated multi-agency feeds with historical carryover entries (e.g.
    winter2025_* rows extending from 2025 into a spring-2026 feed) don't pull the
    representative date into a period where most services aren't running.

    Falls back to the calendar-based approach, restricted to the most recent year
    present in the feed, if feed_info.txt is absent.
    """
    feed_info_path = gtfs_dir / "feed_info.txt"
    if feed_info_path.exists():
        fi = pd.read_csv(feed_info_path, low_memory=False)
        fi.columns = fi.columns.str.strip()
        if "feed_start_date" in fi.columns:
            max_start = int(pd.to_numeric(fi["feed_start_date"], errors="coerce").max())
            # +7 days so all agencies in a consolidated feed are live before counting
            dt = datetime.strptime(str(max_start), "%Y%m%d") + timedelta(days=7)
            days_ahead = (1 - dt.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            dt += timedelta(days=days_ahead)
            return int(dt.strftime("%Y%m%d"))

    # Fallback: use calendar rows from the most recent year only (avoids 2025 carryover rows)
    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(int)
    dow_cols = [c for c in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                if c in cal.columns]
    active = cal[cal[dow_cols].sum(axis=1) > 0] if dow_cols else cal
    max_year = int(str(active["start_date"].max())[:4])
    main_window = active[active["start_date"].astype(str).str.startswith(str(max_year))]
    if main_window.empty:
        main_window = active
    anchor = int(main_window["start_date"].max())
    dt = datetime.strptime(str(anchor), "%Y%m%d")
    days_ahead = (1 - dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    dt += timedelta(days=days_ahead)
    return int(dt.strftime("%Y%m%d"))


def _active_service_ids(
    calendar: pd.DataFrame,
    calendar_dates: pd.DataFrame,
    target_date: int,
) -> set[str]:
    """Return service_ids active on target_date per GTFS calendar + exception rules."""
    dt = datetime.strptime(str(target_date), "%Y%m%d")
    dow = dt.strftime("%A").lower()

    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(int)
    cal["end_date"]   = cal["end_date"].astype(int)

    active: set[str] = set(
        cal.loc[
            (cal[dow] == 1)
            & (cal["start_date"] <= target_date)
            & (cal["end_date"]   >= target_date),
            "service_id",
        ]
    )

    if not calendar_dates.empty:
        cd = calendar_dates.copy()
        cd["date"] = cd["date"].astype(int)
        removes = set(cd.loc[(cd["date"] == target_date) & (cd["exception_type"] == 2), "service_id"])
        adds    = set(cd.loc[(cd["date"] == target_date) & (cd["exception_type"] == 1), "service_id"])
        active  = (active - removes) | adds

    return active


def _load_gtfs_tables(gtfs_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("stops", "stop_times", "trips", "routes", "calendar", "calendar_dates"):
        p = gtfs_dir / f"{name}.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Required GTFS file missing: {p}\n"
                f"Stage C cannot continue without {name}.txt."
            )
        tables[name] = pd.read_csv(p, low_memory=False)
    return tables


def compute_gtfs_features(
    points: gpd.GeoDataFrame,
    metro: str,
    gtfs_dir: Path,
) -> pd.DataFrame:
    """
    Derive per-point transit access features from a GTFS feed.

    Features (per buffer radius in BUFFER_RADII_M):
      stops_{r}m          — unique stop count within r metres
      trips_per_hr_{r}m   — sum of per-stop trips/hr within r metres
                            (trips served during 06:00-22:00 on representative Tuesday / 16 h)

    Uses haversine BallTree for efficient radius queries; same method for both cities.
    """
    print(f"\n[GTFS] {metro} — {gtfs_dir.name}/")
    tables = _load_gtfs_tables(gtfs_dir)

    # Build clean stops table (boarding stops only)
    raw_stops = tables["stops"].copy()
    if "location_type" in raw_stops.columns:
        raw_stops = raw_stops[raw_stops["location_type"].fillna(0).astype(int).isin([0, 1])]
    stops = raw_stops[["stop_id", "stop_lat", "stop_lon"]].dropna(
        subset=["stop_lat", "stop_lon"]
    ).copy()
    print(f"  Stops (valid lat/lon, boarding): {len(stops):,}")

    # Representative service date
    rep_date = _pick_representative_tuesday(tables["calendar"], gtfs_dir)
    print(f"  Representative Tuesday: {rep_date}")

    active_sids = _active_service_ids(tables["calendar"], tables["calendar_dates"], rep_date)
    print(f"  Active service_ids: {len(active_sids):,}")
    if not active_sids:
        raise RuntimeError(
            f"No active service_ids found for date {rep_date} in {gtfs_dir}. "
            "Check that the feed's validity window includes this date."
        )

    active_trips = set(
        tables["trips"].loc[tables["trips"]["service_id"].isin(active_sids), "trip_id"]
    )
    print(f"  Active trip_ids: {len(active_trips):,}")

    # Filter stop_times to active trips within service window
    st = tables["stop_times"][["trip_id", "stop_id", "departure_time"]].copy()
    st = st[st["trip_id"].isin(active_trips)].dropna(subset=["departure_time"])
    st["dep_min"] = st["departure_time"].apply(_parse_gtfs_time)
    st = st[(st["dep_min"] >= SERVICE_START_M) & (st["dep_min"] < SERVICE_END_M)]

    # Trips per stop in service window → trips/hr
    trips_per_stop = (
        st.groupby("stop_id")["trip_id"].nunique()
          .rename("trip_count")
          .reset_index()
    )
    stops = stops.merge(trips_per_stop, on="stop_id", how="left")
    stops["trip_count"]  = stops["trip_count"].fillna(0)
    stops["trips_per_hr"] = stops["trip_count"] / SERVICE_HOURS
    print(
        f"  Stops with >=1 trip: {(stops['trip_count'] > 0).sum():,}  |  "
        f"trips_per_hr  median={stops['trips_per_hr'].median():.2f}  "
        f"max={stops['trips_per_hr'].max():.2f}"
    )

    # City subset of points
    city_pts = points[points["metro"] == metro][["point_id", "lat", "lon"]].copy()
    if len(city_pts) == 0:
        raise ValueError(f"No points found with metro=='{metro}'. Check metro column values.")

    # BallTree with haversine metric (inputs in radians)
    stops_rad = np.radians(stops[["stop_lat", "stop_lon"]].values)
    pts_rad   = np.radians(city_pts[["lat", "lon"]].values)
    tree = BallTree(stops_rad, metric="haversine")

    out = city_pts[["point_id"]].copy()
    out["metro"] = metro

    for r_m in BUFFER_RADII_M:
        r_rad    = r_m / EARTH_RADIUS_M
        idx_list = tree.query_radius(pts_rad, r=r_rad)
        stop_counts = np.array([len(idx) for idx in idx_list])
        freq_sums   = np.array([
            stops.iloc[idx]["trips_per_hr"].sum() for idx in idx_list
        ])
        out[f"stops_{r_m}m"]        = stop_counts
        out[f"trips_per_hr_{r_m}m"] = np.round(freq_sums, 3)

    # Sanity: no fully-empty columns
    feature_cols = [c for c in out.columns if c not in ("point_id", "metro")]
    for col in feature_cols:
        if out[col].isna().all():
            raise RuntimeError(f"GTFS feature '{col}' is entirely null for {metro}.")
        print(
            f"  {col}: "
            f"min={out[col].min():.1f}  "
            f"med={out[col].median():.1f}  "
            f"max={out[col].max():.1f}"
        )

    return out


# --------------------------------------------------------------------------- #
# QA report
# --------------------------------------------------------------------------- #

def print_qa_report(
    points:  gpd.GeoDataFrame,
    epa:     pd.DataFrame,
    places:  pd.DataFrame,
    gtfs:    pd.DataFrame,
    joined:  pd.DataFrame,
) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("QA REPORT — WalkCLIP v2 Stage C")
    print(sep)

    for metro in sorted(points["metro"].dropna().astype(str).unique()):
        n = (points["metro"] == metro).sum()
        print(f"  {metro}: {n:,} points")

    print(f"\nEPA join ({len(epa):,} rows)")
    print(f"  blockgroup_geoid nulls : {epa['blockgroup_geoid'].isna().sum()}")
    print(f"  tract_geoid nulls      : {epa['tract_geoid'].isna().sum()}")
    if "NatWalkInd" in epa.columns:
        v = epa["NatWalkInd"].dropna()
        print(f"  NatWalkInd  n={len(v):,}  min={v.min():.1f}  mean={v.mean():.2f}  max={v.max():.1f}")

    print(f"\nPLACES join ({len(places):,} rows)")
    for col in ["LPA_CrudePrev", "OBESITY_CrudePrev"]:
        if col in places.columns:
            null_n = places[col].isna().sum()
            v = places[col].dropna()
            print(
                f"  {col:<26} nulls={null_n:,}  "
                f"mean={v.mean():.1f}  min={v.min():.1f}  max={v.max():.1f}"
            )

    print(f"\nGTFS access features ({len(gtfs):,} rows)")
    gtfs_metros = sorted(gtfs["metro"].dropna().astype(str).unique()) if len(gtfs) else []
    for metro in gtfs_metros:
        sub = gtfs[gtfs["metro"] == metro]
        print(f"  [{metro}]")
        for col in ["stops_400m", "stops_800m", "trips_per_hr_400m", "trips_per_hr_800m"]:
            if col in sub.columns:
                print(
                    f"    {col:<26} "
                    f"med={sub[col].median():.1f}  "
                    f"max={sub[col].max():.1f}"
                )

    print(f"\nJoined master table: {len(joined):,} rows × {len(joined.columns)} columns")
    total_nulls = joined.isna().sum()
    cols_with_nulls = total_nulls[total_nulls > 0]
    if len(cols_with_nulls):
        print("  Columns with nulls:")
        for col, n in cols_with_nulls.items():
            print(f"    {col:<30} {n:,} ({100*n/len(joined):.1f}%)")
    else:
        print("  No null values in joined table.")

    print(sep)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="WalkCLIP v2 Stage C: build EPA + PLACES + GTFS label tables"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip writing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit each city to N points for fast testing.",
    )
    parser.add_argument(
        "--include-cities",
        nargs="+",
        choices=["msp", "seattle", "dc", "pittsburgh"],
        default=["msp", "seattle", "dc"],
        help="Cities to process.",
    )
    args = parser.parse_args()

    # --- Input validation ---
    required: dict[str, Path] = {"EPA GDB": EPA_GDB, "PLACES CSV": PLACES_CSV}
    for city in args.include_cities:
        required[f"{city} points"] = CITY_POINTS_CSV[city]
        required[f"GTFS {city}"] = CITY_GTFS_DIR[city]
    errors = [f"  {label}: {path}" for label, path in required.items() if not path.exists()]
    if errors:
        print("ERROR: Missing required inputs:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    print("=" * 64)
    print("WalkCLIP v2 — Stage C label pipeline")
    if args.limit:
        print(f"  [--limit {args.limit}] Running on first {args.limit} points per city")
    if args.dry_run:
        print("  [--dry-run] Output files will NOT be written")
    print("=" * 64)

    # 1. Points
    points = load_points(include_cities=args.include_cities, limit=args.limit)

    # 2. EPA
    epa_labels = join_epa(points)

    # 3. PLACES
    places_labels = join_places(epa_labels)

    # 4. GTFS — per city, using city-specific GTFS directories
    gtfs_pieces: list[pd.DataFrame] = []
    city_metro_labels = {
        "msp": "Minneapolis",
        "seattle": "Seattle",
        "dc": "Washington DC",
        "pittsburgh": "Pittsburgh",
    }
    for city in args.include_cities:
        gtfs_dir = CITY_GTFS_DIR[city]
        if gtfs_dir.exists():
            metro_label = city_metro_labels[city]
            gtfs_pieces.append(compute_gtfs_features(points, metro_label, gtfs_dir))
        else:
            print(f"  WARNING: GTFS dir not found for {city}: {gtfs_dir} — skipping GTFS features", flush=True)
    gtfs_labels = pd.concat(gtfs_pieces, ignore_index=True) if gtfs_pieces else pd.DataFrame()

    # 5. Joined master
    master = (
        epa_labels
        .merge(places_labels.drop(columns=["tract_geoid"]), on="point_id", how="left")
        .merge(gtfs_labels.drop(columns=["metro"]),         on="point_id", how="left")
        .merge(
            points[["point_id", "metro", "state", "lat", "lon"]],
            on="point_id",
            how="left",
        )
    )
    # Re-order: identifiers first
    id_cols    = ["point_id", "metro", "state", "lat", "lon", "blockgroup_geoid", "tract_geoid"]
    other_cols = [c for c in master.columns if c not in id_cols]
    master     = master[id_cols + other_cols]

    # 6. QA
    print_qa_report(points, epa_labels, places_labels, gtfs_labels, master)

    if args.dry_run:
        print("\n[--dry-run] Skipping file writes. Done.")
        return 0

    # 7. Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "epa_walkability_blockgroup.csv": epa_labels,
        "places_health_tract.csv":        places_labels,
        "gtfs_transit_access.csv":        gtfs_labels,
        "labels_joined_v2.csv":           master,
    }
    def merge_city_subset(path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
        """Replace only the requested cities' prefixed point IDs in an existing output."""
        if not path.exists() or "point_id" not in new_df.columns:
            return new_df
        prefixes = tuple(f"{_CITY_PREFIX[city]}_" for city in args.include_cities)
        existing = pd.read_csv(path, low_memory=False)
        if "point_id" not in existing.columns:
            return new_df
        existing = existing[~existing["point_id"].astype(str).str.startswith(prefixes)]
        return pd.concat([existing, new_df], ignore_index=True)

    print()
    for fname, df in outputs.items():
        path = OUT_DIR / fname
        merged = merge_city_subset(path, df)
        merged.to_csv(path, index=False)
        print(f"  Wrote {path.relative_to(REPO_ROOT)}  ({len(df):,} rows, {len(df.columns)} cols)")

    # 8. Append new cities into features_labels_agreement.csv
    _append_to_features_labels_agreement(master, args.include_cities)

    return 0


# --------------------------------------------------------------------------- #
# Append helper: merges stage-C output into the master FLA CSV
# --------------------------------------------------------------------------- #

_MASTER_CSV = OUT_DIR / "features_labels_agreement.csv"
_OVERTURE_CSV = OUT_DIR / "overture_targets.csv"
_PREFIX_TO_CITY: dict[str, str] = {}   # populated lazily from _CITY_PREFIX


def _split_compound_id(cid: str) -> tuple[int, str] | None:
    """'PGH_42' → (42, 'pittsburgh').  Returns None if cid doesn't match."""
    global _PREFIX_TO_CITY
    if not _PREFIX_TO_CITY:
        _PREFIX_TO_CITY = {v: k for k, v in _CITY_PREFIX.items()}
    parts = str(cid).split("_", 1)
    if len(parts) != 2 or not parts[1].isdigit() or parts[0] not in _PREFIX_TO_CITY:
        return None
    return int(parts[1]), _PREFIX_TO_CITY[parts[0]]


def _append_to_features_labels_agreement(
    master: pd.DataFrame,
    include_cities: list[str],
) -> None:
    """Append rows for *include_cities* from the stage-C master table into
    features_labels_agreement.csv, filling SAM2/caption/agreement columns with NaN.

    Existing rows for those cities are replaced so the function is idempotent.
    """
    if not _MASTER_CSV.exists():
        print(f"  [FLA] {_MASTER_CSV.name} not found — skipping FLA append.", flush=True)
        return

    # 1. Parse compound IDs, keep only requested cities
    parsed = master["point_id"].apply(_split_compound_id)
    new_rows = master.copy()
    new_rows["point_id"] = [p[0] if p else None for p in parsed]
    new_rows["city"]     = [p[1] if p else None for p in parsed]
    new_rows = new_rows[
        new_rows["city"].isin(include_cities) & new_rows["point_id"].notna()
    ].copy()
    new_rows["point_id"] = new_rows["point_id"].astype(int)

    if new_rows.empty:
        print("  [FLA] No rows matched requested cities — skipping FLA append.", flush=True)
        return

    # 2. Join Overture targets (sidewalk/crosswalk/building/intersection cols)
    if _OVERTURE_CSV.exists():
        ot = pd.read_csv(_OVERTURE_CSV, low_memory=False)
        ot["point_id"] = ot["point_id"].astype(int)
        overture_cols = [c for c in ot.columns if c not in ("point_id", "city", "metro", "state")]
        new_rows = new_rows.merge(
            ot[["point_id", "city"] + overture_cols],
            on=["point_id", "city"],
            how="left",
        )
        # overture_targets.csv uses bare column names (e.g. "sidewalk_present") but the FLA
        # schema uses the "overture_" prefix (e.g. "overture_sidewalk_present").  Rename any
        # bare column whose prefixed variant exists in the FLA schema so step 4 aligns them.
        existing_check = pd.read_csv(_MASTER_CSV, nrows=0, low_memory=False)
        rename_map = {
            c: f"overture_{c}"
            for c in overture_cols
            if c in new_rows.columns
            and f"overture_{c}" in existing_check.columns
            and c not in existing_check.columns
        }
        if rename_map:
            new_rows = new_rows.rename(columns=rename_map)

    # 3. Read existing FLA to get column schema
    existing = pd.read_csv(_MASTER_CSV, low_memory=False)
    fla_cols = list(existing.columns)

    # 4. Align new rows to FLA schema (fill missing cols with NaN)
    new_rows["n_headings"] = 4
    for col in fla_cols:
        if col not in new_rows.columns:
            new_rows[col] = float("nan")
    new_rows = new_rows[fla_cols].copy()

    # 5. Replace existing rows for these cities, then concatenate
    existing_trimmed = existing[~existing["city"].isin(include_cities)]
    combined = pd.concat([existing_trimmed, new_rows], ignore_index=True)
    combined.to_csv(_MASTER_CSV, index=False)
    print(
        f"  [FLA] Appended {len(new_rows):,} rows to features_labels_agreement.csv "
        f"(total now: {len(combined):,} rows)",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
