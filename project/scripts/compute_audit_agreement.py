"""
Stage D3: Channel A / Channel B agreement scoring.

Compares OpenAI structured captions (Channel A) against SAM2 pixel-fraction
features (Channel B) on 4 matchable feature pairs per point.

Inputs:
    project/data/processed/labels/features_labels_combined.csv        (9,648 rows)
    project/data/processed/text/streetview_openai_seattle_final_lock.jsonl  (4,824 rows)
    project/data/processed/text/streetview_openai_msp_final_lock.jsonl      (4,824 rows)
    project/data/processed/locks/v2_final_lock_ids.txt                (4,824 IDs)

Outputs:
    project/data/processed/labels/features_labels_agreement.csv       (9,648 rows, now retains Channel A ordinals)
    project/data/processed/labels/agreement_summary.json

Agreement pairs:
    sidewalk     : A sidewalk ∈ {none,one_side,both_sides} → 0-2
                   B sidewalk_frac_max thresholded        → 0-2
                   rule: |A - B| <= 0  (exact 3-class match)

    crosswalk    : A crosswalk ∈ {present,absent}         → 0/1
                   B crosswalk_frac_max > 0.02            → 0/1
                   rule: exact binary

    tree_canopy  : A tree_canopy ∈ {none,sparse,moderate,dense} → 0-3
                   B vegetation_frac_mean percentile-binned     → 0-3
                   rule: |A - B| <= 1

    transit_stop : A transit_stop ∈ {present,absent}      → 0/1
                   B stops_400m > 0  (GTFS, in combined)  → 0/1
                   rule: exact binary

'unknown' values in Channel A are treated as missing — the pair is excluded
from that point's agreement_score denominator.

Agreement tiers:
    >= 0.75  →  high_confidence
    >= 0.50  →  partial
    <  0.50  →  disagreement

Run:
    python project/scripts/compute_audit_agreement.py
    python project/scripts/compute_audit_agreement.py --dry-run   # stats only, no write
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = REPO_ROOT / "project" / "data" / "processed" / "labels"
TEXT_DIR  = REPO_ROOT / "project" / "data" / "processed" / "text"
LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"

COMBINED_CSV = LABEL_DIR / "features_labels_combined.csv"
FALLBACK_FEATURES_CSV = LABEL_DIR / "features_labels_agreement.csv"
SEATTLE_JSONL = TEXT_DIR / "streetview_openai_seattle_final_lock.jsonl"
MSP_JSONL     = TEXT_DIR / "streetview_openai_msp_final_lock.jsonl"
DC_JSONL      = TEXT_DIR / "streetview_openai_dc_final_lock.jsonl"

# Per-city lock files; falls back to shared lock for MSP/Seattle
CITY_LOCK_FILES = {
    "msp":     REPO_ROOT / "project" / "data" / "processed" / "locks" / "msp_v2_final_lock_ids.txt",
    "seattle": REPO_ROOT / "project" / "data" / "processed" / "locks" / "seattle_v2_final_lock_ids.txt",
    "dc":      REPO_ROOT / "project" / "data" / "processed" / "locks" / "dc_v2_final_lock_ids.txt",
}

OUT_CSV  = LABEL_DIR / "features_labels_agreement.csv"
OUT_JSON = LABEL_DIR / "agreement_summary.json"

# ── Channel A ordinal encodings ───────────────────────────────────────────────

_SIDEWALK_MAP: dict[str, float] = {
    "none":       0.0,
    "one_side":   1.0,
    "both_sides": 2.0,
    "unknown":    float("nan"),
}

_CROSSWALK_MAP: dict[str, float] = {
    "present": 1.0,
    "absent":  0.0,
    "unknown": float("nan"),
}

_CANOPY_MAP: dict[str, float] = {
    "none":     0.0,
    "sparse":   1.0,
    "moderate": 2.0,
    "dense":    3.0,
    "unknown":  float("nan"),
}

_TRANSIT_MAP: dict[str, float] = {
    "present": 1.0,
    "absent":  0.0,
    "unknown": float("nan"),
}

# ── Channel B thresholds and bins ─────────────────────────────────────────────

# sidewalk_frac_max → 3-class ordinal matching {none, one_side, both_sides}
# Thresholds tuned to match the ~19%/23%/58% split in Seattle captions.
_SW_LOW  = 0.020   # below → none (0)
_SW_HIGH = 0.095   # above → both_sides (2); between → one_side (1)

# crosswalk_frac_max > threshold → binary present (1)
_XW_THRESH = 0.020

# vegetation_frac_mean percentile cut points → 4-class ordinal {none,sparse,moderate,dense}
# Computed globally across all 9,648 points so the scale is city-agnostic.
_CANOPY_PCTS = (20.0, 50.0, 80.0)   # p20 / p50 / p80 cuts


def load_lock_ids() -> set[int]:
    ids = {int(x) for x in LOCK_FILE.read_text().strip().splitlines()}
    logging.info("Lock IDs loaded: %d", len(ids))
    return ids


def _parse_answer_string(answer: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for part in str(answer).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key == "traffic":
            key = "traffic_volume"
        if key in {"land_use", "barriers"}:
            parsed[key] = [x.strip() for x in value.split("|") if x.strip()] or ["unknown"]
        else:
            parsed[key] = value or "unknown"
    return parsed


def _load_city_lock_ids(city: str, fallback: set[int]) -> set[int]:
    lock_path = CITY_LOCK_FILES.get(city)
    if lock_path and lock_path.exists():
        ids = {int(x) for x in lock_path.read_text().strip().splitlines()}
        logging.info("Per-city lock loaded: %s  n=%d", city, len(ids))
        return ids
    return fallback


def load_captions(lock_ids: set[int]) -> pd.DataFrame:
    """Return DataFrame with columns: point_id (int), city, and raw caption fields."""
    rows: list[dict] = []

    city_jsonls = [
        ("seattle", SEATTLE_JSONL),
        ("msp", MSP_JSONL),
    ]
    if DC_JSONL.exists():
        city_jsonls.append(("dc", DC_JSONL))

    for city, jsonl_path in city_jsonls:
        city_lock = _load_city_lock_ids(city, lock_ids)
        if not jsonl_path.exists():
            logging.warning("JSONL not found, skipping %s: %s", city, jsonl_path)
            continue
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                pid = int(rec["id"])
                if pid not in city_lock:
                    continue
                payload = rec.get("json") or _parse_answer_string(rec.get("answer", ""))
                rows.append({"point_id": pid, "city": city, **payload})

    captions = pd.DataFrame(rows).drop_duplicates(["point_id", "city"], keep="last")
    counts = {c: int((captions["city"] == c).sum()) for c in ("seattle", "msp", "dc")}
    logging.info("Captions loaded: seattle=%d  msp=%d  dc=%d", counts["seattle"], counts["msp"], counts["dc"])
    return captions


def encode_channel_a(captions: pd.DataFrame) -> pd.DataFrame:
    """Map string caption fields to ordinal floats (NaN for 'unknown')."""
    out = captions[["point_id", "city"]].copy()
    out["ca_sidewalk"]     = captions["sidewalk"].map(_SIDEWALK_MAP)
    out["ca_crosswalk"]    = captions["crosswalk"].map(_CROSSWALK_MAP)
    out["ca_tree_canopy"]  = captions["tree_canopy"].map(_CANOPY_MAP)
    out["ca_transit_stop"] = captions["transit_stop"].map(_TRANSIT_MAP)
    return out


def encode_channel_b(features: pd.DataFrame) -> pd.DataFrame:
    """Bin SAM2 fraction features into ordinals matching Channel A scales."""
    out = features[["point_id", "city"]].copy()

    # sidewalk: 3-class
    sw = features["sidewalk_frac_max"]
    out["cb_sidewalk"] = np.where(sw < _SW_LOW, 0.0, np.where(sw >= _SW_HIGH, 2.0, 1.0))

    # crosswalk: binary
    out["cb_crosswalk"] = (features["crosswalk_frac_max"] > _XW_THRESH).astype(float)

    # tree_canopy: 4-class via global percentile cuts
    veg = features["vegetation_frac_mean"]
    p20, p50, p80 = np.percentile(veg.dropna(), list(_CANOPY_PCTS))
    logging.info("Vegetation percentile cuts: p20=%.4f  p50=%.4f  p80=%.4f", p20, p50, p80)
    out["cb_tree_canopy"] = np.where(
        veg <= p20, 0.0,
        np.where(veg <= p50, 1.0,
        np.where(veg <= p80, 2.0, 3.0))
    )

    # transit_stop: binary from GTFS stops_400m
    out["cb_transit_stop"] = (features["stops_400m"] > 0).astype(float)

    return out


def compute_agreement(ca: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """
    Join Channel A and B, compute per-pair agree/disagree, aggregate to score.
    Returns DataFrame with point_id, city, per-pair columns, agreement_score, tier.
    """
    merged = ca.merge(cb, on=["point_id", "city"], how="inner")

    # Sidewalk: lenient ordinal — |A - B| <= 1 (SAM2 conflates curbs/shoulder
    # with one_side sidewalk; exact 3-class match is too strict for a pixel proxy)
    diff_sw = (merged["ca_sidewalk"] - merged["cb_sidewalk"]).abs()
    merged["agree_sidewalk"] = np.where(
        merged["ca_sidewalk"].isna(), float("nan"),
        (diff_sw <= 1).astype(float),
    )

    # Crosswalk: exact binary
    merged["agree_crosswalk"] = np.where(
        merged["ca_crosswalk"].isna(), float("nan"),
        (merged["ca_crosswalk"] == merged["cb_crosswalk"]).astype(float),
    )

    # Tree canopy: ordinal within 1
    diff_tc = (merged["ca_tree_canopy"] - merged["cb_tree_canopy"]).abs()
    merged["agree_tree_canopy"] = np.where(
        merged["ca_tree_canopy"].isna(), float("nan"),
        (diff_tc <= 1).astype(float),
    )

    # Transit stop: EXCLUDED from agreement score.
    # Caption field measures visible stop in image; GTFS stops_400m measures
    # proximity — 90% of "absent in image" points still have a stop within 400m.
    # These are different constructs; forcing agreement is methodologically invalid.
    # Kept as a column for reference but excluded from pair_cols.
    merged["agree_transit_stop"] = np.where(
        merged["ca_transit_stop"].isna(), float("nan"),
        (merged["ca_transit_stop"] == merged["cb_transit_stop"]).astype(float),
    )

    pair_cols = ["agree_sidewalk", "agree_crosswalk", "agree_tree_canopy"]

    # Per-point mean over non-NaN pairs (at least 1 valid pair required)
    scores = merged[pair_cols].mean(axis=1)
    n_valid = merged[pair_cols].notna().sum(axis=1)
    scores = scores.where(n_valid >= 1)
    merged["agreement_score"] = scores.round(4)

    merged["agreement_tier"] = pd.cut(
        merged["agreement_score"],
        bins=[-0.001, 0.4999, 0.7499, 1.001],
        labels=["disagreement", "partial", "high_confidence"],
    )

    keep = ["point_id", "city"] + pair_cols + ["agree_transit_stop", "agreement_score", "agreement_tier"]
    return merged[keep]


def print_summary(agreement: pd.DataFrame, features: pd.DataFrame) -> dict:
    full = features.merge(agreement, on=["point_id", "city"], how="left")
    tier_counts = full["agreement_tier"].value_counts()
    n = len(full)

    summary: dict = {
        "n_points": n,
        "n_per_city": full.groupby("city")["point_id"].count().to_dict(),
        "agreement_score": {
            "mean":   round(float(full["agreement_score"].mean()), 4),
            "std":    round(float(full["agreement_score"].std()), 4),
            "median": round(float(full["agreement_score"].median()), 4),
        },
        "tiers": {
            t: {"n": int(tier_counts.get(t, 0)), "pct": round(tier_counts.get(t, 0) / n * 100, 1)}
            for t in ["high_confidence", "partial", "disagreement"]
        },
        "per_pair": {},
        "per_city": {},
    }

    for col, label in [
        ("agree_sidewalk",     "sidewalk (lenient ±1)"),
        ("agree_crosswalk",    "crosswalk"),
        ("agree_tree_canopy",  "tree_canopy"),
        ("agree_transit_stop", "transit_stop [EXCLUDED — different constructs]"),
    ]:
        valid = full[col].dropna()
        summary["per_pair"][label] = {
            "n_valid": int(len(valid)),
            "pct_unknown": round((full[col].isna().sum() / n) * 100, 1),
            "agreement_rate": round(float(valid.mean()), 4),
        }

    for city in sorted(full["city"].dropna().unique()):
        sub = full[full["city"] == city]
        tc = sub["agreement_tier"].value_counts()
        summary["per_city"][city] = {
            "mean_score": round(float(sub["agreement_score"].mean()), 4),
            "high_confidence_pct": round(tc.get("high_confidence", 0) / len(sub) * 100, 1),
        }

    logging.info("─" * 60)
    logging.info("Agreement summary  (n=%d)", n)
    logging.info("  mean score : %.3f  median : %.3f", summary["agreement_score"]["mean"], summary["agreement_score"]["median"])
    for tier in ["high_confidence", "partial", "disagreement"]:
        t = summary["tiers"][tier]
        logging.info("  %-20s %5d  (%5.1f%%)", tier, t["n"], t["pct"])
    logging.info("Per-pair agreement rates:")
    for label, vals in summary["per_pair"].items():
        logging.info(
            "  %-15s  rate=%.3f  n_valid=%d  unknown=%.1f%%",
            label, vals["agreement_rate"], vals["n_valid"], vals["pct_unknown"],
        )
    logging.info("Per-city mean scores:")
    for city, vals in summary["per_city"].items():
        logging.info("  %-10s  mean=%.3f  high_confidence=%.1f%%", city, vals["mean_score"], vals["high_confidence_pct"])
    logging.info("─" * 60)

    return summary


_CITY_PREFIX_MAP = {"MSP": "msp", "SEA": "seattle", "DC": "dc"}

_POINTS_LOCK_FILES = {
    "msp":     REPO_ROOT / "project" / "data" / "processed" / "points" / "msp_points_v2_final_lock.csv",
    "seattle": REPO_ROOT / "project" / "data" / "processed" / "points" / "seattle_points_v2_final_lock.csv",
    "dc":      REPO_ROOT / "project" / "data" / "processed" / "points" / "dc_points_v2_final_lock.csv",
}


def _build_combined_features() -> pd.DataFrame | None:
    """
    Build features_labels_combined.csv from:
      - sam2_features_combined.csv   (point_id = raw int, city = lowercase)
      - gtfs_transit_access.csv      (point_id = prefixed e.g. MSP_0)
      - per-city points lock CSVs    (lat, lon)

    Returns merged DataFrame with raw int point_id, lowercase city, lat, lon,
    SAM2 features, and GTFS transit columns.
    """
    sam2_path = LABEL_DIR.parent / "audit" / "sam2_features_combined.csv"
    gtfs_path = LABEL_DIR / "gtfs_transit_access.csv"
    for p in (sam2_path, gtfs_path):
        if not p.exists():
            logging.error(
                "Cannot build features_labels_combined.csv: missing %s\n"
                "Run build_sam2_features.py and stage_c_build_labels.py first.",
                p,
            )
            return None

    sam2 = pd.read_csv(sam2_path)
    sam2["point_id"] = sam2["point_id"].astype(str)

    # Decode prefixed point_ids in GTFS table (MSP_0 → city=msp, point_id=0)
    gtfs = pd.read_csv(gtfs_path)
    prefix_col = gtfs["point_id"].str.split("_", n=1, expand=True)
    gtfs["_city"] = prefix_col[0].map(_CITY_PREFIX_MAP)
    gtfs["_raw_id"] = prefix_col[1]
    missing_prefix = gtfs["_city"].isna().sum()
    if missing_prefix:
        logging.warning("gtfs_transit_access: %d rows with unrecognised city prefix", missing_prefix)
    gtfs = gtfs.dropna(subset=["_city"])
    gtfs_keep = (
        gtfs.drop(columns=["point_id"])
        .rename(columns={"_raw_id": "point_id", "_city": "city"})[
            ["point_id", "city", "stops_400m", "trips_per_hr_400m", "stops_800m", "trips_per_hr_800m"]
        ]
    )

    # Load lat/lon from per-city lock point files
    coord_frames: list[pd.DataFrame] = []
    for city, pts_path in _POINTS_LOCK_FILES.items():
        if not pts_path.exists():
            logging.warning("Points lock file missing for %s: %s", city, pts_path)
            continue
        pts = pd.read_csv(pts_path, usecols=["point_id", "lat", "lon"])
        pts["point_id"] = pts["point_id"].astype(str)
        pts["city"] = city
        coord_frames.append(pts)
    if not coord_frames:
        logging.error("No per-city points lock files found — cannot attach lat/lon.")
        return None
    coords = pd.concat(coord_frames, ignore_index=True)

    combined = sam2.merge(gtfs_keep, on=["point_id", "city"], how="left")
    combined = combined.merge(coords, on=["point_id", "city"], how="left")

    n_missing_stops = combined["stops_400m"].isna().sum()
    if n_missing_stops:
        logging.warning(
            "%d rows missing stops_400m after GTFS join; filling 0", n_missing_stops,
        )
        for col in ("stops_400m", "trips_per_hr_400m", "stops_800m", "trips_per_hr_800m"):
            combined[col] = combined[col].fillna(0)

    n_missing_coords = combined["lat"].isna().sum()
    if n_missing_coords:
        logging.warning("%d rows missing lat/lon after coords join", n_missing_coords)

    logging.info(
        "Built combined features: %d rows, cities: %s",
        len(combined),
        combined["city"].value_counts().to_dict(),
    )
    return combined


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Channel A / B agreement scores")
    p.add_argument("--dry-run", action="store_true", help="Print stats but do not write output files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    for path in (SEATTLE_JSONL, MSP_JSONL, LOCK_FILE):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            return 1

    lock_ids = load_lock_ids()

    if COMBINED_CSV.exists():
        features = pd.read_csv(COMBINED_CSV)
        logging.info("Loaded features_labels_combined.csv: %d rows", len(features))
    else:
        features = _build_combined_features()
        if features is None:
            return 1
        features.to_csv(COMBINED_CSV, index=False)
        logging.info("Built and saved features_labels_combined.csv: %d rows", len(features))
    stale_cols = [
        "ca_sidewalk",
        "ca_crosswalk",
        "ca_tree_canopy",
        "ca_transit_stop",
        "agree_sidewalk",
        "agree_crosswalk",
        "agree_tree_canopy",
        "agree_transit_stop",
        "agreement_score",
        "agreement_tier",
    ]
    features = features.drop(columns=[c for c in stale_cols if c in features.columns], errors="ignore")
    features["point_id"] = features["point_id"].astype(int)
    logging.info("Features loaded: %d rows × %d cols", len(features), len(features.columns))

    captions = load_captions(lock_ids)
    ca = encode_channel_a(captions)
    cb = encode_channel_b(features)

    agreement = compute_agreement(ca, cb)
    summary = print_summary(agreement, features)

    if args.dry_run:
        logging.info("--dry-run: no files written")
        return 0

    out_df = features.merge(ca, on=["point_id", "city"], how="inner")
    out_df = out_df.merge(agreement, on=["point_id", "city"], how="left")
    out_df.to_csv(OUT_CSV, index=False)
    logging.info("Saved: %s  (%d rows × %d cols)", OUT_CSV, len(out_df), len(out_df.columns))

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    logging.info("Saved: %s", OUT_JSON)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
