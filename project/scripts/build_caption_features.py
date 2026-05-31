"""
Build model-ready caption features from final-lock structured OpenAI audit outputs.

Inputs:
    project/data/processed/text/streetview_openai_msp_final_lock.jsonl
    project/data/processed/text/streetview_openai_seattle_final_lock.jsonl
    project/data/processed/locks/v2_final_lock_ids.txt

Outputs:
    project/data/processed/text/caption_features_final_lock.csv
    project/data/processed/text/caption_features_summary.json

Feature design:
    - Ordinal features for ordered walkability cues
    - One-hot features for single-valued categorical fields
    - Multi-hot features for list-valued fields (land_use, barriers)
    - Unknown-count diagnostic feature

Run:
    python project/scripts/build_caption_features.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = REPO_ROOT / "project" / "data" / "processed" / "text"
LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"

# Per-city lock files; keys must match CITY_JSONLS keys
CITY_LOCK_FILES: dict[str, Path] = {
    "msp":     REPO_ROOT / "project" / "data" / "processed" / "locks" / "msp_v2_final_lock_ids.txt",
    "seattle": REPO_ROOT / "project" / "data" / "processed" / "locks" / "seattle_v2_final_lock_ids.txt",
    "dc":      REPO_ROOT / "project" / "data" / "processed" / "locks" / "dc_v2_final_lock_ids.txt",
}

CITY_JSONLS: dict[str, Path] = {
    "msp":     TEXT_DIR / "streetview_openai_msp_final_lock.jsonl",
    "seattle": TEXT_DIR / "streetview_openai_seattle_final_lock.jsonl",
    "dc":      TEXT_DIR / "streetview_openai_dc_final_lock.jsonl",
}

MSP_JSONL = CITY_JSONLS["msp"]         # backward-compat aliases
SEATTLE_JSONL = CITY_JSONLS["seattle"]
OUT_CSV = TEXT_DIR / "caption_features_final_lock.csv"
OUT_JSON = TEXT_DIR / "caption_features_summary.json"

ORDERED_MAPS: dict[str, dict[str, int]] = {
    "sidewalk": {"none": 0, "one_side": 1, "both_sides": 2},
    "crosswalk": {"absent": 0, "present": 1},
    "traffic_volume": {"low": 0, "med": 1, "high": 2},
    "tree_canopy": {"none": 0, "sparse": 1, "moderate": 2, "dense": 3},
    "transit_stop": {"absent": 0, "present": 1},
    "lighting": {"poor": 0, "fair": 1, "good": 2},
    "maintenance": {"poor": 0, "fair": 1, "good": 2},
}

SINGLE_FIELDS = [
    "sidewalk",
    "crosswalk",
    "bike_lane",
    "traffic_volume",
    "tree_canopy",
    "transit_stop",
    "lighting",
    "maintenance",
]
MULTI_FIELDS = ["land_use", "barriers"]


def load_lock_ids() -> set[int]:
    return {int(x) for x in LOCK_FILE.read_text().strip().splitlines()}


def _normalise_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    else:
        items = [value]

    cleaned = [str(x).strip().lower() for x in items if str(x).strip()]
    return cleaned or ["unknown"]


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
        if key in MULTI_FIELDS:
            parsed[key] = [x.strip() for x in value.split("|") if x.strip()] or ["unknown"]
        else:
            parsed[key] = value or "unknown"
    return parsed


def _city_lock_ids(city: str, fallback: set[int]) -> set[int]:
    lock_path = CITY_LOCK_FILES.get(city)
    if lock_path and lock_path.exists():
        return {int(x) for x in lock_path.read_text().strip().splitlines()}
    return fallback


def load_caption_rows(lock_ids: set[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for city, path in CITY_JSONLS.items():
        if not path.exists():
            continue
        city_lock = _city_lock_ids(city, lock_ids)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                point_id = int(rec["id"])
                if point_id not in city_lock:
                    continue
                payload = rec.get("json") or _parse_answer_string(rec.get("answer", ""))
                row = {"point_id": point_id, "city": city}
                for field in SINGLE_FIELDS:
                    row[field] = str(payload.get(field, "unknown")).strip().lower()
                for field in MULTI_FIELDS:
                    row[field] = _normalise_list(payload.get(field, ["unknown"]))
                rows.append(row)

    captions = pd.DataFrame(rows).drop_duplicates(["point_id", "city"]).sort_values(["city", "point_id"])
    return captions.reset_index(drop=True)


def build_features(captions: pd.DataFrame) -> pd.DataFrame:
    feat = captions[["point_id", "city"]].copy()

    unknown_count = pd.Series(0, index=captions.index, dtype="int32")

    for field in SINGLE_FIELDS:
        values = captions[field].fillna("unknown").astype(str).str.lower()
        categories = sorted(values.unique())
        for category in categories:
            col = f"cap_{field}__{category}"
            feat[col] = (values == category).astype("float32")
        if field in ORDERED_MAPS:
            ordinal = values.map(ORDERED_MAPS[field]).fillna(-1).astype("float32")
            feat[f"cap_{field}_ord"] = ordinal
            feat[f"cap_{field}_known"] = (ordinal >= 0).astype("float32")
        unknown_count += (values == "unknown").astype("int32")

    for field in MULTI_FIELDS:
        all_categories = sorted(
            {
                item
                for items in captions[field]
                for item in items
            }
        )
        for category in all_categories:
            col = f"cap_{field}__{category}"
            feat[col] = captions[field].apply(lambda xs, cat=category: float(cat in xs)).astype("float32")
        unknown_count += captions[field].apply(lambda xs: int("unknown" in xs)).astype("int32")

    feat["cap_unknown_count"] = unknown_count.astype("float32")
    feat["cap_known_fraction"] = ((len(SINGLE_FIELDS) + len(MULTI_FIELDS)) - unknown_count).astype("float32")
    feat["cap_known_fraction"] /= float(len(SINGLE_FIELDS) + len(MULTI_FIELDS))

    return feat


def write_summary(captions: pd.DataFrame, feat: pd.DataFrame) -> None:
    summary = {
        "n_rows": int(len(feat)),
        "n_per_city": captions.groupby("city")["point_id"].count().to_dict(),
        "n_features": int(len(feat.columns) - 2),
        "unknown_rate": {},
    }
    for field in SINGLE_FIELDS:
        vals = captions[field].fillna("unknown").astype(str).str.lower()
        summary["unknown_rate"][field] = round(float((vals == "unknown").mean()), 4)
    for field in MULTI_FIELDS:
        summary["unknown_rate"][field] = round(float(captions[field].apply(lambda xs: "unknown" in xs).mean()), 4)
    OUT_JSON.write_text(json.dumps(summary, indent=2))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if not LOCK_FILE.exists():
        logging.error("Lock file not found: %s", LOCK_FILE)
        return 1
    present = [p for p in CITY_JSONLS.values() if p.exists()]
    if not present:
        logging.error(
            "No caption JSONL files found. Expected at least one of: %s",
            ", ".join(str(p) for p in CITY_JSONLS.values()),
        )
        return 1

    lock_ids = load_lock_ids()
    captions = load_caption_rows(lock_ids)
    features = build_features(captions)

    features.to_csv(OUT_CSV, index=False)
    write_summary(captions, features)

    logging.info("Saved: %s  (%d rows × %d cols)", OUT_CSV, len(features), len(features.columns))
    logging.info("Saved: %s", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
