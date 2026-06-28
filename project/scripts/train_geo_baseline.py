"""
Coordinates-only geo baseline for WalkBench (advisor floor check, 2026-06-24).

Purpose
-------
Quantify how well the walkability targets can be predicted from raw geography
(lat/lon) alone, under the same city-holdout protocol as the imagery models.
If coordinates-only models match the frozen-imagery transfer numbers the imagery
is not earning its keep; if imagery clearly wins, that gap is the motivation for
the method. Spec: docs/internal/ADVISOR_FEEDBACK_LOG.md (2026-06-24,
"coordinates-only baseline"). This is a full rewrite of the old RidgeCV /
NatWalkInd / 2-city script.

Protocol (city holdout, matching the main results)
--------------------------------------------------
  * 6 cross-city directions among MSP / Seattle / DC (train one city, test
    another).
  * Pittsburgh held out: train msp+seattle+dc -> test pittsburgh (zero-shot).
  * Supplementary in-city diagonal (seeded 70/30 split within each city) to show
    that geography predicts *within* a city but does not transfer *across*
    cities -- the contrast that motivates a visual method.

Models : RandomForest, HistGradientBoosting, KNN. Classifiers for the two binary
         targets, regressors for the three continuous ones. Seeds 41/42/43.
Features: lat/lon only (headline) plus an optional census/GTFS "geo+context" set.
Hygiene: median-impute + StandardScaler fit on the training cities only (KNN
         needs scaling; trees are invariant to it). Every estimator is seeded.

Targets (5, apples-to-apples with the main table)
  overture_sidewalk_present              (binary, AUROC)
  overture_crosswalk_present             (binary, AUROC)
  overture_intersection_count_200m       (regression, Spearman rho)
  overture_building_footprint_frac_100m  (regression, Spearman rho)
  walkability_index_v2                   (regression, Spearman rho -- HEADLINE)
walkability_index_v2 is NOT in the FLA; it is joined from walkability_index.csv
on (point_id, city).

Outputs
-------
  project/artifacts/reports/geo_baseline_results.json  -- one record per
      (eval_mode x feature_set x train_set x test_city x seed x model) with a
      results_per_target block comparable to multitarget_results.jsonl.
  project/artifacts/reports/geo_baseline_summary.json  -- seed-averaged headline
      walk-index comparison against the frozen-imagery anchors + decision verdict.

Run
---
    & ".venv\\Scripts\\python.exe" project\\scripts\\train_geo_baseline.py            # full sweep
    & ".venv\\Scripts\\python.exe" project\\scripts\\train_geo_baseline.py --quick    # seed 42, latlon only
    & ".venv\\Scripts\\python.exe" project\\scripts\\train_geo_baseline.py --dry-run  # print the run plan
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
WALK_INDEX_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "walkability_index.csv"
LOCK_DIR = REPO_ROOT / "project" / "data" / "processed" / "locks"
REPORT_DIR = REPO_ROOT / "project" / "artifacts" / "reports"
RESULTS_PATH = REPORT_DIR / "geo_baseline_results.json"
SUMMARY_PATH = REPORT_DIR / "geo_baseline_summary.json"

CITIES = ["msp", "seattle", "dc", "pittsburgh"]
CROSS_CITY_TRIO = ["msp", "seattle", "dc"]
PITTSBURGH_TRAIN = ["msp", "seattle", "dc"]

SEEDS_DEFAULT = [41, 42, 43]
KNN_NEIGHBORS = 15
RF_N_ESTIMATORS = 300
MODEL_NAMES = ["random_forest", "hist_gbdt", "knn"]

LATLON_FEATURES = ["lat", "lon"]
# Optional geo+context comparison. PLACES vars (LPA/OBESITY) are null for
# Pittsburgh -> median-imputed from the training cities at test time; that only
# affects the geo_context variant, never the lat/lon-only headline.
GEO_CONTEXT_FEATURES = [
    "lat", "lon",
    "D1A", "D1B", "D2B_E8MIX", "D3A", "D3B", "D3APO", "D4A", "D4C", "D4D",
    "stops_400m", "trips_per_hr_400m",
    "LPA_CrudePrev", "OBESITY_CrudePrev",
]
FEATURE_SETS = {"latlon": LATLON_FEATURES, "geo_context": GEO_CONTEXT_FEATURES}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str   # "binary" | "regression"
    source: str  # "fla" | "walk_index"


TARGETS: list[TargetSpec] = [
    TargetSpec("overture_sidewalk_present", "binary", "fla"),
    TargetSpec("overture_crosswalk_present", "binary", "fla"),
    TargetSpec("overture_intersection_count_200m", "regression", "fla"),
    TargetSpec("overture_building_footprint_frac_100m", "regression", "fla"),
    TargetSpec("walkability_index_v2", "regression", "walk_index"),
]
HEADLINE_TARGET = "walkability_index_v2"

# Frozen-imagery walk-index Spearman rho the baseline must be compared against.
IMAGERY_ANCHORS = {
    "cross_city_trio": {
        "siglip_frozen": 0.732,
        "clip_frozen": 0.713,
        "source": "docs/RESULTS.md / bootstrap_summary.json (walk-index rho, mean over 6 MSP/Seattle/DC LOCO directions)",
    },
    "pittsburgh_holdout": {
        "frozen_rho_low": 0.833,
        "frozen_rho_high": 0.862,
        "source": "CLAUDE.md SRC-3 / bootstrap_summary.json (Pittsburgh zero-shot, frozen backbones x 3 seeds)",
    },
}
DECISION_MARGIN = 0.03


def primary_metric_key(kind: str) -> str:
    return "auroc" if kind == "binary" else "spearman_rho"


def load_dataset() -> pd.DataFrame:
    """Load the 4-city FLA, join walkability_index_v2, filter to the per-city locks."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"{LABELS_CSV} not found. Build the labels file first.")
    if not WALK_INDEX_CSV.exists():
        raise FileNotFoundError(f"{WALK_INDEX_CSV} not found. Run the walkability-index build first.")

    fla = pd.read_csv(LABELS_CSV, low_memory=False)
    fla["point_id"] = fla["point_id"].astype(int)
    walk_index = pd.read_csv(WALK_INDEX_CSV)
    walk_index["point_id"] = walk_index["point_id"].astype(int)

    merged = fla.merge(
        walk_index[["point_id", "city", "walkability_index_v2"]],
        on=["point_id", "city"],
        how="left",
    )

    required = (
        set(LATLON_FEATURES)
        | {t.name for t in TARGETS if t.source == "fla"}
        | {HEADLINE_TARGET, "city", "point_id"}
    )
    missing = sorted(required - set(merged.columns))
    if missing:
        raise KeyError(f"Missing required columns in joined dataset: {missing}")
    if merged[HEADLINE_TARGET].isna().any():
        n_missing = int(merged[HEADLINE_TARGET].isna().sum())
        raise ValueError(
            f"{n_missing} rows missing {HEADLINE_TARGET} after join on (point_id, city). "
            "Check walkability_index.csv coverage."
        )

    kept: list[pd.DataFrame] = []
    for city in CITIES:
        lock_path = LOCK_DIR / f"{city}_v2_final_lock_ids.txt"
        if not lock_path.exists():
            raise FileNotFoundError(f"{lock_path} not found.")
        lock_ids = {int(x) for x in lock_path.read_text().split()}
        city_rows = merged[merged["city"] == city]
        locked = city_rows[city_rows["point_id"].isin(lock_ids)]
        dropped = len(city_rows) - len(locked)
        logging.info(
            "city=%-10s FLA=%4d  lock=%4d  kept=%4d%s",
            city, len(city_rows), len(lock_ids), len(locked),
            f"  (dropped {dropped} not in lock)" if dropped else "",
        )
        kept.append(locked)

    dataset = pd.concat(kept, ignore_index=True)
    logging.info("Dataset ready: %d rows across %d cities.", len(dataset), dataset["city"].nunique())
    return dataset


def build_estimator(model_name: str, kind: str, seed: int):
    is_binary = kind == "binary"
    if model_name == "random_forest":
        cls = RandomForestClassifier if is_binary else RandomForestRegressor
        return cls(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=seed)
    if model_name == "hist_gbdt":
        cls = HistGradientBoostingClassifier if is_binary else HistGradientBoostingRegressor
        return cls(random_state=seed)
    if model_name == "knn":
        cls = KNeighborsClassifier if is_binary else KNeighborsRegressor
        return cls(n_neighbors=KNN_NEIGHBORS, weights="distance")
    raise ValueError(f"Unknown model_name: {model_name!r}")


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute (train stats) then StandardScaler (fit on train cities only)."""
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)

    fill = np.nanmedian(X_train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    X_train = np.where(np.isnan(X_train), fill, X_train)
    X_test = np.where(np.isnan(X_test), fill, X_test)

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def evaluate_binary(y_true: np.ndarray, proba: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    out: dict[str, object] = {
        "type": "binary",
        "n": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 4),
    }
    if len(np.unique(y_true)) < 2:
        out.update({"auroc": None, "avg_precision": None, "note": "single_class_test_set"})
        return out
    out["auroc"] = round(float(roc_auc_score(y_true, proba)), 4)
    out["avg_precision"] = round(float(average_precision_score(y_true, proba)), 4)
    return out


def evaluate_regression(y_true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    # A constant prediction carries no rank signal; report rho=0 and flag it
    # rather than emitting a NaN (StandardScaler pushes a held-out city far
    # outside the training leaves, collapsing trees/KNN to a near-constant).
    constant_pred = bool(np.std(pred) < 1e-12)
    if constant_pred:
        rho, pearson = 0.0, 0.0
    else:
        rho = float(scipy.stats.spearmanr(pred, y_true)[0])
        pearson = float(scipy.stats.pearsonr(pred, y_true)[0])
    return {
        "type": "regression",
        "n": int(len(y_true)),
        "spearman_rho": round(rho, 4),
        "pearson_r": round(pearson, 4),
        "r2": round(float(r2_score(y_true, pred)), 4),
        "rmse": round(float(np.sqrt(np.mean((pred - y_true) ** 2))), 4),
        "mae": round(float(np.mean(np.abs(pred - y_true))), 4),
        "pred_is_constant": constant_pred,
    }


def fit_eval_all_targets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
    seed: int,
) -> dict[str, dict]:
    """Fit every (model, target) on the same prepared features; return per-model results."""
    X_train, X_test = prepare_features(train_df, test_df, feature_cols)
    per_model: dict[str, dict] = {}
    for model_name in model_names:
        results_per_target: dict[str, dict] = {}
        for tspec in TARGETS:
            y_train = train_df[tspec.name].to_numpy()
            y_test = test_df[tspec.name].to_numpy()
            estimator = build_estimator(model_name, tspec.kind, seed)
            if tspec.kind == "binary":
                estimator.fit(X_train, y_train.astype(int))
                proba = estimator.predict_proba(X_test)[:, 1]
                results_per_target[tspec.name] = evaluate_binary(y_test, proba)
            else:
                estimator.fit(X_train, y_train.astype(np.float64))
                pred = estimator.predict(X_test)
                results_per_target[tspec.name] = evaluate_regression(y_test, pred)
        per_model[model_name] = results_per_target
    return per_model


def run_holdout(
    dataset: pd.DataFrame,
    train_cities: list[str],
    test_city: str,
    eval_mode: str,
    feature_set_name: str,
    model_names: list[str],
    seed: int,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    feature_cols = FEATURE_SETS[feature_set_name]
    train_df = dataset[dataset["city"].isin(train_cities)]
    test_df = dataset[dataset["city"] == test_city]
    per_model = fit_eval_all_targets(train_df, test_df, feature_cols, model_names, seed)
    return [
        {
            "eval_mode": eval_mode,
            "feature_set": feature_set_name,
            "model": model_name,
            "train_cities": list(train_cities),
            "test_city": test_city,
            "seed": seed,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": len(feature_cols),
            "results_per_target": results_per_target,
        }
        for model_name, results_per_target in per_model.items()
    ]


def run_in_city(
    dataset: pd.DataFrame,
    city: str,
    feature_set_name: str,
    model_names: list[str],
    seed: int,
) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    feature_cols = FEATURE_SETS[feature_set_name]
    city_df = dataset[dataset["city"] == city].reset_index(drop=True)
    perm = np.random.default_rng(seed).permutation(len(city_df))
    n_train = int(round(len(city_df) * 0.70))
    train_df = city_df.iloc[perm[:n_train]]
    test_df = city_df.iloc[perm[n_train:]]
    per_model = fit_eval_all_targets(train_df, test_df, feature_cols, model_names, seed)
    return [
        {
            "eval_mode": "in_city",
            "feature_set": feature_set_name,
            "model": model_name,
            "train_cities": [city],
            "test_city": city,
            "seed": seed,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": len(feature_cols),
            "results_per_target": results_per_target,
        }
        for model_name, results_per_target in per_model.items()
    ]


def build_plan() -> list[tuple[str, list[str], str]]:
    """Return (eval_mode, train_cities, test_city) tuples for the holdout protocol."""
    plan: list[tuple[str, list[str], str]] = []
    for train_city in CROSS_CITY_TRIO:
        for test_city in CROSS_CITY_TRIO:
            if train_city != test_city:
                plan.append(("cross_city", [train_city], test_city))
    plan.append(("pittsburgh_holdout", PITTSBURGH_TRAIN, "pittsburgh"))
    return plan


def aggregate_metric(
    records: list[dict], target: str, model: str, metric_key: str
) -> dict | None:
    values = [
        r["results_per_target"][target][metric_key]
        for r in records
        if r["model"] == model and r["results_per_target"][target].get(metric_key) is not None
    ]
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def best_model_for(
    records: list[dict], target: str, metric_key: str, model_names: list[str]
) -> tuple[str | None, dict | None]:
    best_name, best_stats = None, None
    for model_name in model_names:
        stats = aggregate_metric(records, target, model_name, metric_key)
        if stats is None:
            continue
        if best_stats is None or stats["mean"] > best_stats["mean"]:
            best_name, best_stats = model_name, stats
    return best_name, best_stats


def verdict(geo_rho: float | None, imagery_rho: float) -> dict[str, object]:
    if geo_rho is None:
        return {"verdict": "no_geo_estimate", "gap": None}
    gap = round(imagery_rho - geo_rho, 4)
    if gap > DECISION_MARGIN:
        label = "imagery_wins"
    elif gap < -DECISION_MARGIN:
        label = "CONCERN_geo_beats_imagery"
    else:
        label = "CONCERN_geo_competitive"
    return {"verdict": label, "imagery_rho": imagery_rho, "geo_best_rho": geo_rho, "gap": gap}


def summarize(records: list[dict], model_names: list[str]) -> dict:
    latlon = [r for r in records if r["feature_set"] == "latlon"]
    cross = [r for r in latlon if r["eval_mode"] == "cross_city"]
    pgh = [r for r in latlon if r["eval_mode"] == "pittsburgh_holdout"]
    in_city = [r for r in latlon if r["eval_mode"] == "in_city"]

    metric_key = primary_metric_key("regression")  # walk index is regression

    cross_per_model = {m: aggregate_metric(cross, HEADLINE_TARGET, m, metric_key) for m in model_names}
    cross_best_name, cross_best = best_model_for(cross, HEADLINE_TARGET, metric_key, model_names)
    pgh_per_model = {m: aggregate_metric(pgh, HEADLINE_TARGET, m, metric_key) for m in model_names}
    pgh_best_name, pgh_best = best_model_for(pgh, HEADLINE_TARGET, metric_key, model_names)
    in_city_per_model = {m: aggregate_metric(in_city, HEADLINE_TARGET, m, metric_key) for m in model_names}

    headline = {
        "target": HEADLINE_TARGET,
        "feature_set": "latlon",
        "cross_city_trio": {
            "per_model_rho": cross_per_model,
            "best_model": cross_best_name,
            "best_rho": cross_best["mean"] if cross_best else None,
            **verdict(
                cross_best["mean"] if cross_best else None,
                IMAGERY_ANCHORS["cross_city_trio"]["siglip_frozen"],
            ),
        },
        "pittsburgh_holdout": {
            "per_model_rho": pgh_per_model,
            "best_model": pgh_best_name,
            "best_rho": pgh_best["mean"] if pgh_best else None,
            **verdict(
                pgh_best["mean"] if pgh_best else None,
                IMAGERY_ANCHORS["pittsburgh_holdout"]["frozen_rho_low"],
            ),
        },
        "in_city_reference": {
            "per_model_rho": in_city_per_model,
            "note": "Same-city 70/30 split: geography predicts within a city. Not a transfer number.",
        },
    }

    # All 5 targets, best geo model, latlon, for both holdout flavours.
    all_targets: dict[str, dict] = {}
    for tspec in TARGETS:
        mkey = primary_metric_key(tspec.kind)
        cross_name, cross_stats = best_model_for(cross, tspec.name, mkey, model_names)
        pgh_name, pgh_stats = best_model_for(pgh, tspec.name, mkey, model_names)
        inc_name, inc_stats = best_model_for(in_city, tspec.name, mkey, model_names)
        all_targets[tspec.name] = {
            "metric": mkey,
            "cross_city_trio": {"best_model": cross_name, **(cross_stats or {})},
            "pittsburgh_holdout": {"best_model": pgh_name, **(pgh_stats or {})},
            "in_city": {"best_model": inc_name, **(inc_stats or {})},
        }

    summary = {
        "config": {
            "seeds": sorted({r["seed"] for r in records}),
            "models": model_names,
            "feature_sets": sorted({r["feature_set"] for r in records}),
            "knn_neighbors": KNN_NEIGHBORS,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "decision_margin": DECISION_MARGIN,
            "n_records": len(records),
        },
        "imagery_anchors": IMAGERY_ANCHORS,
        "headline_walkability_index": headline,
        "all_targets_latlon": all_targets,
    }

    if any(r["feature_set"] == "geo_context" for r in records):
        geo_ctx = [r for r in records if r["feature_set"] == "geo_context"]
        gc_cross = [r for r in geo_ctx if r["eval_mode"] == "cross_city"]
        gc_pgh = [r for r in geo_ctx if r["eval_mode"] == "pittsburgh_holdout"]
        gc_cross_name, gc_cross_stats = best_model_for(gc_cross, HEADLINE_TARGET, metric_key, model_names)
        gc_pgh_name, gc_pgh_stats = best_model_for(gc_pgh, HEADLINE_TARGET, metric_key, model_names)
        summary["geo_context_walkability_index"] = {
            "note": "Optional geo+context comparison (census/GTFS features added to lat/lon).",
            "cross_city_trio": {
                "best_model": gc_cross_name,
                "best_rho": gc_cross_stats["mean"] if gc_cross_stats else None,
                **verdict(
                    gc_cross_stats["mean"] if gc_cross_stats else None,
                    IMAGERY_ANCHORS["cross_city_trio"]["siglip_frozen"],
                ),
            },
            "pittsburgh_holdout": {
                "best_model": gc_pgh_name,
                "best_rho": gc_pgh_stats["mean"] if gc_pgh_stats else None,
                **verdict(
                    gc_pgh_stats["mean"] if gc_pgh_stats else None,
                    IMAGERY_ANCHORS["pittsburgh_holdout"]["frozen_rho_low"],
                ),
            },
        }
    return summary


def log_walkindex_table(records: list[dict], model_names: list[str]) -> None:
    """Compact per-direction walk-index rho table (lat/lon) for the console."""
    latlon = [r for r in records if r["feature_set"] == "latlon"]
    if not latlon:
        return
    logging.info("--- walk-index (lat/lon) Spearman rho, mean over seeds ---")
    header = f"{'eval_mode':<18}{'train->test':<22}" + "".join(f"{m:>16}" for m in model_names)
    logging.info(header)
    seen: list[tuple[str, str]] = []
    for rec in latlon:
        key = (rec["eval_mode"], f"{'+'.join(rec['train_cities'])}->{rec['test_city']}")
        if key in seen:
            continue
        seen.append(key)
        subset = [
            r for r in latlon
            if r["eval_mode"] == rec["eval_mode"]
            and r["train_cities"] == rec["train_cities"]
            and r["test_city"] == rec["test_city"]
        ]
        cells = ""
        for model_name in model_names:
            stats = aggregate_metric(subset, HEADLINE_TARGET, model_name, "spearman_rho")
            cells += f"{stats['mean']:>16.4f}" if stats else f"{'-':>16}"
        logging.info(f"{key[0]:<18}{key[1]:<22}{cells}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coordinates-only geo baseline (city holdout, 5 targets).")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT, help="Random seeds (default 41 42 43).")
    p.add_argument(
        "--feature-sets", nargs="+", default=["latlon", "geo_context"],
        choices=list(FEATURE_SETS), help="Feature sets to run (default both).",
    )
    p.add_argument(
        "--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES,
        help="Models to run (default all three).",
    )
    p.add_argument("--no-in-city", action="store_true", help="Skip the supplementary in-city diagonal.")
    p.add_argument("--quick", action="store_true", help="Smoke test: seed 42, latlon only.")
    p.add_argument("--dry-run", action="store_true", help="Print the run plan and exit without fitting.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    seeds = [42] if args.quick else args.seeds
    feature_sets = ["latlon"] if args.quick else args.feature_sets
    model_names = [m for m in MODEL_NAMES if m in args.models]

    holdout_plan = build_plan()
    in_city_cities = [] if args.no_in_city else CITIES
    n_records = len(feature_sets) * len(seeds) * (len(holdout_plan) + len(in_city_cities)) * len(model_names)

    logging.info("Seeds=%s  feature_sets=%s  models=%s", seeds, feature_sets, model_names)
    logging.info(
        "Plan: %d holdout dirs + %d in-city, x %d feature_sets x %d seeds x %d models = %d records",
        len(holdout_plan), len(in_city_cities), len(feature_sets), len(seeds), len(model_names), n_records,
    )
    for eval_mode, train_cities, test_city in holdout_plan:
        logging.info("  [%s] %s -> %s", eval_mode, "+".join(train_cities), test_city)
    if in_city_cities:
        logging.info("  [in_city] %s (70/30 split)", ", ".join(in_city_cities))
    if args.dry_run:
        logging.info("Dry run -- no models fitted.")
        return 0

    dataset = load_dataset()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for feature_set_name in feature_sets:
        for seed in seeds:
            for eval_mode, train_cities, test_city in holdout_plan:
                records.extend(
                    run_holdout(dataset, train_cities, test_city, eval_mode,
                                feature_set_name, model_names, seed)
                )
            for city in in_city_cities:
                records.extend(run_in_city(dataset, city, feature_set_name, model_names, seed))
            logging.info("Done: feature_set=%s seed=%d (%d records so far)", feature_set_name, seed, len(records))

    RESULTS_PATH.write_text(json.dumps(records, indent=2))
    logging.info("Wrote %d run records -> %s", len(records), RESULTS_PATH)

    summary = summarize(records, model_names)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote summary -> %s", SUMMARY_PATH)

    log_walkindex_table(records, model_names)

    head = summary["headline_walkability_index"]
    cc = head["cross_city_trio"]
    pg = head["pittsburgh_holdout"]
    inc = head["in_city_reference"]["per_model_rho"]
    inc_best = max((s["mean"] for s in inc.values() if s), default=float("nan"))
    logging.info("=" * 72)
    logging.info("HEADLINE -- walkability_index_v2, lat/lon only:")
    logging.info(
        "  cross-city (MSP/SEA/DC):  geo best rho=%.4f (%s)  vs imagery %.3f  gap=%.4f  -> %s",
        cc["best_rho"] if cc["best_rho"] is not None else float("nan"),
        cc["best_model"], cc["imagery_rho"], cc["gap"], cc["verdict"],
    )
    logging.info(
        "  Pittsburgh zero-shot:     geo best rho=%.4f (%s)  vs imagery %.3f  gap=%.4f  -> %s",
        pg["best_rho"] if pg["best_rho"] is not None else float("nan"),
        pg["best_model"], pg["imagery_rho"], pg["gap"], pg["verdict"],
    )
    logging.info("  in-city reference (geography is local): best rho=%.4f", inc_best)
    logging.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
