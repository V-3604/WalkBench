"""
Geography-only Ridge baseline for WalkCLIP v2 (Diagnostic Experiment 2.2).

Answers: how well does NatWalkInd (and other EPA targets) predict from lat/lon
and census/GTFS features alone, with zero image information?

This baseline is essential for the paper — images must beat it, not just beat zero.

Run:
    # Cross-city
    python project/scripts/train_geo_baseline.py --train-city msp --test-city seattle
    python project/scripts/train_geo_baseline.py --train-city seattle --test-city msp

    # In-city
    python project/scripts/train_geo_baseline.py --in-city-city msp
    python project/scripts/train_geo_baseline.py --in-city-city seattle

    # All four at once
    python project/scripts/train_geo_baseline.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"
REPORT_DIR = REPO_ROOT / "project" / "artifacts" / "reports"

SEED = 42

GEO_ONLY_FEATURES = [
    "lat",
    "lon",
    "D1A",
    "D1B",
    "D2B_E8MIX",
    "D3A",
    "D3B",
    "D3APO",
    "D4A",
    "D4C",
    "D4D",
    "stops_400m",
    "trips_per_hr_400m",
    "LPA_CrudePrev",
    "OBESITY_CrudePrev",
]

PRIMARY_TARGET = "NatWalkInd"
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def evaluate_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r2 = float(r2_score(y_true, pred))
    rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(pred - y_true)))
    pearson_r, _ = scipy.stats.pearsonr(pred, y_true)
    spearman_rho, _ = scipy.stats.spearmanr(pred, y_true)
    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
    }


def evaluate_recalibrated(
    y_true: np.ndarray,
    pred: np.ndarray,
    train_mean: float,
    train_std: float,
) -> dict[str, float]:
    """Z-score pred using train stats, rescale to test distribution, recompute R²."""
    pred_z = (pred - train_mean) / max(train_std, 1e-8)
    test_mean = float(np.mean(y_true))
    test_std = float(np.std(y_true))
    pred_recal = pred_z * test_std + test_mean
    r2_recal = float(r2_score(y_true, pred_recal))
    rmse_recal = float(np.sqrt(np.mean((pred_recal - y_true) ** 2)))
    return {
        "r2_recalibrated": round(r2_recal, 4),
        "rmse_recalibrated": round(rmse_recal, 4),
    }


def evaluate_block_group_aggregated(
    point_ids: np.ndarray,
    pred: np.ndarray,
    label_frame: pd.DataFrame,
) -> dict[str, float]:
    """Average point-level preds within block groups, then compute BG-level R²."""
    df = pd.DataFrame(
        {
            "point_id": point_ids,
            "pred": pred,
            "bg": label_frame.loc[point_ids, "blockgroup_geoid"].values,
            "true": label_frame.loc[point_ids, PRIMARY_TARGET].values,
        }
    )
    bg = (
        df.groupby("bg")
        .agg(pred_mean=("pred", "mean"), true_mean=("true", "first"))
        .dropna()
    )
    if len(bg) < 2:
        return {"bg_r2": float("nan"), "bg_pearson_r": float("nan"), "n_block_groups": len(bg)}
    bg_r2 = float(r2_score(bg["true_mean"], bg["pred_mean"]))
    bg_r, _ = scipy.stats.pearsonr(bg["pred_mean"], bg["true_mean"])
    return {
        "bg_r2": round(bg_r2, 4),
        "bg_pearson_r": round(float(bg_r), 4),
        "n_block_groups": int(len(bg)),
    }


def prepare_city(
    city: str,
    labels: pd.DataFrame,
    lock_ids: list[int],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return X, y, point_ids, label_frame for a city."""
    city_df = labels[labels["city"] == city].copy()
    city_df["point_id"] = city_df["point_id"].astype(int)
    city_df = city_df[city_df["point_id"].isin(lock_ids)].set_index("point_id")
    present_ids = np.array([pid for pid in lock_ids if pid in city_df.index], dtype=np.int32)
    if len(present_ids) == 0:
        raise ValueError(f"No locked rows found for city={city}.")

    X = city_df.loc[present_ids, feature_cols].to_numpy(dtype=np.float64)
    y = city_df.loc[present_ids, PRIMARY_TARGET].to_numpy(dtype=np.float64)
    return X, y, present_ids, city_df


def run_cross_city(
    train_city: str,
    test_city: str,
    labels: pd.DataFrame,
    lock_ids: list[int],
    feature_cols: list[str],
) -> dict[str, object]:
    rng = np.random.default_rng(SEED)

    X_train_full, y_train_full, train_ids, train_frame = prepare_city(
        train_city, labels, lock_ids, feature_cols
    )
    X_test, y_test, test_ids, test_frame = prepare_city(
        test_city, labels, lock_ids, feature_cols
    )

    # 85/15 train/val split on training city
    n = len(y_train_full)
    perm = rng.permutation(n)
    n_train = int(round(n * 0.85))
    train_idx = perm[:n_train]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

    # Impute missing with training median, scale
    fill = np.nanmedian(X_train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    X_train_f = np.where(np.isnan(X_train), fill, X_train)
    X_test_f = np.where(np.isnan(X_test), fill, X_test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_train_s, y_train)
    train_mean = float(np.mean(y_train))
    train_std = float(np.std(y_train))

    pred_test = model.predict(X_test_s).astype(np.float64)
    metrics = evaluate_metrics(y_test, pred_test)
    recal = evaluate_recalibrated(y_test, pred_test, train_mean, train_std)
    bg = evaluate_block_group_aggregated(test_ids, pred_test, test_frame)

    return {
        "eval_mode": "cross_city",
        "model": "ridge_cv",
        "ablation": "geo_only",
        "train_city": train_city,
        "test_city": test_city,
        "selected_alpha": float(model.alpha_),
        "train_mean": round(train_mean, 4),
        "train_std": round(train_std, 4),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "primary_metrics": {**metrics, **recal},
        "block_group_metrics": bg,
    }


def run_in_city(
    city: str,
    labels: pd.DataFrame,
    lock_ids: list[int],
    feature_cols: list[str],
) -> dict[str, object]:
    rng = np.random.default_rng(SEED)

    X_full, y_full, point_ids, label_frame = prepare_city(
        city, labels, lock_ids, feature_cols
    )

    # 70/15/15 split
    n = len(y_full)
    perm = rng.permutation(n)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]
    X_test = X_full[test_idx]
    y_test = y_full[test_idx]
    test_ids = point_ids[test_idx]

    fill = np.nanmedian(X_train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    X_train_f = np.where(np.isnan(X_train), fill, X_train)
    X_val_f = np.where(np.isnan(X_val), fill, X_val)
    X_test_f = np.where(np.isnan(X_test), fill, X_test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f)
    X_val_s = scaler.transform(X_val_f)
    X_test_s = scaler.transform(X_test_f)

    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    model.fit(X_train_s, y_train)
    train_mean = float(np.mean(y_train))
    train_std = float(np.std(y_train))

    pred_val = model.predict(X_val_s).astype(np.float64)
    pred_test = model.predict(X_test_s).astype(np.float64)
    val_metrics = evaluate_metrics(y_val, pred_val)
    test_metrics = evaluate_metrics(y_test, pred_test)
    recal = evaluate_recalibrated(y_test, pred_test, train_mean, train_std)
    bg = evaluate_block_group_aggregated(test_ids, pred_test, label_frame)

    return {
        "eval_mode": "in_city",
        "model": "ridge_cv",
        "ablation": "geo_only",
        "train_city": city,
        "test_city": city,
        "selected_alpha": float(model.alpha_),
        "train_mean": round(train_mean, 4),
        "train_std": round(train_std, 4),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "val_metrics": val_metrics,
        "primary_metrics": {**test_metrics, **recal},
        "block_group_metrics": bg,
    }


def print_result(res: dict[str, object]) -> None:
    m = res["primary_metrics"]
    bg = res["block_group_metrics"]
    logging.info(
        "%-12s %-8s %-8s | R²=%.4f  R²_recal=%.4f  ρ=%.4f  RMSE=%.4f  BG_R²=%.4f  (n_BG=%d)  α=%.3g",
        res["eval_mode"],
        res["train_city"],
        res["test_city"],
        m["r2"],
        m.get("r2_recalibrated", float("nan")),
        m["spearman_rho"],
        m["rmse"],
        bg["bg_r2"],
        bg["n_block_groups"],
        res["selected_alpha"],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Geography-only Ridge baseline for WalkCLIP v2")
    p.add_argument("--train-city", choices=["msp", "seattle"])
    p.add_argument("--test-city", choices=["msp", "seattle"])
    p.add_argument("--in-city-city", choices=["msp", "seattle"])
    p.add_argument("--all", action="store_true", help="Run all four combinations")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in (LABELS_CSV, LOCK_FILE):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            return 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lock_ids = [int(x) for x in LOCK_FILE.read_text().strip().splitlines()]
    labels = pd.read_csv(LABELS_CSV)
    labels["point_id"] = labels["point_id"].astype(int)

    missing_cols = [c for c in GEO_ONLY_FEATURES if c not in labels.columns]
    if missing_cols:
        logging.error("Missing columns in labels file: %s", missing_cols)
        return 1

    feature_cols = GEO_ONLY_FEATURES

    runs: list[dict[str, object]] = []

    if args.all:
        for train_city, test_city in [("msp", "seattle"), ("seattle", "msp")]:
            res = run_cross_city(train_city, test_city, labels, lock_ids, feature_cols)
            runs.append(res)
            print_result(res)
        for city in ["msp", "seattle"]:
            res = run_in_city(city, labels, lock_ids, feature_cols)
            runs.append(res)
            print_result(res)
    elif args.in_city_city:
        res = run_in_city(args.in_city_city, labels, lock_ids, feature_cols)
        runs.append(res)
        print_result(res)
    elif args.train_city and args.test_city:
        if args.train_city == args.test_city:
            logging.error("Use --in-city-city for same-city runs.")
            return 1
        res = run_cross_city(args.train_city, args.test_city, labels, lock_ids, feature_cols)
        runs.append(res)
        print_result(res)
    else:
        logging.error("Provide --train-city/--test-city, --in-city-city, or --all.")
        return 1

    out_path = REPORT_DIR / "geo_baseline_results.json"
    existing: list[dict[str, object]] = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())

    # Deduplicate by (eval_mode, train_city, test_city)
    key_fn = lambda r: (r["eval_mode"], r["train_city"], r["test_city"])
    existing_keys = {key_fn(r) for r in existing}
    for r in runs:
        if key_fn(r) not in existing_keys:
            existing.append(r)
        else:
            existing = [r if key_fn(e) == key_fn(r) else e for e in existing]

    out_path.write_text(json.dumps(existing, indent=2))
    logging.info("Results saved: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
