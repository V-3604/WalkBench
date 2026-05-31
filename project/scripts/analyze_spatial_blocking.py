"""
Spatially blocked in-city cross-validation for WalkCLIP v2 (Section 4.6).

Replaces random 70/15/15 splits with leakage-controlled spatial folds.

Block construction
------------------
1. Project (lat, lon) → local equirectangular metres anchored at the city centroid
   (avoids anisotropic clusters at high latitudes, where 1° lon ≪ 1° lat in metres).
2. KMeans(n_clusters=n_folds) in projected metres assigns each point to a spatial fold.
3. For each outer fold k: test = fold k, pool = other folds. Pool is split 80/20 into
   train/val (random within pool, seeded per fold).
4. **Buffer**: training and val points within ``--block-buffer-m`` metres of any test
   point are dropped (Euclidean in projected metres, KDTree). This closes two leakage
   paths simultaneously:
     (a) Pedgraph kNN feature contamination — pedgraph_features.csv aggregates BASE_COLS
         from each point's k=8 graph neighbors within max_graph_distance_m=400 m. Any
         training point within 400 m of a test point may have contributed to that test
         point's feature vector. Buffer ≥ 400 m closes this.
     (b) Target spatial autocorrelation — built-environment targets decorrelate over
         a few hundred metres. The buffer enforces a clear gap between train and test.
   Default buffer = 500 m (≥400 m for (a), with 100 m margin for (b)). Use
   ``--block-buffer-m 0`` to disable for legacy comparisons.

Metrics (per fold, then aggregated mean ± std):
  Regression: R², R²_recal (post-hoc affine recalibration to test mean/std — *uses test
              labels and is therefore an upper-bound rank-faithful score, not a deployed
              R²; report alongside Spearman ρ*), Spearman ρ, RMSE, MAE.
  Binary:     AUROC, AvgPrecision, F1@0.5, Accuracy@0.5. Threshold-based F1/Acc are
              naive on raw Ridge scores; treat AUROC/AP as the primary signals.

Outputs:
  project/artifacts/reports/spatial_cv_results.jsonl  (append; full record per run, incl.
                                                       block_buffer_m and per-fold drops)
  project/artifacts/reports/spatial_cv_summary.json   (latest summary, overwritten)

Run:
    python project/scripts/analyze_spatial_blocking.py --city dc
    python project/scripts/analyze_spatial_blocking.py --city msp --block-buffer-m 500
    python project/scripts/analyze_spatial_blocking.py --city seattle --block-buffer-m 0   # legacy
    python project/scripts/analyze_spatial_blocking.py --city dc --target NatWalkInd
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import LinAlgWarning
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from walkclip_stage_e_v2 import (
    FEATURES_BY_ABLATION,
    LABELS_CSV,
    CAPTION_CSV,
    SPATIAL_CSV,
    LOCK_FILE,
    EMBED_DIR,
    LOG_DIR,
    REPORT_DIR,
    SEED,
    ArrayScaler,
    set_seeds,
    tabular_fill_and_scale,
    prepare_city_data,
    build_feature_matrices,
    feature_groups_for_run,
    SAM2_INPUT_COLS,
)
from train_multitarget import TARGET_SPECS, fit_target_scalers, extract_multitarget_labels


def project_to_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project (lat, lon) in degrees to local equirectangular meters anchored at the
    city centroid. Adequate for kNN/blocking at city scale (errors <0.5% over <50 km).
    Returns array of shape (n, 2) with columns (x_m, y_m)."""
    lat0 = float(np.mean(lat))
    R = 6_378_137.0  # WGS84 mean radius (m)
    x = np.radians(lon - float(np.mean(lon))) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return np.column_stack([x, y]).astype(np.float64)


def spatial_kfold(lat: np.ndarray, lon: np.ndarray, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """KMeans fold assignment in projected meters. Returns fold ids, shape (n_points,).

    Projecting to meters before clustering avoids the anisotropy of clustering directly
    on (lat, lon) degrees (a 1° lat ≈ 111 km but a 1° lon ≈ 76 km at Seattle's 47°N,
    yielding north-south-elongated clusters). Block geometry should be isotropic for
    spatial CV blocking to be physically meaningful.
    """
    coords_m = project_to_meters(lat, lon)
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit(coords_m)
    return km.labels_


def buffer_filter_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    coords_m: np.ndarray,
    buffer_m: float,
) -> tuple[np.ndarray, int]:
    """Remove training indices that fall within ``buffer_m`` of any test point in
    Euclidean (projected meters) distance. Returns (kept_train_idx, n_dropped).

    Why this exists
    ---------------
    The headline leakage closure for spatially blocked CV. Without a buffer, points in
    the training fold that sit within the same spatial-autocorrelation neighbourhood as
    test points contaminate the evaluation through two distinct mechanisms:

    1. Pedgraph kNN feature leakage: ``pedgraph_features.csv`` aggregates BASE_COLS
       from each point's k=8 graph neighbors within ``max_graph_distance_m`` (default
       400 m). A test point's own *input feature vector* therefore mixes in values
       computed from points that may sit in the training fold across the block boundary.
       Even though the buffer here drops those points from training (not from the
       test point's neighborhood), it bounds the joint dependency: the model is never
       fit on points whose features were aggregated from the same local
       neighborhood as any test point.

    2. Target spatial autocorrelation: built-environment targets (sidewalk presence,
       building footprint fraction) decorrelate with distance on a scale of a few
       hundred metres. Adjacent train/test pairs across a block boundary yield
       optimistic AUROC/R² that is structurally indistinguishable from a random split.

    For full closure of (1), buffer_m should be ≥ pedgraph max_graph_distance_m
    (currently 400 m). The default 500 m gives a 100 m margin and matches the
    spatial-autocorrelation scale.

    Note: this drops *training* points, never test points (test fold ids are sacrosanct
    so cross-fold metric aggregates remain comparable). With a 500 m buffer on city-
    sized 5-fold KMeans blocks the typical drop is 5–15% of the training set.
    """
    if buffer_m <= 0.0 or len(test_idx) == 0:
        return train_idx, 0

    # KDTree query: for each train point, distance to nearest test point.
    from scipy.spatial import cKDTree
    test_tree = cKDTree(coords_m[test_idx])
    train_coords = coords_m[train_idx]
    nearest_dist, _ = test_tree.query(train_coords, k=1)
    keep_mask = nearest_dist > buffer_m
    n_dropped = int((~keep_mask).sum())
    return train_idx[keep_mask], n_dropped


def _compute_fold_metrics(
    name: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    train_mean: float,
    train_std: float,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute per-fold metrics. For binary targets fit by Ridge (linear probability
    model), F1/Accuracy are derived using a *prevalence-matched* threshold computed
    from the training positive rate (Finding C in v2_CODE_AUDIT_2026-04-26.md).

    Concretely: threshold = (1 - p_train) quantile of the test predictions, where
    p_train is the positive rate in the training labels passed via ``y_train``. This
    forces the test predicted-positive-rate to match the training prior, which is the
    principled threshold for an unbounded linear-probability score (the legacy 0.5
    threshold is meaningless because Ridge outputs are not bounded to [0, 1]).

    AUROC and AvgPrecision are threshold-free and remain the primary signals.
    """
    valid = np.isfinite(y_true) & np.isfinite(pred)
    pred_v = pred[valid]
    y_v = y_true[valid]
    n = int(valid.sum())

    if n < 2:
        return {"n": n, "error": "insufficient_data"}

    spec_type = TARGET_SPECS[name]["type"]

    if spec_type == "binary":
        probs = pred_v
        try:
            auroc = float(roc_auc_score(y_v, probs))
        except Exception:
            auroc = float("nan")
        try:
            ap = float(average_precision_score(y_v, probs))
        except Exception:
            ap = float("nan")

        # Prevalence-matched threshold from train labels (no test leak).
        if y_train is not None:
            yt = y_train[np.isfinite(y_train)]
            if yt.size >= 2:
                p_train = float(np.clip(yt.mean(), 1e-3, 1 - 1e-3))
                threshold = float(np.quantile(probs, 1.0 - p_train))
            else:
                threshold = 0.5
        else:
            threshold = 0.5
        preds_bin = (probs >= threshold).astype(np.float32)
        f1 = float(f1_score(y_v, preds_bin, zero_division=0))
        acc = float(accuracy_score(y_v, preds_bin))
        return {
            "type": "binary",
            "n": n,
            "auroc": round(auroc, 4),
            "avg_precision": round(ap, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "threshold": round(threshold, 4),
            "threshold_rule": "prevalence_matched_train" if y_train is not None else "fixed_0.5",
        }
    else:
        r2 = float(r2_score(y_v, pred_v))
        rmse = float(np.sqrt(np.mean((pred_v - y_v) ** 2)))
        mae = float(np.mean(np.abs(pred_v - y_v)))
        spearman_rho, _ = stats.spearmanr(pred_v, y_v)

        pred_z = (pred_v - train_mean) / max(train_std, 1e-8)
        test_mean = float(np.mean(y_v))
        test_std = float(np.std(y_v))
        pred_recal = pred_z * test_std + test_mean
        r2_recal = float(r2_score(y_v, pred_recal))

        return {
            "type": "regression",
            "n": n,
            "r2": round(r2, 4),
            "r2_recalibrated": round(r2_recal, 4),
            "spearman_rho": round(float(spearman_rho), 4) if np.isfinite(spearman_rho) else float("nan"),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }


def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alpha: float | None,
) -> Ridge:
    valid = np.isfinite(y_train)
    X_tr = X_train[valid]
    y_tr = y_train[valid]

    if ridge_alpha is None:
        alphas = np.logspace(-3, 4, 30)
        cv = RidgeCV(alphas=alphas, cv=5)
        # The alpha sweep intentionally tries very small values (1e-3) on high-dimensional
        # PCA feature matrices, which causes Cholesky to see near-zero eigenvalues and emit
        # LinAlgWarning. This is expected and harmless — the final selected alpha is always
        # larger. Suppress here; use SVD solver for the final fit.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            cv.fit(X_tr, y_tr)
        best_alpha = float(cv.alpha_)
    else:
        best_alpha = ridge_alpha

    # solver='svd' is numerically stable for any alpha and any feature-matrix condition number
    model = Ridge(alpha=best_alpha, solver="svd")
    model.fit(X_tr, y_tr)
    return model


def _run_fold_ridge(
    fold_id: int,
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels_all: dict[str, np.ndarray],
    active_targets: list[str],
    ridge_alpha: float | None,
    pca_dim: int,
    feature_groups: tuple[str, ...],
) -> dict[str, object]:
    # Build simple train/test split data dicts (no modality groups needed for Ridge)
    X_tr = X[train_idx]
    X_te = X[test_idx]

    fold_metrics: dict[str, dict] = {}
    for name in active_targets:
        y_all = labels_all[name]
        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]

        valid_train = np.isfinite(y_tr)
        if valid_train.sum() < 5:
            fold_metrics[name] = {"n": int(valid_train.sum()), "error": "insufficient_train_data"}
            continue

        scaler = fit_target_scalers({name: y_tr}, [name])[name]
        train_mean = float(scaler.mean[0]) if scaler is not None else float(np.nanmean(y_tr))
        train_std = float(scaler.scale[0]) if scaler is not None else float(np.nanstd(y_tr))

        model = _fit_ridge(X_tr, y_tr, ridge_alpha)
        pred = model.predict(X_te).astype(np.float32)
        metrics = _compute_fold_metrics(name, pred, y_te, train_mean, train_std, y_train=y_tr)
        fold_metrics[name] = metrics

    return {
        "fold_id": fold_id,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "metrics": fold_metrics,
    }


def _run_fold_mlp(
    fold_id: int,
    city_data: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels_all: dict[str, np.ndarray],
    active_targets: list[str],
    pca_dim: int,
    feature_groups: tuple[str, ...],
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    n_epochs: int,
    patience: int,
    device,
) -> dict[str, object]:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset
    from train_multitarget import WalkCLIPMultiTarget, compute_multitarget_loss

    def _make_split(full_data: dict, idx: np.ndarray) -> dict:
        return {
            "groups": {k: v[idx] for k, v in full_data["groups"].items()},
            "targets": full_data["targets"],
            "point_ids": full_data["point_ids"][idx],
            "label_frame": full_data["label_frame"],
        }

    train_split = _make_split(city_data, train_idx)
    val_split = _make_split(city_data, val_idx)
    test_split = _make_split(city_data, test_idx)

    X_train, X_val, X_test, feature_transforms = build_feature_matrices(
        train_split, val_split, test_split,
        feature_groups=feature_groups,
        model_family="mlp",
        pca_dim=pca_dim,
    )
    group_dims: dict[str, int] = feature_transforms["group_dims"]
    street_dim = group_dims.get("street", 0)
    naip_dim = group_dims.get("naip", 0)
    tabular_dim = sum(v for k, v in group_dims.items() if k not in ("street", "naip"))

    train_labels_fold = {n: labels_all[n][train_idx] for n in active_targets}
    val_labels_fold = {n: labels_all[n][val_idx] for n in active_targets}
    test_labels_fold = {n: labels_all[n][test_idx] for n in active_targets}
    target_scalers = fit_target_scalers(train_labels_fold, active_targets)

    model = WalkCLIPMultiTarget(
        street_dim=street_dim,
        naip_dim=naip_dim,
        tabular_dim=tabular_dim,
        target_names=active_targets,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr))

    label_matrix_train = np.stack([train_labels_fold[n] for n in active_targets], axis=1)
    ds = TensorDataset(to_t(X_train), to_t(label_matrix_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_val_t = to_t(X_val).to(device)
    val_label_tensors = {n: to_t(val_labels_fold[n]) for n in active_targets}

    best_val = float("inf")
    best_state: dict = {}
    patience_left = patience

    for epoch in range(1, n_epochs + 1):
        model.train()
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            batch_tgts = {active_targets[i]: Yb[:, i].to(device) for i in range(len(active_targets))}
            loss = compute_multitarget_loss(model(Xb), batch_tgts, target_scalers, active_targets, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
        val_loss = float(
            compute_multitarget_loss(val_preds, val_label_tensors, target_scalers, active_targets, device).item()
        )
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    X_test_t = to_t(X_test).to(device)
    with torch.no_grad():
        test_preds = model(X_test_t)

    fold_metrics: dict[str, dict] = {}
    for name in active_targets:
        logits_np = test_preds[name].cpu().numpy()
        y_te = test_labels_fold[name]
        scaler = target_scalers[name]
        train_mean = float(scaler.mean[0]) if scaler is not None else 0.0
        train_std = float(scaler.scale[0]) if scaler is not None else 1.0

        if TARGET_SPECS[name]["type"] == "regression" and scaler is not None:
            pred = logits_np * scaler.scale[0] + scaler.mean[0]
        else:
            pred = logits_np

        fold_metrics[name] = _compute_fold_metrics(
            name, pred, y_te, train_mean, train_std,
            y_train=train_labels_fold[name],
        )

    return {
        "fold_id": fold_id,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "metrics": fold_metrics,
    }


def _aggregate_fold_results(
    per_fold_results: list[dict],
    active_targets: list[str],
) -> dict[str, dict]:
    """Compute mean ± std of each metric across folds per target."""
    summary: dict[str, dict] = {}
    for name in active_targets:
        spec_type = TARGET_SPECS[name]["type"]
        if spec_type == "binary":
            metric_keys = ["auroc", "avg_precision", "f1", "accuracy"]
        else:
            metric_keys = ["r2", "r2_recalibrated", "spearman_rho", "rmse", "mae"]

        fold_vals: dict[str, list[float]] = {k: [] for k in metric_keys}
        n_positive_r2 = 0
        for fold in per_fold_results:
            m = fold["metrics"].get(name, {})
            if "error" in m:
                continue
            for k in metric_keys:
                v = m.get(k)
                if v is not None and np.isfinite(v):
                    fold_vals[k].append(v)
            if spec_type == "regression" and m.get("r2", -999) > 0:
                n_positive_r2 += 1

        agg: dict[str, float] = {}
        for k, vals in fold_vals.items():
            if vals:
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"] = round(float(np.std(vals)), 4)
                agg[f"{k}_min"] = round(float(np.min(vals)), 4)
                agg[f"{k}_max"] = round(float(np.max(vals)), 4)
            else:
                agg[f"{k}_mean"] = float("nan")
                agg[f"{k}_std"] = float("nan")

        if spec_type == "regression":
            agg["n_positive_r2_folds"] = n_positive_r2

        summary[name] = {"type": spec_type, "n_folds_with_data": len(fold_vals[metric_keys[0]]), **agg}

    return summary


def print_spatial_cv_table(
    city: str,
    n_folds: int,
    model_family: str,
    summary: dict[str, dict],
) -> None:
    logging.info("")
    logging.info("─" * 110)
    logging.info("SPATIAL CV RESULTS  [city=%s  n_folds=%d  model=%s]", city, n_folds, model_family)
    logging.info("─" * 110)

    for tier in (1, 2, 3):
        tier_names = [n for n in summary if TARGET_SPECS[n]["tier"] == tier]
        if not tier_names:
            continue
        logging.info("Tier %d:", tier)
        for name in tier_names:
            m = summary[name]
            if m.get("type") == "binary":
                logging.info(
                    "  %-40s  AUROC=%.4f±%.4f  AP=%.4f±%.4f  F1=%.4f±%.4f",
                    name,
                    m.get("auroc_mean", float("nan")), m.get("auroc_std", float("nan")),
                    m.get("avg_precision_mean", float("nan")), m.get("avg_precision_std", float("nan")),
                    m.get("f1_mean", float("nan")), m.get("f1_std", float("nan")),
                )
            else:
                n_pos = m.get("n_positive_r2_folds", "?")
                logging.info(
                    "  %-40s  R²=%.4f±%.4f  R²_recal=%.4f±%.4f  ρ=%.4f±%.4f  RMSE=%.4f±%.4f  pos_folds=%s/%d",
                    name,
                    m.get("r2_mean", float("nan")), m.get("r2_std", float("nan")),
                    m.get("r2_recalibrated_mean", float("nan")), m.get("r2_recalibrated_std", float("nan")),
                    m.get("spearman_rho_mean", float("nan")), m.get("spearman_rho_std", float("nan")),
                    m.get("rmse_mean", float("nan")), m.get("rmse_std", float("nan")),
                    n_pos, n_folds,
                )
    logging.info("─" * 110)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WalkCLIP v2 spatially blocked cross-validation (Section 4.6)")
    p.add_argument("--city", choices=["msp", "seattle", "dc"], required=True)
    p.add_argument("--target", default=None, help="Single target column name (default: run all TARGET_SPECS)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--model-family", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--ablation", choices=sorted(FEATURES_BY_ABLATION), default="full")
    p.add_argument("--pca-dim", type=int, default=0)
    p.add_argument("--ridge-alpha", type=float, default=None,
                   help="Fixed Ridge alpha. Default: auto-select via RidgeCV on each training fold.")
    p.add_argument("--block-buffer-m", type=float, default=500.0,
                   help="Buffer in metres around each test fold. Training points within "
                        "this distance of any test point are dropped from the train set. "
                        "Set to 0 to disable (legacy behaviour, leakage-prone). Default 500m: "
                        "≥ pedgraph max_graph_distance_m (400m), closing pedgraph kNN feature leakage.")
    def _backbone_type(s: str) -> str:
        allowed = {"siglip", "clip", "siglip_lora",
                   "siglip_lora_loco_msp", "siglip_lora_loco_seattle", "siglip_lora_loco_dc",
                   "siglip_lora_loco_v2_msp", "siglip_lora_loco_v2_seattle", "siglip_lora_loco_v2_dc"}
        if s not in allowed:
            raise argparse.ArgumentTypeError(
                f"invalid backbone {s!r}; choose from {sorted(allowed)}")
        return s
    p.add_argument("--backbone", type=_backbone_type, default="siglip",
                   help="Image backbone tag. For LOCO under spatial CV, pass the adapter "
                        "tag whose holdout matches --city (e.g. --city dc --backbone "
                        "siglip_lora_loco_dc) so the in-city CV uses an adapter that has "
                        "never seen this city's heading labels.")
    # MLP-specific (only used with --model-family mlp)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (LOCK_FILE, "Lock file"),
        (LABELS_CSV, "Labels CSV"),
        (CAPTION_CSV, "Caption CSV"),
        (SPATIAL_CSV, "Spatial CSV"),
    ):
        if not path.exists():
            logging.error("%s not found: %s. Ensure preprocessing has run.", label, path)
            return 1

    expected = list(EMBED_DIR.glob(f"{args.backbone}_street_{args.city}.npy"))
    if not EMBED_DIR.exists() or not expected:
        logging.error(
            "No embeddings found for backbone=%s city=%s in %s.",
            args.backbone, args.city, EMBED_DIR,
        )
        return 1

    set_seeds(SEED)
    logging.info("seed=%d  city=%s  n_folds=%d  model=%s", SEED, args.city, args.n_folds, args.model_family)

    lock_ids = [int(x) for x in LOCK_FILE.read_text().strip().splitlines()]
    labels = pd.read_csv(LABELS_CSV)
    # Merge walkability_index.csv (same pattern as train_multitarget.py)
    wi_path = LABELS_CSV.parent / "walkability_index.csv"
    if wi_path.exists():
        _wi_cols = ["point_id", "city", "walkability_index_v1"]
        _wi_all = pd.read_csv(wi_path)
        if "walkability_index_v2" in _wi_all.columns:
            _wi_cols.append("walkability_index_v2")
        wi = _wi_all[_wi_cols]
        labels = labels.merge(wi, on=["point_id", "city"], how="left")
    else:
        logging.warning("walkability_index.csv not found; walkability_index_v1 target will be all-NaN.")
    captions = pd.read_csv(CAPTION_CSV)
    spatial = pd.read_csv(SPATIAL_CSV)
    for df in (labels, captions, spatial):
        df["point_id"] = df["point_id"].astype(int)

    if args.model_family == "mlp":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("GPU device: %s", device)
    else:
        device = None

    caption_cols = [c for c in captions.columns if c.startswith("cap_")]
    spatial_cols = [c for c in spatial.columns if c.startswith("spatial_")]
    feature_groups = feature_groups_for_run(args.ablation)

    logging.info("Loading city data for %s ...", args.city)
    city_data = prepare_city_data(
        args.city, labels, captions, spatial, lock_ids, caption_cols, spatial_cols,
        backbone=args.backbone,
    )
    point_ids = city_data["point_ids"]
    label_frame = city_data["label_frame"]
    n_points = len(point_ids)
    logging.info("City points: %d", n_points)

    # Coordinates for spatial fold assignment
    lat = label_frame.loc[point_ids, "lat"].to_numpy(dtype=np.float64)
    lon = label_frame.loc[point_ids, "lon"].to_numpy(dtype=np.float64)
    fold_ids = spatial_kfold(lat, lon, n_folds=args.n_folds, seed=SEED)
    coords_m = project_to_meters(lat, lon)
    logging.info("Spatial folds: %s", {int(f): int((fold_ids == f).sum()) for f in np.unique(fold_ids)})
    logging.info("Block buffer: %.0f m (0 disables; ≥400m closes pedgraph kNN leakage)", args.block_buffer_m)

    if args.target is not None:
        if args.target not in TARGET_SPECS:
            logging.error(
                "Target '%s' not in TARGET_SPECS. Valid targets: %s",
                args.target, list(TARGET_SPECS.keys()),
            )
            return 1
        active_targets = [args.target]
    else:
        active_targets = list(TARGET_SPECS.keys())

    # Extract all target arrays aligned to city point_ids
    labels_all = extract_multitarget_labels(label_frame, point_ids, active_targets)

    # For Ridge we need a single concatenated feature matrix
    # Build it using dummy split (all train, no val/test) to get the scaler then re-use transforms
    # We build it once with a 90/10 dummy split just to get normalization transforms, then
    # apply per-fold. For Ridge this is acceptable since the features are re-scaled per-fold.
    # The cleaner approach: normalize per fold inside the loop.

    per_fold_results: list[dict] = []
    rng = np.random.default_rng(SEED)

    buffer_drop_log: list[dict] = []
    for fold_k in tqdm(range(args.n_folds), desc="spatial folds"):
        test_idx = np.where(fold_ids == fold_k)[0]
        pool_idx = np.where(fold_ids != fold_k)[0]

        # 20% of pool → val, 80% → train (seeded per fold for reproducibility)
        fold_rng = np.random.default_rng(SEED + fold_k)
        perm = fold_rng.permutation(len(pool_idx))
        n_val = max(1, int(round(len(pool_idx) * 0.20)))
        val_local = perm[:n_val]
        train_local = perm[n_val:]
        val_idx = pool_idx[val_local]
        train_idx = pool_idx[train_local]

        # Apply spatial buffer: drop train (and val) points within block_buffer_m of any
        # test point. Test fold is preserved verbatim. Val is filtered too because val
        # is used for early stopping on the MLP path; if val is contaminated with
        # test-adjacent points, early-stopping picks a model that overfits to spatial
        # autocorrelation.
        n_train_pre = len(train_idx)
        n_val_pre = len(val_idx)
        train_idx, n_train_dropped = buffer_filter_train_indices(
            train_idx, test_idx, coords_m, args.block_buffer_m,
        )
        val_idx, n_val_dropped = buffer_filter_train_indices(
            val_idx, test_idx, coords_m, args.block_buffer_m,
        )
        buffer_drop_log.append({
            "fold_id": fold_k,
            "buffer_m": args.block_buffer_m,
            "n_train_before": n_train_pre,
            "n_train_after": int(len(train_idx)),
            "n_train_dropped": n_train_dropped,
            "n_val_before": n_val_pre,
            "n_val_after": int(len(val_idx)),
            "n_val_dropped": n_val_dropped,
        })

        logging.info(
            "Fold %d: n_train=%d (-%d buffer) n_val=%d (-%d buffer) n_test=%d",
            fold_k, len(train_idx), n_train_dropped,
            len(val_idx), n_val_dropped, len(test_idx),
        )

        if args.model_family == "ridge":
            # Build normalized feature matrix for this fold using train stats
            train_split = {
                "groups": {k: v[train_idx] for k, v in city_data["groups"].items()},
                "targets": city_data["targets"],
                "point_ids": point_ids[train_idx],
                "label_frame": label_frame,
            }
            val_split = {
                "groups": {k: v[val_idx] for k, v in city_data["groups"].items()},
                "targets": city_data["targets"],
                "point_ids": point_ids[val_idx],
                "label_frame": label_frame,
            }
            test_split = {
                "groups": {k: v[test_idx] for k, v in city_data["groups"].items()},
                "targets": city_data["targets"],
                "point_ids": point_ids[test_idx],
                "label_frame": label_frame,
            }
            X_train_f, _X_val_f, X_test_f, _transforms = build_feature_matrices(
                train_split, val_split, test_split,
                feature_groups=feature_groups,
                model_family="ridge",
                pca_dim=args.pca_dim,
            )

            fold_result = _run_fold_ridge(
                fold_id=fold_k,
                X=np.concatenate([X_train_f, X_test_f], axis=0),  # not used; pass pre-split
                train_idx=np.arange(len(X_train_f)),
                val_idx=np.arange(len(X_train_f)),  # not used for ridge
                test_idx=np.arange(len(X_train_f), len(X_train_f) + len(X_test_f)),
                labels_all={n: np.concatenate([labels_all[n][train_idx], labels_all[n][test_idx]]) for n in active_targets},
                active_targets=active_targets,
                ridge_alpha=args.ridge_alpha,
                pca_dim=args.pca_dim,
                feature_groups=feature_groups,
            )
            # Override n counts with actual fold sizes
            fold_result["n_train"] = len(train_idx)
            fold_result["n_val"] = len(val_idx)
            fold_result["n_test"] = len(test_idx)

        else:
            fold_result = _run_fold_mlp(
                fold_id=fold_k,
                city_data=city_data,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                labels_all=labels_all,
                active_targets=active_targets,
                pca_dim=args.pca_dim,
                feature_groups=feature_groups,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                patience=args.patience,
                device=device,
            )

        per_fold_results.append(fold_result)
        for name in active_targets:
            m = fold_result["metrics"].get(name, {})
            if "error" not in m:
                if m.get("type") == "regression":
                    logging.info("  [fold %d] %-35s R²=%.4f  ρ=%.4f", fold_k, name, m.get("r2", float("nan")), m.get("spearman_rho", float("nan")))
                else:
                    logging.info("  [fold %d] %-35s AUROC=%.4f", fold_k, name, m.get("auroc", float("nan")))

    summary = _aggregate_fold_results(per_fold_results, active_targets)
    print_spatial_cv_table(args.city, args.n_folds, args.model_family, summary)

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "backbone": args.backbone,
        "n_folds": args.n_folds,
        "model_family": args.model_family,
        "ablation": args.ablation,
        "pca_dim": args.pca_dim,
        "active_targets": active_targets,
        "target_filter": args.target,
        "ridge_alpha": args.ridge_alpha,
        "block_buffer_m": args.block_buffer_m,
        "block_buffer_drops": buffer_drop_log,
        "per_fold_results": per_fold_results,
        "summary": summary,
    }

    results_path = REPORT_DIR / "spatial_cv_results.jsonl"
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_record) + "\n")
    logging.info("Appended run to %s", results_path)

    summary_path = REPORT_DIR / "spatial_cv_summary.json"
    summary_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    logging.info("Summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
