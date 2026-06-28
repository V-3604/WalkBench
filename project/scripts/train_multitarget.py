"""
Multi-target training for WalkCLIP v2 (Section 4.3, Phase 2).

Trains WalkCLIPMultiTarget on the full tier taxonomy simultaneously:
  Tier 1 (visually grounded): overture_sidewalk_present, overture_crosswalk_present,
    overture_intersection_count_200m, overture_building_footprint_frac_100m
  Tier 2 (partially visual): D3B, D1B, D3APO, stops_400m, trips_per_hr_400m
  Tier 3 (administrative): NatWalkInd, LPA_CrudePrev, OBESITY_CrudePrev

Architecture: WalkCLIPMultiTarget — shared modality-specific encoder (same projections as
WalkCLIPMultiModal) with per-target nn.Linear(hidden_dim, 1) heads in an nn.ModuleDict.
Binary targets use BCEWithLogitsLoss; regression targets use MSE on z-scored values.
Losses weighted by tier importance and summed. Early stopping on average validation loss.

Per-target evaluation: binary → AUROC on raw logits; avg_precision, F1, accuracy on isotonic-
calibrated logits (val→test); regression → R², R²_recal (z-score recalibration to test-city
distribution), Spearman ρ, RMSE.

Outputs:
  project/artifacts/reports/multitarget_results.jsonl  (one JSON line per run, appended)
  project/artifacts/reports/multitarget_summary.json   (latest full per-target summary)

Run:
    python project/scripts/train_multitarget.py --train-city msp --test-city seattle --target-tier 1
    python project/scripts/train_multitarget.py --train-city msp --test-city seattle --target-tier all --pca-dim 128
    python project/scripts/train_multitarget.py --train-city seattle --test-city msp --target-tier all --pca-dim 128
    python project/scripts/train_multitarget.py --in-city-city seattle --target-tier all --pca-dim 128
    python project/scripts/train_multitarget.py --train-city msp --test-city seattle --target-tier 1 --ablation vision_only --pca-dim 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    MODEL_DIR,
    SEED,
    ArrayScaler,
    set_seeds,
    load_embeddings,
    align_embedding_block,
    fit_array_scaler,
    transform_with_scaler,
    inverse_primary,
    tabular_fill_and_scale,
    compute_sample_weights,
    stratified_split_indices,
    prepare_city_data,
    build_feature_matrices,
    evaluate_primary_metrics_recalibrated,
    feature_groups_for_run,
    SAM2_INPUT_COLS,
)

TARGET_SPECS: dict[str, dict] = {
    # Tier 1 — visually grounded (directly visible in street-view or aerial imagery)
    "overture_sidewalk_present":             {"tier": 1, "type": "binary",     "weight": 1.0},
    "overture_crosswalk_present":            {"tier": 1, "type": "binary",     "weight": 1.0},
    "osm_marked_crossing_present":           {"tier": 1, "type": "binary",     "weight": 1.0},
    "overture_intersection_count_200m":      {"tier": 1, "type": "regression", "weight": 0.5},
    "overture_building_footprint_frac_100m": {"tier": 1, "type": "regression", "weight": 0.5},
    "walkability_index_v1":                  {"tier": 1, "type": "regression", "weight": 1.0},
    "walkability_index_v2":                  {"tier": 1, "type": "regression", "weight": 1.0},
    # Tier 2 — partially visual (visible from aerial; weak from street-view)
    "D3B":               {"tier": 2, "type": "regression", "weight": 0.5},
    "D1B":               {"tier": 2, "type": "regression", "weight": 0.3},
    "D3APO":             {"tier": 2, "type": "regression", "weight": 0.3},
    "stops_400m":        {"tier": 2, "type": "regression", "weight": 0.5},
    "trips_per_hr_400m": {"tier": 2, "type": "regression", "weight": 0.3},
    # Tier 3 — administrative / composite (expected to fail cross-city)
    "NatWalkInd":        {"tier": 3, "type": "regression", "weight": 0.5},
    "LPA_CrudePrev":     {"tier": 3, "type": "regression", "weight": 0.2},
    "OBESITY_CrudePrev": {"tier": 3, "type": "regression", "weight": 0.2},
}


@dataclass
class MultiTargetConfig:
    train_city: str
    test_city: str
    eval_mode: str          # "cross_city" | "in_city"
    ablation: str
    target_tier: str        # "1" | "2" | "3" | "all"
    pca_dim: int            # 0 = no PCA
    hidden_dim: int
    lr: float
    weight_decay: float
    batch_size: int
    n_epochs: int
    patience: int
    train_split: float
    val_split: float
    seed: int
    backbone: str = "siglip"
    # Per-modality backbone overrides (empty string = use backbone for both)
    backbone_street: str = ""   # e.g. "siglip2" for siglip2-street + remoteclip-naip fusion
    backbone_naip: str = ""     # e.g. "remoteclip"
    # Multi-city training: if non-empty, train on all listed cities and test on test_city
    train_cities: tuple[str, ...] = ()  # e.g. ("msp", "seattle", "dc") → Pittsburgh zero-shot


class WalkCLIPMultiTarget(nn.Module):
    """Shared modality-specific encoder with per-target linear heads.

    Mirrors WalkCLIPMultiModal's projection structure but replaces the single
    primary head with an nn.ModuleDict of Linear(hidden_dim, 1) heads.
    """

    def __init__(
        self,
        street_dim: int,
        naip_dim: int,
        tabular_dim: int,
        target_names: list[str],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self._street_dim = street_dim
        self._naip_dim = naip_dim
        self.has_street = street_dim > 0
        self.has_naip = naip_dim > 0
        self.has_tabular = tabular_dim > 0

        modality_dims: list[int] = []
        if self.has_street:
            self.street_proj = nn.Sequential(
                nn.LayerNorm(street_dim),
                nn.Linear(street_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            modality_dims.append(hidden_dim)

        if self.has_naip:
            self.naip_proj = nn.Sequential(
                nn.LayerNorm(naip_dim),
                nn.Linear(naip_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            modality_dims.append(hidden_dim)

        if self.has_tabular:
            self.tabular_proj = nn.Sequential(
                nn.LayerNorm(tabular_dim),
                nn.Linear(tabular_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            modality_dims.append(hidden_dim)

        fusion_in = sum(modality_dims) if modality_dims else hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in target_names}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        parts: list[torch.Tensor] = []
        offset = 0
        if self.has_street:
            s = x[:, offset : offset + self._street_dim]
            parts.append(self.street_proj(s))
            offset += self._street_dim
        if self.has_naip:
            n = x[:, offset : offset + self._naip_dim]
            parts.append(self.naip_proj(n))
            offset += self._naip_dim
        if self.has_tabular and offset < x.shape[1]:
            t = x[:, offset:]
            parts.append(self.tabular_proj(t))

        fused = self.fusion(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])
        return {name: head(fused).squeeze(-1) for name, head in self.heads.items()}


def select_active_targets(tier_arg: str) -> list[str]:
    if tier_arg == "all":
        return list(TARGET_SPECS.keys())
    tier_int = int(tier_arg)
    return [name for name, spec in TARGET_SPECS.items() if spec["tier"] == tier_int]


def extract_multitarget_labels(
    label_frame: pd.DataFrame,
    point_ids: np.ndarray,
    active_targets: list[str],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in active_targets:
        if name not in label_frame.columns:
            raise ValueError(
                f"Target column '{name}' not found in features_labels_agreement.csv. "
                f"Run the Overture join (step 3.4) before training."
            )
        result[name] = label_frame.loc[point_ids, name].to_numpy(dtype=np.float32)
    return result


def fit_target_scalers(
    labels_train: dict[str, np.ndarray],
    active_targets: list[str],
) -> dict[str, ArrayScaler | None]:
    scalers: dict[str, ArrayScaler | None] = {}
    for name in active_targets:
        if TARGET_SPECS[name]["type"] == "binary":
            scalers[name] = None
        else:
            vals = labels_train[name]
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                scalers[name] = ArrayScaler(
                    mean=np.array([0.0], dtype=np.float32),
                    scale=np.array([1.0], dtype=np.float32),
                )
            else:
                mean = float(valid.mean())
                std = float(valid.std())
                scalers[name] = ArrayScaler(
                    mean=np.array([mean], dtype=np.float32),
                    scale=np.array([max(std, 1e-6)], dtype=np.float32),
                )
    return scalers


def compute_multitarget_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    target_scalers: dict[str, ArrayScaler | None],
    active_targets: list[str],
    device: torch.device,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=device)
    n_active = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for name in active_targets:
        logits = predictions[name]
        y = targets[name].to(device)
        valid_mask = torch.isfinite(y)
        if valid_mask.sum() == 0:
            continue

        weight = float(TARGET_SPECS[name]["weight"])
        spec_type = TARGET_SPECS[name]["type"]

        if spec_type == "binary":
            loss_vals = bce(logits[valid_mask], y[valid_mask])
            loss = loss_vals.mean()
        else:
            scaler = target_scalers[name]
            mean_t = torch.tensor(scaler.mean[0], device=device, dtype=torch.float32)
            scale_t = torch.tensor(scaler.scale[0], device=device, dtype=torch.float32)
            y_z = (y[valid_mask] - mean_t) / scale_t
            loss = nn.functional.mse_loss(logits[valid_mask], y_z)

        total = total + weight * loss
        n_active += 1

    if n_active == 0:
        return total
    return total / n_active


def evaluate_target_metrics(
    name: str,
    logits_np: np.ndarray,
    y_true_np: np.ndarray,
    scaler: ArrayScaler | None,
    train_mean: float,
    train_std: float,
    logits_calibrated_np: np.ndarray | None = None,
) -> dict[str, float]:
    valid = np.isfinite(y_true_np) & np.isfinite(logits_np)
    logits_v = logits_np[valid]
    y_v = y_true_np[valid]
    n = int(valid.sum())

    if n < 2:
        return {"type": TARGET_SPECS[name]["type"], "n": n, "error": "insufficient_data"}

    spec_type = TARGET_SPECS[name]["type"]
    if spec_type == "binary":
        probs_raw = 1.0 / (1.0 + np.exp(-logits_v))
        try:
            auroc = float(roc_auc_score(y_v, probs_raw))
        except Exception:
            auroc = float("nan")
        if logits_calibrated_np is not None:
            logits_cal_v = logits_calibrated_np[valid]
            probs = 1.0 / (1.0 + np.exp(-logits_cal_v))
        else:
            probs = probs_raw
        preds_bin = (probs >= 0.5).astype(np.float32)
        try:
            ap = float(average_precision_score(y_v, probs))
        except Exception:
            ap = float("nan")
        f1 = float(f1_score(y_v, preds_bin, zero_division=0))
        acc = float(accuracy_score(y_v, preds_bin))
        pos_rate = float(y_v.mean())
        return {
            "type": "binary",
            "n": n,
            "auroc": round(auroc, 4),
            "avg_precision": round(ap, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "positive_rate": round(pos_rate, 4),
        }
    else:
        pred = logits_v * scaler.scale[0] + scaler.mean[0]
        pred_z = (pred - train_mean) / max(train_std, 1e-8)
        test_mean = float(np.mean(y_v))
        test_std = float(np.std(y_v))
        pred_recal = pred_z * test_std + test_mean
        y_r2 = y_v
        p_r2 = pred
        pr_recal = pred_recal
        if name == "overture_intersection_count_200m":
            y_r2 = np.expm1(y_v)
            p_r2 = np.expm1(pred)
            pr_recal = np.expm1(pred_recal)
        r2 = float(r2_score(y_r2, p_r2))
        rmse = float(np.sqrt(np.mean((p_r2 - y_r2) ** 2)))
        mae = float(np.mean(np.abs(p_r2 - y_r2)))
        spearman_rho, _ = stats.spearmanr(p_r2, y_r2)

        r2_recal = float(r2_score(y_r2, pr_recal))

        return {
            "type": "regression",
            "n": n,
            "r2": round(r2, 4),
            "r2_recalibrated": round(r2_recal, 4),
            "spearman_rho": round(float(spearman_rho), 4) if np.isfinite(spearman_rho) else float("nan"),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "train_mean": round(train_mean, 4),
            "train_std": round(train_std, 4),
        }


def train_multitarget_model(
    cfg: MultiTargetConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    target_scalers: dict[str, ArrayScaler | None],
    active_targets: list[str],
    group_dims: dict[str, int],
    device: torch.device,
) -> tuple[dict[str, dict], dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    street_dim = group_dims.get("street", 0)
    naip_dim = group_dims.get("naip", 0)
    tabular_dim = sum(v for k, v in group_dims.items() if k not in ("street", "naip"))

    model = WalkCLIPMultiTarget(
        street_dim=street_dim,
        naip_dim=naip_dim,
        tabular_dim=tabular_dim,
        target_names=active_targets,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    logging.info(
        "  WalkCLIPMultiTarget: street=%d naip=%d tabular=%d heads=%d",
        street_dim, naip_dim, tabular_dim, len(active_targets),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr))

    # Pack labels into a 2D array for the DataLoader; unpack by index in loop
    train_label_matrix = np.stack([train_labels[n] for n in active_targets], axis=1)
    ds_train = TensorDataset(to_tensor(X_train), to_tensor(train_label_matrix))
    loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    X_val_t = to_tensor(X_val).to(device)
    val_label_tensors = {n: to_tensor(val_labels[n]) for n in active_targets}

    best_val = float("inf")
    best_epoch = 0
    best_state: dict = {}
    patience_left = cfg.patience
    t0 = time.time()

    for epoch in tqdm(range(1, cfg.n_epochs + 1), desc="training", unit="epoch"):
        model.train()
        for Xb, Yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            batch_targets = {
                active_targets[i]: Yb[:, i].to(device, non_blocking=True)
                for i in range(len(active_targets))
            }
            preds = model(Xb)
            loss = compute_multitarget_loss(preds, batch_targets, target_scalers, active_targets, device)
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
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logging.info("  Early stop at epoch %d (best=%d, val_loss=%.6f)", epoch, best_epoch, best_val)
                break

    model.load_state_dict(best_state)
    model.eval()
    X_test_t = to_tensor(X_test).to(device)
    with torch.no_grad():
        test_preds = model(X_test_t)
        val_preds = model(X_val_t)

    elapsed = round(time.time() - t0, 1)
    logging.info("  Training complete: best_epoch=%d elapsed=%.1fs", best_epoch, elapsed)

    test_logits = {name: test_preds[name].cpu().numpy() for name in active_targets}
    test_logits_uncal = {name: arr.copy() for name, arr in test_logits.items()}

    # Isotonic calibration per binary target (AUROC unchanged; AP/F1 cleaner).
    test_logits_calibrated = {}
    val_logits_dict = {n: val_preds[n].cpu().numpy() for n in active_targets}
    for name in active_targets:
        spec = TARGET_SPECS.get(name, {})
        if spec.get("type") != "binary":
            test_logits_calibrated[name] = test_logits[name]
            continue
        y_val = val_labels[name]
        finite = np.isfinite(y_val) & np.isfinite(val_logits_dict[name])
        if finite.sum() < 50 or len(np.unique(y_val[finite].astype(int))) < 2:
            test_logits_calibrated[name] = test_logits[name]
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_logits_dict[name][finite], y_val[finite].astype(int))
        test_logits_calibrated[name] = iso.transform(test_logits[name])
    test_logits = test_logits_calibrated

    results_per_target: dict[str, dict] = {}
    for name in active_targets:
        logits_np = test_logits_uncal[name]
        y_true_np = test_labels[name]
        scaler = target_scalers[name]
        train_mean = float(scaler.mean[0]) if scaler is not None else 0.0
        train_std = float(scaler.scale[0]) if scaler is not None else 1.0
        logits_cal = (
            test_logits[name] if TARGET_SPECS[name]["type"] == "binary" else None
        )
        metrics = evaluate_target_metrics(
            name,
            logits_np,
            y_true_np,
            scaler,
            train_mean,
            train_std,
            logits_calibrated_np=logits_cal,
        )
        metrics["tier"] = TARGET_SPECS[name]["tier"]
        results_per_target[name] = metrics

    checkpoint = {
        "model_state_dict": best_state,
        "config": asdict(cfg),
        "active_targets": active_targets,
        "target_scalers": {
            name: asdict(scaler) if scaler is not None else None
            for name, scaler in target_scalers.items()
        },
        "group_dims": group_dims,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_s": elapsed,
    }
    return results_per_target, checkpoint, test_logits, test_logits_uncal


def _make_split_data(full_data: dict, idx: np.ndarray) -> dict:
    return {
        "groups": {k: v[idx] for k, v in full_data["groups"].items()},
        "targets": {k: v[idx] for k, v in full_data["targets"].items()},
        "point_ids": full_data["point_ids"][idx],
        "label_frame": full_data["label_frame"],
    }


def _concat_city_data(data_list: list[dict]) -> dict:
    """Concatenate per-city data dicts for multi-city training.

    Assigns a new sequential index (0..N-1) to the combined label_frame so
    that extract_multitarget_labels can look up by position rather than
    per-city point_id (which has duplicates across cities).
    """
    if len(data_list) == 1:
        return data_list[0]
    offset = 0
    frames: list[pd.DataFrame] = []
    for d in data_list:
        n = len(d["point_ids"])
        frame = d["label_frame"].copy()
        frame.index = range(offset, offset + n)
        frames.append(frame)
        offset += n
    combined_label_frame = pd.concat(frames)
    combined_point_ids = np.arange(offset, dtype=np.int32)  # sequential index = "point_id"
    combined_groups: dict[str, np.ndarray] = {}
    for key in data_list[0]["groups"]:
        combined_groups[key] = np.concatenate([d["groups"][key] for d in data_list], axis=0)
    combined_targets: dict[str, np.ndarray] = {}
    for key in data_list[0]["targets"]:
        combined_targets[key] = np.concatenate([d["targets"][key] for d in data_list], axis=0)
    return {
        "point_ids": combined_point_ids,
        "label_frame": combined_label_frame,
        "groups": combined_groups,
        "targets": combined_targets,
    }


def run_one(
    cfg: MultiTargetConfig,
    labels: pd.DataFrame,
    captions: pd.DataFrame,
    spatial: pd.DataFrame,
    lock_ids: list[int],
    device: torch.device,
) -> dict[str, dict]:
    set_seeds(cfg.seed)
    caption_cols = [c for c in captions.columns if c.startswith("cap_")]
    spatial_cols = [c for c in spatial.columns if c.startswith("spatial_")]
    feature_groups = feature_groups_for_run(cfg.ablation)
    active_targets = select_active_targets(cfg.target_tier)

    if "captions" in feature_groups and not caption_cols:
        raise ValueError("Caption features requested but no `cap_` columns found in caption CSV.")
    if "spatial" in feature_groups and not spatial_cols:
        raise ValueError("Spatial features requested but no `spatial_` columns found in spatial CSV.")

    bs_kwargs = dict(
        backbone=cfg.backbone,
        backbone_street=cfg.backbone_street or None,
        backbone_naip=cfg.backbone_naip or None,
    )

    if cfg.train_cities:
        # Multi-city training: concatenate data from all training cities.
        # Each city uses its own lock file; lock_ids arg is ignored here.
        _city_lock_files = {
            "msp":         REPO_ROOT / "project/data/processed/locks/msp_v2_final_lock_ids.txt",
            "seattle":     REPO_ROOT / "project/data/processed/locks/seattle_v2_final_lock_ids.txt",
            "dc":          REPO_ROOT / "project/data/processed/locks/dc_v2_final_lock_ids.txt",
            "pittsburgh":  REPO_ROOT / "project/data/processed/locks/pittsburgh_v2_final_lock_ids.txt",
        }
        data_list = []
        for city in cfg.train_cities:
            city_lock = _city_lock_files[city]
            city_lock_ids = [int(x) for x in city_lock.read_text().strip().splitlines()]
            data_list.append(prepare_city_data(
                city, labels, captions, spatial, city_lock_ids, caption_cols, spatial_cols,
                **bs_kwargs,
            ))
        train_full = _concat_city_data(data_list)
    else:
        train_full = prepare_city_data(
            cfg.train_city, labels, captions, spatial, lock_ids, caption_cols, spatial_cols,
            **bs_kwargs,
        )

    # Stratified split on NatWalkInd for consistent bin coverage
    primary_vals = train_full["label_frame"]["NatWalkInd"].to_numpy(dtype=np.float32)
    if cfg.eval_mode == "cross_city":
        train_idx, val_idx, _ = stratified_split_indices(
            primary_vals, train_frac=cfg.train_split, val_frac=cfg.val_split, seed=cfg.seed
        )
        test_full = prepare_city_data(
            cfg.test_city, labels, captions, spatial, lock_ids, caption_cols, spatial_cols,
            **bs_kwargs,
        )
        test_idx = np.arange(len(test_full["point_ids"]), dtype=int)
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(
            primary_vals, train_frac=cfg.train_split, val_frac=cfg.val_split, seed=cfg.seed
        )
        test_full = train_full

    train_split = _make_split_data(train_full, train_idx)
    val_split = _make_split_data(train_full, val_idx)
    test_split = _make_split_data(test_full, test_idx)

    X_train, X_val, X_test, feature_transforms = build_feature_matrices(
        train_split, val_split, test_split,
        feature_groups=feature_groups,
        model_family="mlp",
        pca_dim=cfg.pca_dim,
    )
    group_dims: dict[str, int] = feature_transforms["group_dims"]

    train_labels = extract_multitarget_labels(
        train_full["label_frame"], train_full["point_ids"][train_idx], active_targets
    )
    val_labels = extract_multitarget_labels(
        train_full["label_frame"], train_full["point_ids"][val_idx], active_targets
    )
    test_labels = extract_multitarget_labels(
        test_full["label_frame"], test_full["point_ids"][test_idx], active_targets
    )

    target_scalers = fit_target_scalers(train_labels, active_targets)

    results_per_target, checkpoint, test_logits, test_logits_uncal = train_multitarget_model(
        cfg=cfg,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        target_scalers=target_scalers,
        active_targets=active_targets,
        group_dims=group_dims,
        device=device,
    )

    model_stem = (
        f"multitarget_{cfg.eval_mode}_{cfg.train_city}_to_{cfg.test_city}"
        f"_{cfg.ablation}_tier{cfg.target_tier}"
    )
    model_path = MODEL_DIR / f"{model_stem}.pt"
    torch.save(checkpoint, model_path)
    logging.info("  Checkpoint saved: %s", model_path)

    return results_per_target, {
        "test_logits": test_logits,
        "test_logits_raw": test_logits_uncal,
        "test_labels": {name: test_labels[name] for name in test_logits},
        "test_point_ids": test_full["point_ids"][test_idx].astype(np.int64),
        "active_targets": list(test_logits.keys()),
    }


def print_results_table(
    train_city: str,
    test_city: str,
    eval_mode: str,
    results_per_target: dict[str, dict],
) -> None:
    logging.info("")
    logging.info("─" * 100)
    logging.info("MULTI-TARGET RESULTS  [%s  %s → %s]", eval_mode, train_city, test_city)
    logging.info("─" * 100)

    for tier in (1, 2, 3):
        tier_names = [n for n in results_per_target if TARGET_SPECS[n]["tier"] == tier]
        if not tier_names:
            continue
        logging.info("Tier %d:", tier)
        for name in tier_names:
            m = results_per_target[name]
            if m.get("type") == "binary":
                logging.info(
                    "  %-40s  AUROC=%s  AP=%s  F1=%s  acc=%s  pos_rate=%s  n=%s",
                    name,
                    f"{m.get('auroc', float('nan')):.4f}",
                    f"{m.get('avg_precision', float('nan')):.4f}",
                    f"{m.get('f1', float('nan')):.4f}",
                    f"{m.get('accuracy', float('nan')):.4f}",
                    f"{m.get('positive_rate', float('nan')):.4f}",
                    m.get("n", 0),
                )
            elif m.get("type") == "regression":
                logging.info(
                    "  %-40s  R²=%s  R²_recal=%s  ρ=%s  RMSE=%s  n=%s",
                    name,
                    f"{m.get('r2', float('nan')):.4f}",
                    f"{m.get('r2_recalibrated', float('nan')):.4f}",
                    f"{m.get('spearman_rho', float('nan')):.4f}",
                    f"{m.get('rmse', float('nan')):.4f}",
                    m.get("n", 0),
                )
            else:
                logging.info("  %-40s  %s", name, m)
    logging.info("─" * 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WalkCLIP v2 multi-target trainer (Section 4.3)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-city", choices=["msp", "seattle", "dc", "pittsburgh"])
    mode.add_argument("--in-city-city", choices=["msp", "seattle", "dc", "pittsburgh"])
    mode.add_argument("--train-cities", nargs="+",
                      choices=["msp", "seattle", "dc", "pittsburgh"],
                      help="Multi-city training. Concatenates all listed cities for training. "
                           "Use with --test-city for Pittsburgh zero-shot: "
                           "--train-cities msp seattle dc --test-city pittsburgh")
    p.add_argument("--test-city", choices=["msp", "seattle", "dc", "pittsburgh"])
    p.add_argument("--heading-supervision", choices=["on", "off"], default="off",
                   help="Use per-(point_id,heading) rows from heading_overture_labels.csv (default: off)")
    p.add_argument("--spatial-features", choices=["euclidean", "pedgraph", "none"], default="euclidean",
                   help="Spatial features source: euclidean=spatial_context_features.csv, "
                        "pedgraph=pedgraph_features.csv (default: euclidean)")
    p.add_argument("--include-terrain", action="store_true",
                   help="Left-join USGS 3DEP terrain features (spatial_terrain_*) into the "
                        "spatial frame. Requires terrain_features.csv built by "
                        "build_terrain_features.py.")
    p.add_argument("--target-tier", choices=["1", "2", "3", "all"], default="all")
    p.add_argument("--ablation", choices=sorted(FEATURES_BY_ABLATION), default="full")
    def _backbone_type(s: str) -> str:
        # Empty string means "use the shared --backbone value" — pass through without validation.
        # argparse calls type= on string defaults (PEP 3107 / Python ≥3.9 behaviour), so we
        # must accept "" here even though it is never a meaningful backbone name.
        if not s:
            return s
        allowed = {
            # Frozen baselines
            "siglip", "clip", "siglip2", "dinov2",
            # LoRA in-corpus (optimistic, test city seen at fine-tune time)
            "siglip_lora",
            # LoRA LOCO v1 (hold-out named city)
            "siglip_lora_loco_msp", "siglip_lora_loco_seattle", "siglip_lora_loco_dc",
            # LoRA LOCO v2 (hold-out named city)
            "siglip_lora_loco_v2_msp", "siglip_lora_loco_v2_seattle", "siglip_lora_loco_v2_dc",
            # Pittsburgh-holdout LoRA LOCO (train on MSP+Seattle+DC)
            "siglip_lora_loco_v2_pittsburgh",
            # RemoteCLIP (NAIP only — use with --backbone-naip remoteclip)
            "remoteclip",
        }
        if s not in allowed:
            raise argparse.ArgumentTypeError(
                f"invalid backbone {s!r}; choose from {sorted(allowed)}")
        return s
    p.add_argument("--backbone", type=_backbone_type, default="siglip",
                   help="Shared backbone tag — used for both street and NAIP unless overridden.")
    p.add_argument("--backbone-street", type=_backbone_type, default="",
                   help="Override backbone for street modality only. "
                        "Used for siglip2-street + remoteclip-naip fusion: "
                        "--backbone-street siglip2 --backbone-naip remoteclip")
    p.add_argument("--backbone-naip", type=_backbone_type, default="",
                   help="Override backbone for NAIP modality only.")
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-split", type=float, default=0.85)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for downstream MLP (default: 42). Used for "
                        "stratified split, weight init, dropout, batch order. "
                        "Run multiple seeds and aggregate to estimate downstream variance.")
    p.add_argument("--save-predictions", action="store_true",
                   help="Persist (test_logits, test_y, point_ids) per target to "
                        "project/artifacts/predictions/<run_id>.npz for offline "
                        "bootstrap CI / paired DeLong tests.")
    p.add_argument("--experiment-tag", default=None,
                   help="Freeform tag written into the result record in "
                        "multitarget_results.jsonl. Use src_pgh, src_backbones, "
                        "src_terrain, src_mmd, src_labelceil, src_acs to "
                        "namespace SRC extension runs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for d in (MODEL_DIR, LOG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for path, label in (
        (LOCK_FILE, "Lock file"),
        (LABELS_CSV, "Labels CSV"),
        (CAPTION_CSV, "Caption CSV"),
        (SPATIAL_CSV, "Spatial CSV"),
    ):
        if not path.exists():
            logging.error(
                "%s not found: %s. Ensure all preprocessing steps have run.", label, path
            )
            return 1

    if not EMBED_DIR.exists() or not any(EMBED_DIR.glob("siglip_*.npy")):
        logging.error(
            "No embeddings found in %s. Run extract_siglip_embeddings.py --source all --city all first.",
            EMBED_DIR,
        )
        return 1

    # Per-city lock file lookup (DC has its own; msp/seattle use shared lock)
    _city_lock_files = {
        "msp":         REPO_ROOT / "project/data/processed/locks/msp_v2_final_lock_ids.txt",
        "seattle":     REPO_ROOT / "project/data/processed/locks/seattle_v2_final_lock_ids.txt",
        "dc":          REPO_ROOT / "project/data/processed/locks/dc_v2_final_lock_ids.txt",
        "pittsburgh":  REPO_ROOT / "project/data/processed/locks/pittsburgh_v2_final_lock_ids.txt",
    }
    _active_city = args.in_city_city or args.train_city
    _city_lock = _city_lock_files.get(_active_city, LOCK_FILE)
    _lock_path = _city_lock if _city_lock.exists() else LOCK_FILE
    lock_ids = [int(x) for x in _lock_path.read_text().strip().splitlines()]

    labels = pd.read_csv(LABELS_CSV)
    wi_path = LABELS_CSV.parent / "walkability_index.csv"
    if wi_path.exists():
        _wi_cols = ["point_id", "city", "walkability_index_v1"]
        _wi_all = pd.read_csv(wi_path)
        if "walkability_index_v2" in _wi_all.columns:
            _wi_cols.append("walkability_index_v2")
        wi = _wi_all[_wi_cols]
        labels = labels.merge(wi, on=["point_id", "city"], how="left")
    else:
        logging.warning(
            "walkability_index.csv not found; walkability_index_v1 target will be all-NaN."
        )

    # OSM marked-crossing label (revived crosswalk target, 2026-06-24). Lives in its own
    # file because Overture's crosswalk column was the dead ~0.56-AUROC label;
    # osm_marked_crossing_present is the cleaner ~0.79 cross-city target. Joined on
    # (point_id, city); points absent from the OSM pull get NaN and are masked in the loss.
    osm_path = LABELS_CSV.parent / "osm_crossing_labels.csv"
    if osm_path.exists():
        osm = pd.read_csv(osm_path)[["point_id", "city", "osm_marked_crossing_present"]]
        osm["point_id"] = osm["point_id"].astype(int)
        labels["point_id"] = labels["point_id"].astype(int)
        labels = labels.merge(osm, on=["point_id", "city"], how="left")
    else:
        logging.warning(
            "osm_crossing_labels.csv not found; osm_marked_crossing_present target will be "
            "all-NaN. Run fetch_osm_crossings.py first."
        )
    if "osm_marked_crossing_present" not in labels.columns:
        labels["osm_marked_crossing_present"] = np.nan

    captions = pd.read_csv(CAPTION_CSV)

    # Spatial features: euclidean or pedgraph
    if args.spatial_features == "pedgraph":
        _pedgraph_csv = REPO_ROOT / "project/data/processed/spatial_context/pedgraph_features.csv"
        if not _pedgraph_csv.exists():
            logging.error(
                "pedgraph_features.csv not found: %s\n"
                "Run build_pedgraph_spatial_features.py --cities msp seattle dc first.",
                _pedgraph_csv,
            )
            return 1
        spatial = pd.read_csv(_pedgraph_csv)
    elif args.spatial_features == "none":
        # Empty spatial frame — trainer will skip spatial features gracefully
        spatial = pd.DataFrame(columns=["point_id", "city"])
    else:
        spatial = pd.read_csv(SPATIAL_CSV)

    if args.include_terrain:
        _terrain_csv = REPO_ROOT / "project/data/processed/spatial_context/terrain_features.csv"
        if not _terrain_csv.exists():
            logging.error(
                "terrain_features.csv not found: %s\n"
                "Run build_terrain_features.py --cities msp seattle dc first.",
                _terrain_csv,
            )
            return 1
        terrain = pd.read_csv(_terrain_csv)
        terrain["point_id"] = terrain["point_id"].astype(int)
        terrain_cols = [c for c in terrain.columns if c.startswith("spatial_terrain_")]
        if not terrain_cols:
            logging.warning("terrain_features.csv has no spatial_terrain_* columns; skipping merge.")
        else:
            join_keys = ["point_id", "city"] if "city" in spatial.columns and "city" in terrain.columns else ["point_id"]
            spatial = spatial.merge(
                terrain[join_keys + terrain_cols], on=join_keys, how="left",
            )
            logging.info("Merged %d terrain columns onto spatial frame.", len(terrain_cols))

    for df in (labels, captions, spatial):
        df["point_id"] = df["point_id"].astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s  seed=%d", device, SEED)
    logging.info("Labels=%d  Captions=%d  Spatial=%d  Lock=%d", len(labels), len(captions), len(spatial), len(lock_ids))

    if args.in_city_city:
        eval_mode = "in_city"
        train_city = args.in_city_city
        test_city = args.in_city_city
        train_split = 0.70
        val_split = 0.15
        train_cities: tuple[str, ...] = ()
    elif args.train_cities:
        if not args.test_city:
            logging.error("--test-city is required when using --train-cities.")
            return 1
        eval_mode = "cross_city"
        train_city = "+".join(sorted(args.train_cities))   # label string, e.g. "dc+msp+seattle"
        test_city = args.test_city
        train_split = args.train_split
        val_split = args.val_split
        train_cities = tuple(args.train_cities)
    else:
        if not args.test_city:
            logging.error("--test-city is required when using --train-city.")
            return 1
        if args.train_city == args.test_city:
            logging.error("Use --in-city-city for same-city runs.")
            return 1
        eval_mode = "cross_city"
        train_city = args.train_city
        test_city = args.test_city
        train_split = args.train_split
        val_split = args.val_split
        train_cities = ()

    cfg = MultiTargetConfig(
        train_city=train_city,
        test_city=test_city,
        eval_mode=eval_mode,
        ablation=args.ablation,
        target_tier=args.target_tier,
        pca_dim=args.pca_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        patience=args.patience,
        train_split=train_split,
        val_split=val_split,
        seed=args.seed,
        backbone=args.backbone,
        backbone_street=args.backbone_street,
        backbone_naip=args.backbone_naip,
        train_cities=train_cities,
    )

    logging.info(
        "=== MultiTarget | %s | %s → %s | tier=%s | ablation=%s | pca=%d ===",
        cfg.eval_mode, cfg.train_city, cfg.test_city, cfg.target_tier, cfg.ablation, cfg.pca_dim,
    )

    results_per_target, pred_bundle = run_one(cfg, labels, captions, spatial, lock_ids, device)

    print_results_table(cfg.train_city, cfg.test_city, cfg.eval_mode, results_per_target)

    eff_street = cfg.backbone_street or cfg.backbone
    eff_naip = cfg.backbone_naip or cfg.backbone
    backbone_tag = eff_street if eff_street == eff_naip else f"{eff_street}+{eff_naip}"

    run_id = (
        f"{backbone_tag}__{cfg.eval_mode}__{cfg.train_city}_to_{cfg.test_city}"
        f"__tier{cfg.target_tier}__{cfg.ablation}__seed{cfg.seed}"
    )
    if args.save_predictions:
        pred_dir = REPO_ROOT / "project/artifacts/predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_path = pred_dir / f"{run_id}.npz"
        np.savez_compressed(
            pred_path,
            point_ids=pred_bundle["test_point_ids"],
            **{f"logit__{name}": pred_bundle["test_logits"][name]
               for name in pred_bundle["active_targets"]},
            **{f"y__{name}": pred_bundle["test_labels"][name].astype(np.float32)
               for name in pred_bundle["active_targets"]},
        )
        logging.info("Saved predictions: %s", pred_path)

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "train_city": cfg.train_city,
        "test_city": cfg.test_city,
        "eval_mode": cfg.eval_mode,
        "ablation": cfg.ablation,
        "target_tier": cfg.target_tier,
        "pca_dim": cfg.pca_dim,
        "hidden_dim": cfg.hidden_dim,
        "spatial_features": args.spatial_features,
        "include_terrain": bool(args.include_terrain),
        "backbone": backbone_tag,
        "backbone_street": eff_street,
        "backbone_naip": eff_naip,
        "heading_supervision": args.heading_supervision,
        "seed": cfg.seed,
        "experiment_tag": args.experiment_tag,
        "results_per_target": results_per_target,
    }

    results_path = REPORT_DIR / "multitarget_results.jsonl"
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_record) + "\n")
    logging.info("Appended run to %s", results_path)

    summary_path = REPORT_DIR / "multitarget_summary.json"
    summary_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    logging.info("Summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
