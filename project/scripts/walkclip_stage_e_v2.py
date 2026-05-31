"""
Enhanced Stage E trainer for WalkCLIP v2.

Upgrades over the original trainer:
    - captions are used as actual model inputs
    - spatial context features are supported
    - train-fold-only feature/target scaling
    - target normalization for NatWalkInd and auxiliary heads
    - missing NAIP tiles are mean-imputed with an explicit missing indicator
    - weighted validation loss matches the weighted training objective
    - optional in-city sanity mode
    - linear / ridge / elastic-net baselines
    - auxiliary heads for GTFS, PLACES, and EPA component targets

Recommended prep:
    python project/scripts/compute_audit_agreement.py
    python project/scripts/build_caption_features.py
    python project/scripts/build_spatial_context_features.py
    python project/scripts/extract_siglip_embeddings.py --source all --city all

Examples:
    python project/scripts/train_walkclip.py --train-city msp --test-city seattle
    python project/scripts/train_walkclip.py --sweep
    python project/scripts/train_walkclip.py --train-city msp --test-city seattle --model-family ridge --ablation caption_spatial
    python project/scripts/train_walkclip.py --in-city-city msp --ablation full
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "project" / "artifacts" / "embeddings"
MODEL_DIR = REPO_ROOT / "project" / "artifacts" / "models" / "downstream_v2"
LOG_DIR = REPO_ROOT / "project" / "artifacts" / "logs"
REPORT_DIR = REPO_ROOT / "project" / "artifacts" / "reports"

LOCK_FILE = REPO_ROOT / "project" / "data" / "processed" / "locks" / "v2_final_lock_ids.txt"
LABELS_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "features_labels_agreement.csv"
CAPTION_CSV = REPO_ROOT / "project" / "data" / "processed" / "text" / "caption_features_final_lock.csv"
SPATIAL_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "spatial_context_features.csv"

SEED = 42
EMBED_DIM = 1152

SAM2_INPUT_COLS = [
    "n_headings",
    "sidewalk_frac_mean",
    "crosswalk_frac_mean",
    "vegetation_frac_mean",
    "sky_frac_mean",
    "building_frac_mean",
    "sidewalk_frac_max",
    "crosswalk_frac_max",
    "vegetation_frac_max",
    "building_frac_max",
    "sidewalk_width_px",
    "street_mask_coverage_mean",
    "naip_vegetation_frac",
    "naip_building_frac",
    "naip_road_frac",
    "naip_water_frac",
    "naip_mask_coverage",
    "naip_n_masks",
    "naip_missing",
    "naip_unclassified",
]

SAM2_AUX_COLS = [
    "sidewalk_frac_mean",
    "crosswalk_frac_mean",
    "vegetation_frac_mean",
    "sky_frac_mean",
    "building_frac_mean",
    "sidewalk_frac_max",
    "crosswalk_frac_max",
    "vegetation_frac_max",
    "building_frac_max",
]

PUBLIC_AUX_COLS = [
    "D2A_Ranked",
    "D2B_Ranked",
    "D3B_Ranked",
    "D4A_Ranked",
    "D1A",
    "D1B",
    "D2B_E8MIX",
    "D3A",
    "D3B",
    "D3APO",
    "D4A",
    "D4C",
    "D4D",
    "LPA_CrudePrev",
    "OBESITY_CrudePrev",
    "stops_400m",
    "trips_per_hr_400m",
    "stops_800m",
    "trips_per_hr_800m",
]

PRIMARY_TARGET = "NatWalkInd"
SECONDARY_TARGETS = ["LPA_CrudePrev", "OBESITY_CrudePrev", "stops_400m", "trips_per_hr_400m"]
TIER_WEIGHTS = {"high_confidence": 1.0, "partial": 0.5, "disagreement": 0.25}

# SAM2 and captions dropped from all ablations (decisions O9+O21, O22, 2026-04-26).
# "full" = SigLIP street + SigLIP NAIP + tabular/spatial (pedgraph kNN).
FEATURES_BY_ABLATION: dict[str, tuple[str, ...]] = {
    "full":           ("street", "naip", "spatial"),
    "vision_only":    ("street", "naip"),
    "street_only":    ("street",),
    "naip_only":      ("naip",),
    "spatial_only":   ("spatial",),
    "tabular_only":   ("spatial",),
    "vision_tabular": ("street", "naip", "spatial"),
    "vision_spatial": ("street", "naip", "spatial"),
    # legacy keys kept so old scripts get a clear error rather than KeyError
    "no_spatial":     ("street", "naip"),
    "no_weights":     ("street", "naip", "spatial"),
    "no_sam2":        ("street", "naip", "spatial"),
    "no_captions":    ("street", "naip", "spatial"),
}

SWEEP_ABLATIONS = [
    "full",
    "vision_only",
    "street_only",
    "naip_only",
    "spatial_only",
    "tabular_only",
    "vision_tabular",
    "vision_spatial",
]


@dataclass
class RunConfig:
    train_city: str
    test_city: str
    eval_mode: str
    ablation: str
    model_family: str
    architecture: str   # "multitask" | "multimodal"
    pca_dim: int        # 0 = no PCA; 64/128/256 = compress street+naip embeddings
    lr: float
    weight_decay: float
    batch_size: int
    n_epochs: int
    patience: int
    train_split: float
    val_split: float
    seed: int
    sam_aux_weight: float
    public_aux_weight: float
    ridge_alpha: float
    elasticnet_alpha: float
    elasticnet_l1_ratio: float


@dataclass
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray


class WalkCLIPMultitask(nn.Module):
    def __init__(self, input_dim: int, n_sam_aux: int, n_public_aux: int, linear_backbone: bool) -> None:
        super().__init__()
        if linear_backbone:
            self.backbone = nn.Identity()
            hidden_dim = input_dim
        else:
            self.backbone = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            hidden_dim = 64

        self.primary = nn.Linear(hidden_dim, 1)
        self.sam_aux = nn.Linear(hidden_dim, n_sam_aux) if n_sam_aux > 0 else None
        self.public_aux = nn.Linear(hidden_dim, n_public_aux) if n_public_aux > 0 else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        h = self.backbone(x)
        primary = self.primary(h).squeeze(-1)
        sam = self.sam_aux(h) if self.sam_aux is not None else None
        public = self.public_aux(h) if self.public_aux is not None else None
        return primary, sam, public


class WalkCLIPMultiModal(nn.Module):
    """Modality-specific projection heads before fusion (Phase 2, section 4.2).

    Each modality (street, naip, tabular) is projected to hidden_dim independently.
    This prevents high-dim embeddings from drowning tabular features in the first
    linear layer gradient.
    """

    def __init__(
        self,
        street_dim: int,
        naip_dim: int,
        tabular_dim: int,
        hidden_dim: int = 128,
        n_sam_aux: int = 0,
        n_public_aux: int = 0,
    ) -> None:
        super().__init__()
        modality_dims: list[int] = []
        self.has_street = street_dim > 0
        self.has_naip = naip_dim > 0
        self.has_tabular = tabular_dim > 0

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
        self.primary = nn.Linear(hidden_dim, 1)
        self.sam_aux = nn.Linear(hidden_dim, n_sam_aux) if n_sam_aux > 0 else None
        self.public_aux = nn.Linear(hidden_dim, n_public_aux) if n_public_aux > 0 else None
        self._street_dim = street_dim
        self._naip_dim = naip_dim
        self._tabular_dim = tabular_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        primary = self.primary(fused).squeeze(-1)
        sam = self.sam_aux(fused) if self.sam_aux is not None else None
        public = self.public_aux(fused) if self.public_aux is not None else None
        return primary, sam, public


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan"), float("nan")
    try:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return float("nan"), float("nan")


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


def load_embeddings(city: str, source: str, backbone: str = "siglip") -> tuple[np.ndarray, np.ndarray]:
    emb_path = EMBED_DIR / f"{backbone}_{source}_{city}.npy"
    ids_path = EMBED_DIR / f"{backbone}_{source}_{city}_ids.npy"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found for {city}/{source} backbone={backbone}. "
            f"Expected:\n  {emb_path}\n  {ids_path}"
        )
    embs = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int32)
    return l2_normalize_rows(embs), ids


def align_embedding_block(lock_ids: np.ndarray, embs: np.ndarray, ids: np.ndarray, allow_missing: bool) -> tuple[np.ndarray, np.ndarray]:
    id_to_row = {int(pid): i for i, pid in enumerate(ids)}
    aligned = np.full((len(lock_ids), embs.shape[1]), np.nan, dtype=np.float32)
    present = np.zeros(len(lock_ids), dtype=bool)
    for i, pid in enumerate(lock_ids):
        row = id_to_row.get(int(pid))
        if row is not None:
            aligned[i] = embs[row]
            present[i] = True
    if not allow_missing and not present.all():
        missing_ids = lock_ids[~present][:10].tolist()
        raise ValueError(f"Missing required embeddings for IDs: {missing_ids}")
    return aligned, present


def fit_array_scaler(arr: np.ndarray, mask: np.ndarray | None = None) -> ArrayScaler:
    if arr.ndim == 1:
        arr = arr[:, None]
    if mask is None:
        mask = np.ones_like(arr, dtype=bool)
    mean = np.zeros(arr.shape[1], dtype=np.float32)
    scale = np.ones(arr.shape[1], dtype=np.float32)
    for j in range(arr.shape[1]):
        valid = mask[:, j]
        if valid.sum() == 0:
            continue
        vals = arr[valid, j].astype(np.float32)
        mean[j] = float(vals.mean())
        std = float(vals.std())
        scale[j] = std if std >= 1e-6 else 1.0
    return ArrayScaler(mean=mean, scale=scale)


def transform_with_scaler(arr: np.ndarray, scaler: ArrayScaler, fill_missing: bool = False) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[:, None]
    out = arr.astype(np.float32).copy()
    if fill_missing:
        fill = np.broadcast_to(scaler.mean, out.shape).astype(np.float32)
        out = np.where(np.isnan(out), fill, out)
    out = (out - scaler.mean) / scaler.scale
    return out.astype(np.float32)


def inverse_primary(arr: np.ndarray, scaler: ArrayScaler) -> np.ndarray:
    out = arr.astype(np.float32).reshape(-1, 1) * scaler.scale.reshape(1, -1) + scaler.mean.reshape(1, -1)
    return out[:, 0]


def compute_sample_weights(labels_city: pd.DataFrame, point_ids: np.ndarray, use_weights: bool) -> np.ndarray:
    if not use_weights:
        return np.ones(len(point_ids), dtype=np.float32)
    tiers = labels_city.loc[point_ids, "agreement_tier"].fillna("partial")
    return np.array([TIER_WEIGHTS.get(x, 0.5) for x in tiers], dtype=np.float32)


def stratified_split_indices(
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = pd.Series(y)
    bins = pd.qcut(series, q=4, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    use_test = (1.0 - train_frac - val_frac) > 1e-6

    for bucket in sorted(pd.Series(bins).dropna().unique().tolist()):
        idx = np.where(np.asarray(bins) == bucket)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if not use_test:
            n_train = n - n_val
            n_test = 0
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        if use_test:
            test_idx.extend(idx[n_train + n_val :].tolist())

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def tabular_fill_and_scale(
    train_arr: np.ndarray,
    other_arrs: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray | StandardScaler]]:
    fill = np.nanmedian(train_arr, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
    scaler = StandardScaler()
    train_filled = np.where(np.isnan(train_arr), fill, train_arr).astype(np.float32)
    train_scaled = scaler.fit_transform(train_filled).astype(np.float32)
    other_scaled: list[np.ndarray] = []
    for arr in other_arrs:
        filled = np.where(np.isnan(arr), fill, arr).astype(np.float32)
        other_scaled.append(scaler.transform(filled).astype(np.float32))
    return train_scaled, other_scaled, {"fill": fill, "scaler": scaler}


def prepare_city_data(
    city: str,
    labels: pd.DataFrame,
    captions: pd.DataFrame,
    spatial: pd.DataFrame,
    lock_ids: list[int],
    caption_cols: list[str],
    spatial_cols: list[str],
    backbone: str = "siglip",
    backbone_street: str | None = None,
    backbone_naip: str | None = None,
) -> dict[str, object]:
    # Per-modality backbone override: supports siglip2-street + remoteclip-naip fusion.
    # If backbone_street / backbone_naip are None, fall back to the shared backbone arg.
    _backbone_street = backbone_street or backbone
    _backbone_naip = backbone_naip or backbone
    city_labels = labels[labels["city"] == city].copy()
    city_labels["point_id"] = city_labels["point_id"].astype(int)
    city_labels = city_labels[city_labels["point_id"].isin(lock_ids)].copy()
    # features_labels_agreement.csv has one row per (city, point_id, heading) because it was
    # built by joining point-level label columns onto the per-heading caption table. Labels are
    # spatially constant per point, so we keep only the first row for each point_id.
    city_labels = city_labels.drop_duplicates(subset=["point_id"], keep="first")
    # log1p transform for heavy-tailed count target. Inversion happens at
    # evaluation time via the wrapper TargetTransform below.
    if "overture_intersection_count_200m" in city_labels.columns:
        col = city_labels["overture_intersection_count_200m"].to_numpy(dtype=np.float64)
        city_labels["overture_intersection_count_200m"] = np.log1p(np.clip(col, 0, None))
    city_labels = city_labels.set_index("point_id")
    point_ids = np.array([pid for pid in lock_ids if pid in city_labels.index], dtype=np.int32)

    captions_city = captions[captions["city"] == city].copy()
    captions_city["point_id"] = captions_city["point_id"].astype(int)
    captions_city = captions_city.set_index("point_id")

    spatial_city = spatial[spatial["city"] == city].copy()
    spatial_city["point_id"] = spatial_city["point_id"].astype(int)
    spatial_city = spatial_city.set_index("point_id")

    street_embs, street_ids = load_embeddings(city, "street", backbone=_backbone_street)
    street_aligned, street_present = align_embedding_block(point_ids, street_embs, street_ids, allow_missing=False)
    if not street_present.all():
        raise ValueError(f"Missing street embeddings for {city}.")

    naip_embs, naip_ids = load_embeddings(city, "naip", backbone=_backbone_naip)
    naip_aligned, naip_present = align_embedding_block(point_ids, naip_embs, naip_ids, allow_missing=True)

    if not point_ids.tolist():
        raise ValueError(f"No locked rows found for city={city}.")

    # Caption check removed: captions dropped from all ablations (O9+O21, O22, 2026-04-26).
    # Cities without captions (e.g. Pittsburgh) get zero-filled arrays; never used in feature matrix.
    missing = [int(pid) for pid in point_ids if pid not in spatial_city.index][:10]
    if missing:
        raise ValueError(f"Missing spatial features for city={city}, ids={missing}")

    return {
        "point_ids": point_ids,
        "label_frame": city_labels.loc[point_ids],
        "groups": {
            "street": street_aligned.astype(np.float32),
            "naip": naip_aligned.astype(np.float32),
            "naip_missing": (~naip_present).astype(np.float32)[:, None],
            "sam2": city_labels.loc[point_ids, [c for c in SAM2_INPUT_COLS if c in city_labels.columns]].to_numpy(dtype=np.float32),
            "captions": captions_city.reindex(point_ids)[caption_cols].fillna(0.0).to_numpy(dtype=np.float32) if caption_cols else np.zeros((len(point_ids), 0), dtype=np.float32),
            "spatial": spatial_city.loc[point_ids, spatial_cols].to_numpy(dtype=np.float32),
        },
        "targets": {
            "primary": city_labels.loc[point_ids, PRIMARY_TARGET].to_numpy(dtype=np.float32),
            "sam_aux": city_labels.loc[point_ids, SAM2_AUX_COLS].to_numpy(dtype=np.float32),
            "public_aux": city_labels.loc[point_ids, PUBLIC_AUX_COLS].to_numpy(dtype=np.float32),
        },
    }


def feature_groups_for_run(ablation: str) -> tuple[str, ...]:
    return FEATURES_BY_ABLATION[ablation]


def build_feature_matrices(
    train_data: dict[str, object],
    val_data: dict[str, object],
    test_data: dict[str, object],
    feature_groups: tuple[str, ...],
    model_family: str,
    pca_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_groups = train_data["groups"]
    val_groups = val_data["groups"]
    test_groups = test_data["groups"]

    transforms: dict[str, object] = {}
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    group_dims: dict[str, int] = {}

    if "street" in feature_groups:
        tr_s = train_groups["street"].astype(np.float32)
        va_s = val_groups["street"].astype(np.float32)
        te_s = test_groups["street"].astype(np.float32)
        if pca_dim > 0 and pca_dim < tr_s.shape[1]:
            pca_street = PCA(n_components=pca_dim, random_state=SEED)
            tr_s = pca_street.fit_transform(tr_s).astype(np.float32)
            va_s = pca_street.transform(va_s).astype(np.float32)
            te_s = pca_street.transform(te_s).astype(np.float32)
            transforms["pca_street"] = pca_street
            var_explained = float(pca_street.explained_variance_ratio_.sum())
            logging.info("  PCA street: %d→%d dims, cumvar=%.3f", train_groups["street"].shape[1], pca_dim, var_explained)
        train_parts.append(tr_s)
        val_parts.append(va_s)
        test_parts.append(te_s)
        group_dims["street"] = tr_s.shape[1]

    if "naip" in feature_groups:
        train_naip = train_groups["naip"]
        fill = np.nanmean(train_naip, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
        transforms["naip_fill"] = fill

        tr_n_raw = np.where(np.isnan(train_naip), fill, train_naip).astype(np.float32)
        va_n_raw = np.where(np.isnan(val_groups["naip"]), fill, val_groups["naip"]).astype(np.float32)
        te_n_raw = np.where(np.isnan(test_groups["naip"]), fill, test_groups["naip"]).astype(np.float32)
        tr_n_raw = l2_normalize_rows(tr_n_raw)
        va_n_raw = l2_normalize_rows(va_n_raw)
        te_n_raw = l2_normalize_rows(te_n_raw)

        if pca_dim > 0 and pca_dim < tr_n_raw.shape[1]:
            pca_naip = PCA(n_components=pca_dim, random_state=SEED)
            tr_n = pca_naip.fit_transform(tr_n_raw).astype(np.float32)
            va_n = pca_naip.transform(va_n_raw).astype(np.float32)
            te_n = pca_naip.transform(te_n_raw).astype(np.float32)
            transforms["pca_naip"] = pca_naip
            var_explained = float(pca_naip.explained_variance_ratio_.sum())
            logging.info("  PCA naip: %d→%d dims, cumvar=%.3f", train_naip.shape[1], pca_dim, var_explained)
        else:
            tr_n, va_n, te_n = tr_n_raw, va_n_raw, te_n_raw

        for parts, arr, missing in (
            (train_parts, tr_n, train_groups["naip_missing"]),
            (val_parts, va_n, val_groups["naip_missing"]),
            (test_parts, te_n, test_groups["naip_missing"]),
        ):
            parts.append(arr)
            parts.append(missing.astype(np.float32))
        group_dims["naip"] = tr_n.shape[1] + 1

    for name in ("captions", "sam2", "spatial"):
        if name not in feature_groups:
            continue
        train_scaled, [val_scaled, test_scaled], transform = tabular_fill_and_scale(
            train_groups[name],
            [val_groups[name], test_groups[name]],
        )
        transforms[f"{name}_fill"] = transform["fill"]
        transforms[f"{name}_scaler_mean"] = transform["scaler"].mean_.astype(np.float32)
        transforms[f"{name}_scaler_scale"] = transform["scaler"].scale_.astype(np.float32)
        train_parts.append(train_scaled)
        val_parts.append(val_scaled)
        test_parts.append(test_scaled)
        group_dims[name] = train_scaled.shape[1]

    X_train = np.concatenate(train_parts, axis=1).astype(np.float32)
    X_val = np.concatenate(val_parts, axis=1).astype(np.float32)
    X_test = np.concatenate(test_parts, axis=1).astype(np.float32)

    if model_family in {"linear", "ridge", "elasticnet"}:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        transforms["global_feature_scaler_mean"] = scaler.mean_.astype(np.float32)
        transforms["global_feature_scaler_scale"] = scaler.scale_.astype(np.float32)

    transforms["group_dims"] = group_dims
    return X_train, X_val, X_test, transforms


def align_targets_for_split(
    city_data: dict[str, object],
    idx: np.ndarray,
    weights: np.ndarray,
    primary_scaler: ArrayScaler,
    sam_scaler: ArrayScaler | None,
    public_scaler: ArrayScaler | None,
) -> dict[str, np.ndarray]:
    primary = city_data["targets"]["primary"][idx]
    sam_aux = city_data["targets"]["sam_aux"][idx]
    public_aux = city_data["targets"]["public_aux"][idx]
    public_mask = np.isfinite(public_aux)

    out = {
        "primary_raw": primary.astype(np.float32),
        "primary": transform_with_scaler(primary, primary_scaler).reshape(-1),
        "weights": weights[idx].astype(np.float32),
        "sam_aux": transform_with_scaler(sam_aux, sam_scaler).astype(np.float32) if sam_scaler else np.zeros((len(idx), 0), dtype=np.float32),
        "public_aux": transform_with_scaler(public_aux, public_scaler, fill_missing=True).astype(np.float32) if public_scaler else np.zeros((len(idx), 0), dtype=np.float32),
        "public_mask": public_mask.astype(np.float32),
        "point_ids": city_data["point_ids"][idx],
    }
    return out


def evaluate_primary_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    r, p = safe_pearsonr(pred, y_true)
    spearman_rho, _ = stats.spearmanr(pred, y_true)
    return {
        "r2": round(float(r2_score(y_true, pred)), 4),
        "rmse": round(float(np.sqrt(np.mean((pred - y_true) ** 2))), 4),
        "mae": round(float(np.mean(np.abs(pred - y_true))), 4),
        "pearson_r": round(r, 4) if np.isfinite(r) else float("nan"),
        "pearson_p": round(p, 6) if np.isfinite(p) else float("nan"),
        "spearman_rho": round(float(spearman_rho), 4) if np.isfinite(spearman_rho) else float("nan"),
    }


def evaluate_primary_metrics_recalibrated(
    y_true: np.ndarray,
    pred: np.ndarray,
    train_mean: float,
    train_std: float,
    name: str | None = None,
) -> dict[str, float]:
    """Z-score pred with training city stats, rescale to test city distribution."""
    if name == "overture_intersection_count_200m":
        y_true = np.expm1(y_true)
        pred = np.expm1(pred)
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
    """Average point predictions within block groups, compute BG-level R²."""
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
        return {"bg_r2": float("nan"), "bg_pearson_r": float("nan"), "n_block_groups": int(len(bg))}
    bg_r2 = float(r2_score(bg["true_mean"], bg["pred_mean"]))
    bg_r, _ = stats.pearsonr(bg["pred_mean"], bg["true_mean"])
    return {
        "bg_r2": round(bg_r2, 4),
        "bg_pearson_r": round(float(bg_r), 4),
        "n_block_groups": int(len(bg)),
    }


def secondary_correlations(label_frame: pd.DataFrame, point_ids: np.ndarray, primary_pred: np.ndarray) -> dict[str, float]:
    aligned = label_frame.loc[point_ids]
    out: dict[str, float] = {}
    for col in SECONDARY_TARGETS:
        vals = aligned[col].to_numpy(dtype=np.float32)
        r, _ = safe_pearsonr(primary_pred, vals)
        out[col] = round(r, 4) if np.isfinite(r) else float("nan")
    return out


def auxiliary_head_metrics(
    pred_scaled: np.ndarray | None,
    actual_raw: np.ndarray,
    mask: np.ndarray,
    scaler: ArrayScaler | None,
    columns: list[str],
) -> dict[str, float]:
    if pred_scaled is None or scaler is None or len(columns) == 0:
        return {}
    pred_raw = pred_scaled * scaler.scale.reshape(1, -1) + scaler.mean.reshape(1, -1)
    out: dict[str, float] = {}
    for j, col in enumerate(columns):
        valid = mask[:, j].astype(bool)
        r, _ = safe_pearsonr(pred_raw[valid, j], actual_raw[valid, j])
        out[col] = round(r, 4) if np.isfinite(r) else float("nan")
    return out


def train_torch_model(
    cfg: RunConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_t: dict[str, np.ndarray],
    val_t: dict[str, np.ndarray],
    test_t: dict[str, np.ndarray],
    primary_scaler: ArrayScaler,
    sam_scaler: ArrayScaler | None,
    public_scaler: ArrayScaler | None,
    feature_groups: tuple[str, ...],
    feature_transforms: dict[str, object],
    device: torch.device,
    label_frame_test: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    linear_backbone = cfg.ablation == "sam2_only"
    n_sam_aux = len(SAM2_AUX_COLS) if ("sam2" in feature_groups and cfg.sam_aux_weight > 0) else 0
    n_public_aux = len(PUBLIC_AUX_COLS) if cfg.public_aux_weight > 0 else 0

    if cfg.architecture == "multimodal":
        # Recover per-modality dims from feature_transforms (set during build_feature_matrices)
        group_dims = feature_transforms.get("group_dims", {})
        street_dim = group_dims.get("street", 0)
        naip_dim = group_dims.get("naip", 0)
        tabular_dim = sum(v for k, v in group_dims.items() if k not in ("street", "naip"))
        model = WalkCLIPMultiModal(
            street_dim=street_dim,
            naip_dim=naip_dim,
            tabular_dim=tabular_dim,
            hidden_dim=128,
            n_sam_aux=n_sam_aux,
            n_public_aux=n_public_aux,
        ).to(device)
        logging.info(
            "  WalkCLIPMultiModal: street=%d naip=%d tabular=%d", street_dim, naip_dim, tabular_dim
        )
    else:
        model = WalkCLIPMultitask(X_train.shape[1], n_sam_aux, n_public_aux, linear_backbone=linear_backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    def cpu_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr))

    ds_train = TensorDataset(
        cpu_tensor(X_train),
        cpu_tensor(train_t["primary"]),
        cpu_tensor(train_t["sam_aux"]),
        cpu_tensor(train_t["public_aux"]),
        cpu_tensor(train_t["public_mask"]),
        cpu_tensor(train_t["weights"]),
    )
    loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    X_val_t = cpu_tensor(X_val).to(device)
    X_test_t = cpu_tensor(X_test).to(device)
    y_val_t = cpu_tensor(val_t["primary"]).to(device)
    w_val_t = cpu_tensor(val_t["weights"]).to(device)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = cfg.patience
    metrics_path = LOG_DIR / "metrics_v2.jsonl"

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        for Xb, yb, y_sam_b, y_pub_b, pub_mask_b, wb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            y_sam_b = y_sam_b.to(device, non_blocking=True)
            y_pub_b = y_pub_b.to(device, non_blocking=True)
            pub_mask_b = pub_mask_b.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            pred_primary, pred_sam, pred_public = model(Xb)
            primary_loss = (((pred_primary - yb) ** 2) * wb).mean()

            sam_loss = torch.tensor(0.0, device=device)
            if pred_sam is not None and y_sam_b.numel() > 0:
                sam_loss = nn.functional.mse_loss(pred_sam, y_sam_b)

            public_loss = torch.tensor(0.0, device=device)
            if pred_public is not None and y_pub_b.numel() > 0:
                diff = (pred_public - y_pub_b) ** 2
                public_loss = (diff * pub_mask_b).sum() / pub_mask_b.sum().clamp_min(1.0)

            loss = primary_loss + cfg.sam_aux_weight * sam_loss + cfg.public_aux_weight * public_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_primary_pred, _, _ = model(X_val_t)
        val_loss = float((((val_primary_pred - y_val_t) ** 2) * w_val_t).mean().item())
        val_pred = inverse_primary(val_primary_pred.detach().cpu().numpy(), primary_scaler)
        val_r2 = float(r2_score(val_t["primary_raw"], val_pred))

        if epoch % 10 == 0 or epoch == 1:
            row = {
                "epoch": epoch,
                "val_weighted_mse": round(val_loss, 6),
                "val_r2": round(val_r2, 4),
                "train_city": cfg.train_city,
                "test_city": cfg.test_city,
                "eval_mode": cfg.eval_mode,
                "ablation": cfg.ablation,
                "model_family": cfg.model_family,
            }
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                logging.info("  Early stop at epoch %d (best=%d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        test_primary_pred, test_sam_pred, test_public_pred = model(X_test_t)
        val_primary_pred, _, _ = model(X_val_t)

    test_pred = inverse_primary(test_primary_pred.cpu().numpy(), primary_scaler)
    val_pred = inverse_primary(val_primary_pred.cpu().numpy(), primary_scaler)

    train_mean = float(primary_scaler.mean[0])
    train_std = float(primary_scaler.scale[0])
    primary_metrics = evaluate_primary_metrics(test_t["primary_raw"], test_pred)
    recal_metrics = evaluate_primary_metrics_recalibrated(test_t["primary_raw"], test_pred, train_mean, train_std)
    bg_metrics = evaluate_block_group_aggregated(test_t["point_ids"], test_pred, label_frame_test)
    result = {
        "train_city": cfg.train_city,
        "test_city": cfg.test_city,
        "eval_mode": cfg.eval_mode,
        "ablation": cfg.ablation,
        "model_family": cfg.model_family,
        "sam_aux_weight": cfg.sam_aux_weight,
        "public_aux_weight": cfg.public_aux_weight,
        "feature_groups": list(feature_groups),
        "best_epoch": best_epoch,
        "elapsed_s": round(time.time() - t0, 1),
        "primary_metrics": {**primary_metrics, **recal_metrics},
        "block_group_metrics": bg_metrics,
        "train_mean": round(train_mean, 4),
        "train_std": round(train_std, 4),
        "val_primary_r2": round(float(r2_score(val_t["primary_raw"], val_pred)), 4),
        "secondary_correlations": secondary_correlations(label_frame_test, test_t["point_ids"], test_pred),
        "auxiliary_head_metrics": {
            "public_heads": {},
            "sam_heads": {},
        },
    }

    # Fill auxiliary metrics after constructing the dict to keep typing simple.
    result["auxiliary_head_metrics"]["public_heads"] = auxiliary_head_metrics(
        test_public_pred.cpu().numpy() if test_public_pred is not None else None,
        actual_raw=test_t["public_aux_raw"],
        mask=test_t["public_mask"],
        scaler=public_scaler,
        columns=PUBLIC_AUX_COLS if test_public_pred is not None else [],
    )
    result["auxiliary_head_metrics"]["sam_heads"] = auxiliary_head_metrics(
        test_sam_pred.cpu().numpy() if test_sam_pred is not None else None,
        actual_raw=test_t["sam_aux_raw"],
        mask=np.ones_like(test_t["sam_aux_raw"], dtype=bool),
        scaler=sam_scaler,
        columns=SAM2_AUX_COLS if test_sam_pred is not None else [],
    )

    checkpoint = {
        "model_state_dict": best_state,
        "config": asdict(cfg),
        "result": result,
        "feature_groups": list(feature_groups),
        "primary_scaler": asdict(primary_scaler),
        "sam_scaler": asdict(sam_scaler) if sam_scaler else None,
        "public_scaler": asdict(public_scaler) if public_scaler else None,
    }
    return result, checkpoint


def train_linear_baseline(
    cfg: RunConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    train_t: dict[str, np.ndarray],
    val_t: dict[str, np.ndarray],
    test_t: dict[str, np.ndarray],
    primary_scaler: ArrayScaler,
    feature_groups: tuple[str, ...],
    label_frame_test: pd.DataFrame,
) -> tuple[dict[str, object], object]:
    if cfg.model_family == "linear":
        model = LinearRegression()
        fit_kwargs = {"sample_weight": train_t["weights"]}
    elif cfg.model_family == "ridge":
        model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed)
        fit_kwargs = {"sample_weight": train_t["weights"]}
    else:
        model = ElasticNet(
            alpha=cfg.elasticnet_alpha,
            l1_ratio=cfg.elasticnet_l1_ratio,
            max_iter=20000,
            random_state=cfg.seed,
        )
        fit_kwargs = {}

    t0 = time.time()
    model.fit(X_train, train_t["primary"], **fit_kwargs)
    val_pred = inverse_primary(model.predict(X_val).astype(np.float32), primary_scaler)
    test_pred = inverse_primary(model.predict(X_test).astype(np.float32), primary_scaler)
    train_mean = float(primary_scaler.mean[0])
    train_std = float(primary_scaler.scale[0])
    primary_metrics = evaluate_primary_metrics(test_t["primary_raw"], test_pred)
    recal_metrics = evaluate_primary_metrics_recalibrated(test_t["primary_raw"], test_pred, train_mean, train_std)
    bg_metrics = evaluate_block_group_aggregated(test_t["point_ids"], test_pred, label_frame_test)
    result = {
        "train_city": cfg.train_city,
        "test_city": cfg.test_city,
        "eval_mode": cfg.eval_mode,
        "ablation": cfg.ablation,
        "model_family": cfg.model_family,
        "feature_groups": list(feature_groups),
        "best_epoch": 1,
        "elapsed_s": round(time.time() - t0, 1),
        "primary_metrics": {**primary_metrics, **recal_metrics},
        "block_group_metrics": bg_metrics,
        "train_mean": round(train_mean, 4),
        "train_std": round(train_std, 4),
        "val_primary_r2": round(float(r2_score(val_t["primary_raw"], val_pred)), 4),
        "secondary_correlations": secondary_correlations(label_frame_test, test_t["point_ids"], test_pred),
        "auxiliary_head_metrics": {},
    }
    return result, model


def run_one(cfg: RunConfig, labels: pd.DataFrame, captions: pd.DataFrame, spatial: pd.DataFrame, lock_ids: list[int], device: torch.device) -> tuple[dict[str, object], dict[str, object] | object]:
    set_seeds(cfg.seed)
    caption_cols = [c for c in captions.columns if c.startswith("cap_")]
    spatial_cols = [c for c in spatial.columns if c.startswith("spatial_")]
    feature_groups = feature_groups_for_run(cfg.ablation)

    if "captions" in feature_groups and not caption_cols:
        raise ValueError("Caption features requested but no `cap_` columns found.")
    if "spatial" in feature_groups and not spatial_cols:
        raise ValueError("Spatial features requested but no `spatial_` columns found.")

    train_city_full = prepare_city_data(cfg.train_city, labels, captions, spatial, lock_ids, caption_cols, spatial_cols)
    use_weights = cfg.ablation != "no_weights"
    weights_full = compute_sample_weights(train_city_full["label_frame"], train_city_full["point_ids"], use_weights=use_weights)

    if cfg.eval_mode == "cross_city":
        train_idx, val_idx, _ = stratified_split_indices(
            train_city_full["targets"]["primary"],
            train_frac=cfg.train_split,
            val_frac=cfg.val_split,
            seed=cfg.seed,
        )
        test_city_full = prepare_city_data(cfg.test_city, labels, captions, spatial, lock_ids, caption_cols, spatial_cols)
        test_idx = np.arange(len(test_city_full["point_ids"]), dtype=int)
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(
            train_city_full["targets"]["primary"],
            train_frac=cfg.train_split,
            val_frac=cfg.val_split,
            seed=cfg.seed,
        )
        test_city_full = train_city_full

    train_split_data = {
        "groups": {k: v[train_idx] for k, v in train_city_full["groups"].items()},
        "targets": {k: v[train_idx] for k, v in train_city_full["targets"].items()},
        "point_ids": train_city_full["point_ids"][train_idx],
        "label_frame": train_city_full["label_frame"],
    }
    val_split_data = {
        "groups": {k: v[val_idx] for k, v in train_city_full["groups"].items()},
        "targets": {k: v[val_idx] for k, v in train_city_full["targets"].items()},
        "point_ids": train_city_full["point_ids"][val_idx],
        "label_frame": train_city_full["label_frame"],
    }
    test_split_data = {
        "groups": {k: v[test_idx] for k, v in test_city_full["groups"].items()},
        "targets": {k: v[test_idx] for k, v in test_city_full["targets"].items()},
        "point_ids": test_city_full["point_ids"][test_idx],
        "label_frame": test_city_full["label_frame"],
    }

    X_train, X_val, X_test, feature_transforms = build_feature_matrices(
        train_split_data,
        val_split_data,
        test_split_data,
        feature_groups=feature_groups,
        model_family=cfg.model_family,
        pca_dim=cfg.pca_dim,
    )

    primary_scaler = fit_array_scaler(train_split_data["targets"]["primary"])
    sam_scaler = fit_array_scaler(train_split_data["targets"]["sam_aux"]) if ("sam2" in feature_groups and cfg.sam_aux_weight > 0 and cfg.model_family == "mlp") else None
    public_mask_train = np.isfinite(train_split_data["targets"]["public_aux"])
    public_scaler = fit_array_scaler(train_split_data["targets"]["public_aux"], public_mask_train) if (cfg.public_aux_weight > 0 and cfg.model_family == "mlp") else None

    train_targets = align_targets_for_split(train_city_full, train_idx, weights_full, primary_scaler, sam_scaler, public_scaler)
    val_targets = align_targets_for_split(train_city_full, val_idx, weights_full, primary_scaler, sam_scaler, public_scaler)
    test_weights = compute_sample_weights(test_city_full["label_frame"], test_city_full["point_ids"], use_weights=use_weights)
    test_targets = align_targets_for_split(test_city_full, test_idx, test_weights, primary_scaler, sam_scaler, public_scaler)
    train_targets["sam_aux_raw"] = train_split_data["targets"]["sam_aux"].astype(np.float32)
    val_targets["sam_aux_raw"] = val_split_data["targets"]["sam_aux"].astype(np.float32)
    test_targets["sam_aux_raw"] = test_split_data["targets"]["sam_aux"].astype(np.float32)
    train_targets["public_aux_raw"] = train_split_data["targets"]["public_aux"].astype(np.float32)
    val_targets["public_aux_raw"] = val_split_data["targets"]["public_aux"].astype(np.float32)
    test_targets["public_aux_raw"] = test_split_data["targets"]["public_aux"].astype(np.float32)

    label_frame_test = test_city_full["label_frame"]

    if cfg.model_family == "mlp":
        result, artifact = train_torch_model(
            cfg=cfg,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            train_t=train_targets,
            val_t=val_targets,
            test_t=test_targets,
            primary_scaler=primary_scaler,
            sam_scaler=sam_scaler,
            public_scaler=public_scaler,
            feature_groups=feature_groups,
            feature_transforms=feature_transforms,
            device=device,
            label_frame_test=label_frame_test,
        )
    else:
        result, artifact = train_linear_baseline(
            cfg=cfg,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            train_t=train_targets,
            val_t=val_targets,
            test_t=test_targets,
            primary_scaler=primary_scaler,
            feature_groups=feature_groups,
            label_frame_test=label_frame_test,
        )

    result["feature_transforms"] = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in feature_transforms.items()
        if k == "group_dims"
    }
    return result, artifact


def save_artifact(cfg: RunConfig, artifact: dict[str, object] | object) -> Path:
    stem = f"walkclip_{cfg.eval_mode}_{cfg.train_city}_to_{cfg.test_city}_{cfg.ablation}_{cfg.model_family}"
    if cfg.model_family == "mlp":
        path = MODEL_DIR / f"{stem}.pt"
        torch.save(artifact, path)
    else:
        path = MODEL_DIR / f"{stem}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh)
    return path


def print_results_table(results: list[dict[str, object]]) -> None:
    logging.info("")
    logging.info("─" * 140)
    logging.info("RESULTS")
    logging.info(
        "%-12s %-10s %-12s %-17s %7s %7s %7s %7s %7s %7s %7s",
        "mode", "model", "ablation", "train→test", "R²", "R²_recal", "ρ", "RMSE", "valR²", "BG_R²", "n_BG",
    )
    logging.info("─" * 140)
    for res in sorted(results, key=lambda x: (-x["primary_metrics"]["r2"],)):
        m = res["primary_metrics"]
        bg = res.get("block_group_metrics", {})
        logging.info(
            "%-12s %-10s %-12s %s→%-8s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7d",
            res["eval_mode"],
            res["model_family"],
            res["ablation"],
            res["train_city"],
            res["test_city"],
            m["r2"],
            m.get("r2_recalibrated", float("nan")),
            m.get("spearman_rho", float("nan")),
            m["rmse"],
            res["val_primary_r2"],
            bg.get("bg_r2", float("nan")),
            bg.get("n_block_groups", 0),
        )
    logging.info("─" * 140)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced Stage E trainer for WalkCLIP v2")
    p.add_argument("--train-city", choices=["msp", "seattle"], help="Train city for cross-city mode")
    p.add_argument("--test-city", choices=["msp", "seattle"], help="Test city for cross-city mode")
    p.add_argument("--in-city-city", choices=["msp", "seattle"], help="Run in-city sanity split for a single city")
    p.add_argument("--ablation", choices=sorted(FEATURES_BY_ABLATION), default="full")
    p.add_argument("--model-family", choices=["mlp", "linear", "ridge", "elasticnet"], default="mlp")
    p.add_argument("--architecture", choices=["multitask", "multimodal"], default="multitask",
                   help="multitask=original single-MLP; multimodal=modality-specific projection heads (Phase 2)")
    p.add_argument("--pca-dim", type=int, default=0,
                   help="PCA compress street+naip embeddings to N dims (0=no PCA; try 64/128/256)")
    p.add_argument("--sweep", action="store_true", help="Run the curated cross-city ablation sweep")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--train-split", type=float, default=0.85)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sam-aux-weight", type=float, default=0.15)
    p.add_argument("--public-aux-weight", type=float, default=0.20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--elasticnet-alpha", type=float, default=0.01)
    p.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    for d in (MODEL_DIR, LOG_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for path in (LOCK_FILE, LABELS_CSV, CAPTION_CSV, SPATIAL_CSV):
        if not path.exists():
            logging.error("Required file not found: %s", path)
            return 1
    if not EMBED_DIR.exists() or not any(EMBED_DIR.glob("siglip_*.npy")):
        logging.error("No embeddings found in %s. Run extract_siglip_embeddings.py first.", EMBED_DIR)
        return 1

    lock_ids = [int(x) for x in LOCK_FILE.read_text().strip().splitlines()]
    labels = pd.read_csv(LABELS_CSV)
    wi_path = LABELS_CSV.parent / "walkability_index.csv"
    if wi_path.exists():
        wi = pd.read_csv(wi_path)[["point_id", "city", "walkability_index_v1"]]
        labels = labels.merge(wi, on=["point_id", "city"], how="left")
    else:
        logging.warning(
            "walkability_index.csv not found; walkability_index_v1 target will be all-NaN."
        )

    captions = pd.read_csv(CAPTION_CSV)
    spatial = pd.read_csv(SPATIAL_CSV)
    labels["point_id"] = labels["point_id"].astype(int)
    captions["point_id"] = captions["point_id"].astype(int)
    spatial["point_id"] = spatial["point_id"].astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("Labels=%d  Captions=%d  Spatial=%d  Lock=%d", len(labels), len(captions), len(spatial), len(lock_ids))

    configs: list[RunConfig] = []
    if args.sweep:
        for train_city, test_city in [("msp", "seattle"), ("seattle", "msp")]:
            for ablation in SWEEP_ABLATIONS:
                configs.append(
                    RunConfig(
                        train_city=train_city,
                        test_city=test_city,
                        eval_mode="cross_city",
                        ablation=ablation,
                        model_family=args.model_family,
                        architecture=args.architecture,
                        pca_dim=args.pca_dim,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        patience=args.patience,
                        train_split=args.train_split,
                        val_split=args.val_split,
                        seed=SEED,
                        sam_aux_weight=args.sam_aux_weight,
                        public_aux_weight=args.public_aux_weight,
                        ridge_alpha=args.ridge_alpha,
                        elasticnet_alpha=args.elasticnet_alpha,
                        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
                    )
                )
    elif args.in_city_city:
        configs.append(
            RunConfig(
                train_city=args.in_city_city,
                test_city=args.in_city_city,
                eval_mode="in_city",
                ablation=args.ablation,
                model_family=args.model_family,
                architecture=args.architecture,
                pca_dim=args.pca_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                patience=args.patience,
                train_split=0.70,
                val_split=0.15,
                seed=SEED,
                sam_aux_weight=args.sam_aux_weight,
                public_aux_weight=args.public_aux_weight,
                ridge_alpha=args.ridge_alpha,
                elasticnet_alpha=args.elasticnet_alpha,
                elasticnet_l1_ratio=args.elasticnet_l1_ratio,
            )
        )
    else:
        if not args.train_city or not args.test_city:
            logging.error("Provide --train-city and --test-city, use --in-city-city, or use --sweep.")
            return 1
        if args.train_city == args.test_city:
            logging.error("Use --in-city-city for same-city sanity runs.")
            return 1
        configs.append(
            RunConfig(
                train_city=args.train_city,
                test_city=args.test_city,
                eval_mode="cross_city",
                ablation=args.ablation,
                model_family=args.model_family,
                architecture=args.architecture,
                pca_dim=args.pca_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                patience=args.patience,
                train_split=args.train_split,
                val_split=args.val_split,
                seed=SEED,
                sam_aux_weight=args.sam_aux_weight,
                public_aux_weight=args.public_aux_weight,
                ridge_alpha=args.ridge_alpha,
                elasticnet_alpha=args.elasticnet_alpha,
                elasticnet_l1_ratio=args.elasticnet_l1_ratio,
            )
        )

    results: list[dict[str, object]] = []
    for cfg in configs:
        logging.info(
            "=== %s | %s | %s → %s | %s ===",
            cfg.eval_mode,
            cfg.model_family,
            cfg.train_city,
            cfg.test_city,
            cfg.ablation,
        )
        result, artifact = run_one(cfg, labels, captions, spatial, lock_ids, device)
        artifact_path = save_artifact(cfg, artifact)
        result["artifact_path"] = str(artifact_path)
        logging.info(
            "  R²=%.4f  RMSE=%.4f  r=%.4f  valR²=%.4f  saved=%s",
            result["primary_metrics"]["r2"],
            result["primary_metrics"]["rmse"],
            result["primary_metrics"]["pearson_r"],
            result["val_primary_r2"],
            artifact_path.name,
        )
        results.append(result)

    print_results_table(results)
    summary_path = REPORT_DIR / "results_summary_v2.json"
    existing: list[dict[str, object]] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
    summary_path.write_text(json.dumps(existing + results, indent=2))
    logging.info("Results saved: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
