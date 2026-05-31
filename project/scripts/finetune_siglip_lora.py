"""
Stage E+ : LoRA fine-tune of SigLIP-SO400M on heading-level Overture supervision.

Backbone: google/siglip-so400m-patch14-384 (vision encoder only).
Adaptation: PEFT LoRA on attention projections (q/k/v/out_proj), r=16, alpha=32,
            dropout 0.1. ~4M trainable params on a 432M-param backbone (~0.93%).
Supervision: per-(point_id, heading) Overture wedge labels — sidewalk_present
             and crosswalk_present binaries — across all 3 cities (~58.7k rows).
Heads: two parallel nn.Linear(1152, 1) classification heads on the SigLIP
       pooled CLS embedding, trained with class-weighted BCE.
Split: 80/10/10 train/val/test, stratified by (city, sidewalk_present), seed=42.
       Test indices are saved to disk for exact reviewer reproducibility.
Augmentation (training only): random horizontal flip with p=0.5. Streetscapes
       are roughly mirror-symmetric for the binary targets; this is the standard
       light-aug protocol used by CLIP / SigLIP fine-tunes in the literature.
Mixed precision: bf16 autocast on RTX 5070 Ti / A100 / H100. Optimizer: AdamW
(lr 1e-4 on LoRA, 1e-3 on heads), cosine schedule with 5% warmup, grad-clip 1.0.
Early stopping on mean(val sidewalk AUROC, val crosswalk AUROC), patience 3.

Pre-requisites
--------------
1) Build the pixel-value cache (one-time, ~25–40 min, ~52 GB on ext4):
    python project/scripts/cache_pixel_values.py --output-dir ~/walkclip_cache
2) Install peft:
    pip install peft

Why a cache: WSL on /mnt/c reads each JPEG via 9P (5–20 ms). Decoding 32 imgs/iter
× 4 workers leaves the GPU at <1 % utilisation (confirmed by 68 W draw on a 300 W
TDP card). A pre-computed float16 memmap on native ext4 takes per-iter time from
~41 s to ~120 ms — a ~270× speedup with bit-exact deterministic preprocessing.

Outputs (project/artifacts/models/siglip_lora_v1/):
    adapter/                          # PEFT adapter weights (PeftModel.from_pretrained-able)
    classification_heads.pt           # the two nn.Linear heads (for diagnostics)
    train_log.jsonl                   # per-step training metrics (loss, lrs, samples/sec)
    val_metrics.json                  # final val + test AUROC + AP
    config.json                       # hyperparameters + env (torch/peft/HF versions, GPU)
    splits.json                       # train/val/test cache_idx — reproducibility-locked

Run (WSL, GPU):
    python project/scripts/finetune_siglip_lora.py \\
        --cache-dir ~/walkclip_cache \\
        --epochs 10 --batch-size 64 --lr 1e-4 --head-lr 1e-3 \\
        --output project/artifacts/models/siglip_lora_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "project/data/processed/labels"


def load_multitask_regression_labels() -> pd.DataFrame:
    """Load point-level regression targets for multi-task LoRA training.

    Returns DataFrame with columns: point_id, city, intersection_count_log1p,
    building_footprint_frac.
    """
    over = pd.read_csv(LABELS_DIR / "overture_targets.csv")
    over_keep = ["point_id", "city",
                 "intersection_count_200m", "building_footprint_frac_100m"]
    over = over[over_keep].copy()
    over["intersection_count_log1p"] = np.log1p(np.clip(
        over["intersection_count_200m"].to_numpy(dtype=np.float64), 0, None))
    over = over.drop(columns=["intersection_count_200m"])
    over = over.rename(columns={"building_footprint_frac_100m": "building_footprint_frac"})
    return over


MODEL_ID = "google/siglip-so400m-patch14-384"
EMBED_DIM = 1152
SEED = 42
LORA_TARGET_MODULES_ATTN = ["q_proj", "k_proj", "v_proj", "out_proj"]
LORA_TARGET_MODULES_FULL = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@dataclass
class LoRAConfig:
    epochs: int
    batch_size: int
    lr: float
    head_lr: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_frac: float
    patience: int
    num_workers: int
    seed: int
    output_dir: str
    cache_dir: str
    horizontal_flip: bool
    grad_clip: float
    grad_checkpoint: bool
    grad_accum_steps: int
    holdout_city: str = "none"  # "none" | "msp" | "seattle" | "dc" — exclude this city's rows from train/val/test entirely
    multitask: bool = False  # if True, train 4 heads (sidewalk, crosswalk, intersection, building-frac)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed if info else 0
    seed = (base_seed + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def load_cache(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    arr_path = cache_dir / "siglip_pixels_street_384.npy"
    idx_path = cache_dir / "siglip_pixels_street_384_index.csv"
    man_path = cache_dir / "cache_manifest.json"
    for p in (arr_path, idx_path, man_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Cache file missing: {p}\n"
                f"Run:  python project/scripts/cache_pixel_values.py --output-dir {cache_dir}"
            )
    arr = np.load(arr_path, mmap_mode="r")
    df  = pd.read_csv(idx_path)
    manifest = json.loads(man_path.read_text())
    if arr.shape[0] != len(df):
        raise RuntimeError(f"Cache mismatch: array has {arr.shape[0]} rows, index has {len(df)}")
    logging.info(
        "Loaded cache: %d rows | shape=%s | dtype=%s | manifest model_id=%s",
        len(df), arr.shape, arr.dtype, manifest.get("model_id"),
    )
    return arr, df, manifest


class CachedTensorDataset(Dataset):
    """Reads pre-computed SigLIP pixel_values from a memmap. Zero-copy, no PIL, no processor."""

    def __init__(
        self,
        cache_arr: np.ndarray,
        labels: pd.DataFrame,
        cache_indices: np.ndarray,
        *,
        horizontal_flip: bool = False,
        multitask: bool = False,
    ) -> None:
        self.cache_arr = cache_arr
        self.cache_indices = np.asarray(cache_indices, dtype=np.int64)
        if "cache_idx" not in labels.columns:
            raise ValueError("labels dataframe must have a 'cache_idx' column")
        self.labels = (
            labels.set_index("cache_idx", drop=False)
                  .loc[self.cache_indices]
                  .reset_index(drop=True)
        )
        if len(self.labels) != len(self.cache_indices):
            raise RuntimeError(
                f"Label lookup mismatch: {len(self.cache_indices)} cache_idx requested, "
                f"{len(self.labels)} returned. Possible cause: cache_idx duplicates or holdout "
                f"filter removed indices that the splitter still references."
            )
        self.horizontal_flip = horizontal_flip
        self.multitask = multitask

    def __len__(self) -> int:
        return len(self.cache_indices)

    def __getitem__(self, idx: int) -> dict:
        cidx = int(self.cache_indices[idx])
        pv = torch.from_numpy(np.array(self.cache_arr[cidx]))   # (3, 384, 384) float16
        if self.horizontal_flip and random.random() < 0.5:
            pv = pv.flip(dims=[-1])
        row = self.labels.iloc[idx]
        out = {
            "pixel_values": pv,
            "sidewalk":     torch.tensor(float(row["sidewalk_present"])),
            "crosswalk":    torch.tensor(float(row["crosswalk_present"])),
        }
        if self.multitask:
            out["intersection_count_log1p"] = torch.tensor(
                float(row.get("intersection_count_log1p", float("nan"))))
            out["building_footprint_frac"] = torch.tensor(
                float(row.get("building_footprint_frac", float("nan"))))
        return out


class LoRASigLIPMultitask(nn.Module):
    """SigLIP vision tower (LoRA-adapted) + classification/regression heads."""

    def __init__(self, vision_model: nn.Module, embed_dim: int = EMBED_DIM,
                 multitask: bool = False) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.multitask = multitask
        self.head_sidewalk  = nn.Linear(embed_dim, 1)
        self.head_crosswalk = nn.Linear(embed_dim, 1)
        if multitask:
            self.head_intersection = nn.Linear(embed_dim, 1)
            self.head_building = nn.Linear(embed_dim, 1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.vision_model(pixel_values=pixel_values)
        feat = out.pooler_output
        result = {
            "sidewalk": self.head_sidewalk(feat).squeeze(-1),
            "crosswalk": self.head_crosswalk(feat).squeeze(-1),
            "feat": feat,
        }
        if self.multitask:
            result["intersection_count_log1p"] = self.head_intersection(feat).squeeze(-1)
            result["building_footprint_frac"] = self.head_building(feat).squeeze(-1)
        return result


def build_lora_model(cfg: LoRAConfig, device: torch.device) -> LoRASigLIPMultitask:
    from peft import LoraConfig, get_peft_model

    logging.info("Loading base model %s …", MODEL_ID)
    base = AutoModel.from_pretrained(MODEL_ID)
    vision = base.vision_model

    target_modules = LORA_TARGET_MODULES_FULL if cfg.multitask else LORA_TARGET_MODULES_ATTN
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    vision = get_peft_model(vision, lora_cfg)
    vision.print_trainable_parameters()

    if cfg.grad_checkpoint:
        if hasattr(vision, "enable_input_require_grads"):
            vision.enable_input_require_grads()
        base_vision = vision.get_base_model() if hasattr(vision, "get_base_model") else vision
        base_vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logging.info("Gradient checkpointing enabled (non-reentrant)")

    return LoRASigLIPMultitask(vision, multitask=cfg.multitask).to(device)


def stratified_three_way_split(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns three arrays of cache_idx values for train/val/test (80/10/10)."""
    cidx = df["cache_idx"].to_numpy()
    strat = (df["city"].astype(str) + "_" + df["sidewalk_present"].astype(str)).to_numpy()
    train, temp, _, strat_t = train_test_split(cidx, strat, test_size=0.2, stratify=strat, random_state=seed)
    val,   test = train_test_split(temp, test_size=0.5, stratify=strat_t, random_state=seed)
    return np.array(train), np.array(val), np.array(test)


def cosine_warmup_schedule(optim: torch.optim.Optimizer, total_steps: int, warmup_frac: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optim, lr_lambda=lr_fn)


def compute_pos_weights(df: pd.DataFrame) -> tuple[float, float]:
    sw = df["sidewalk_present"].mean()
    cw = df["crosswalk_present"].mean()
    return (1 - sw) / max(sw, 1e-6), (1 - cw) / max(cw, 1e-6)


@torch.no_grad()
def eval_loop(
    model: LoRASigLIPMultitask,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> dict:
    model.eval()
    sw_logits, cw_logits, sw_y, cw_y = [], [], [], []
    int_logits, int_y, bld_logits, bld_y = [], [], [], []
    for batch in loader:
        px = batch["pixel_values"].to(device, dtype=autocast_dtype, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            preds = model(px)
        sw_logits.append(preds["sidewalk"].float().cpu().numpy())
        cw_logits.append(preds["crosswalk"].float().cpu().numpy())
        sw_y.append(batch["sidewalk"].numpy())
        cw_y.append(batch["crosswalk"].numpy())
        if model.multitask:
            int_logits.append(preds["intersection_count_log1p"].float().cpu().numpy())
            bld_logits.append(preds["building_footprint_frac"].float().cpu().numpy())
            int_y.append(batch["intersection_count_log1p"].numpy())
            bld_y.append(batch["building_footprint_frac"].numpy())
    sw_logits = np.concatenate(sw_logits); cw_logits = np.concatenate(cw_logits)
    sw_y = np.concatenate(sw_y).astype(int); cw_y = np.concatenate(cw_y).astype(int)
    out: dict = {}
    if len(np.unique(sw_y)) > 1:
        out["sidewalk_auroc"] = float(roc_auc_score(sw_y, sw_logits))
        out["sidewalk_ap"]    = float(average_precision_score(sw_y, sw_logits))
    if len(np.unique(cw_y)) > 1:
        out["crosswalk_auroc"] = float(roc_auc_score(cw_y, cw_logits))
        out["crosswalk_ap"]    = float(average_precision_score(cw_y, cw_logits))
    out["mean_auroc"] = float(np.mean([out.get("sidewalk_auroc", 0.0), out.get("crosswalk_auroc", 0.0)]))
    if model.multitask and int_logits:
        int_logits_np = np.concatenate(int_logits)
        int_y_np = np.concatenate(int_y)
        bld_logits_np = np.concatenate(bld_logits)
        bld_y_np = np.concatenate(bld_y)
        valid_int = np.isfinite(int_y_np) & np.isfinite(int_logits_np)
        valid_bld = np.isfinite(bld_y_np) & np.isfinite(bld_logits_np)
        if valid_int.sum() > 10:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(int_logits_np[valid_int], int_y_np[valid_int])
            out["intersection_rho"] = float(rho)
        if valid_bld.sum() > 10:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(bld_logits_np[valid_bld], bld_y_np[valid_bld])
            out["building_frac_rho"] = float(rho)
    return out


def env_fingerprint() -> dict:
    info = {
        "python":       sys.version.split()[0],
        "platform":     platform.platform(),
        "torch":        torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy":        np.__version__,
    }
    try:
        info["peft"] = __import__("peft").__version__
    except Exception:
        info["peft"] = "unknown"
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["cuda_runtime"] = torch.version.cuda
    return info


def train(cfg: LoRAConfig) -> dict:
    set_all_seeds(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for LoRA fine-tuning. Got device=cpu. "
            "Verify that python -c 'import torch; print(torch.cuda.is_available())' returns True."
        )
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    logging.info("Device: %s | autocast dtype: %s | GPU mem free=%.1f/%.1f GB | env: %s",
                 device, autocast_dtype, free / 1e9, total / 1e9, env_fingerprint())

    cache_dir = Path(cfg.cache_dir).expanduser().resolve()
    arr, df, manifest = load_cache(cache_dir)

    if cfg.multitask:
        reg_labels = load_multitask_regression_labels()
        df = df.merge(reg_labels, on=["point_id", "city"], how="left")
        logging.info("Multi-task: added intersection_count_log1p, building_footprint_frac columns")

    if cfg.holdout_city and cfg.holdout_city != "none":
        if cfg.holdout_city not in {"msp", "seattle", "dc", "pittsburgh"}:
            raise ValueError(f"--holdout-city must be one of msp/seattle/dc/pittsburgh/none, got {cfg.holdout_city!r}")
        n_before = len(df)
        df_holdout = df[df["city"] == cfg.holdout_city].copy()
        df = df[df["city"] != cfg.holdout_city].reset_index(drop=True)
        logging.info(
            "Leave-one-city-out: holding out %s (%d rows of %d). Fine-tune sees only %d rows from %s.",
            cfg.holdout_city, len(df_holdout), n_before, len(df),
            sorted(df["city"].unique().tolist()),
        )

    train_idx, val_idx, test_idx = stratified_three_way_split(df, cfg.seed)
    logging.info("Splits — train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    train_df = df[df["cache_idx"].isin(train_idx)]
    pos_sw, pos_cw = compute_pos_weights(train_df)
    pos_sw_t = torch.tensor(pos_sw, device=device)
    pos_cw_t = torch.tensor(pos_cw, device=device)
    logging.info("Pos-weights — sidewalk=%.3f crosswalk=%.3f", pos_sw, pos_cw)

    model = build_lora_model(cfg, device)

    train_ds = CachedTensorDataset(arr, df, train_idx, horizontal_flip=cfg.horizontal_flip, multitask=cfg.multitask)
    val_ds   = CachedTensorDataset(arr, df, val_idx,   horizontal_flip=False, multitask=cfg.multitask)
    test_ds  = CachedTensorDataset(arr, df, test_idx,  horizontal_flip=False, multitask=cfg.multitask)

    common = dict(
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, drop_last=False, **common)

    lora_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "head_" in n else lora_params).append(p)
    logging.info(
        "Trainable: LoRA=%d tensors (%d params), heads=%d tensors (%d params)",
        len(lora_params), sum(p.numel() for p in lora_params),
        len(head_params), sum(p.numel() for p in head_params),
    )

    optim = AdamW([
        {"params": lora_params, "lr": cfg.lr,      "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
    ])
    accum = max(1, cfg.grad_accum_steps)
    total_steps = max(1, (len(train_loader) // accum) * cfg.epochs)
    sched = cosine_warmup_schedule(optim, total_steps, cfg.warmup_frac)
    if accum > 1:
        logging.info("Gradient accumulation: %d micro-batches → effective batch = %d", accum, cfg.batch_size * accum)

    out_dir = Path(cfg.output_dir)
    (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")

    (out_dir / "splits.json").write_text(json.dumps({
        "train_cache_idx": train_idx.tolist(),
        "val_cache_idx":   val_idx.tolist(),
        "test_cache_idx":  test_idx.tolist(),
        "seed": cfg.seed,
    }))

    best_mean_auroc = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", dynamic_ncols=True)):
            px = batch["pixel_values"].to(device, dtype=autocast_dtype, non_blocking=True)
            ys = batch["sidewalk"].to(device, non_blocking=True)
            yc = batch["crosswalk"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                preds = model(px)
                loss_s = F.binary_cross_entropy_with_logits(preds["sidewalk"], ys, pos_weight=pos_sw_t)
                loss_c = F.binary_cross_entropy_with_logits(preds["crosswalk"], yc, pos_weight=pos_cw_t)
                loss = loss_s + loss_c
                if cfg.multitask:
                    y_int = batch["intersection_count_log1p"].to(device, non_blocking=True)
                    y_bld = batch["building_footprint_frac"].to(device, non_blocking=True)
                    mask_int = torch.isfinite(y_int)
                    mask_bld = torch.isfinite(y_bld)
                    if mask_int.sum() >= 2:
                        loss = loss + 0.5 * F.mse_loss(preds["intersection_count_log1p"][mask_int], y_int[mask_int])
                    if mask_bld.sum() >= 2:
                        loss = loss + 0.5 * F.mse_loss(preds["building_footprint_frac"][mask_bld], y_bld[mask_bld])
                loss = loss / accum
            loss.backward()
            running += loss.item() * accum
            n_seen  += px.size(0)

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % 50 == 0:
                with log_path.open("a") as f:
                    f.write(json.dumps({
                        "epoch": epoch, "step": step,
                        "loss": float(loss.item() * accum),
                        "loss_sidewalk": float(loss_s.item()),
                        "loss_crosswalk": float(loss_c.item()),
                        "lr_lora": optim.param_groups[0]["lr"],
                        "lr_head": optim.param_groups[1]["lr"],
                        "samples_per_sec": n_seen / max(time.time() - t0, 1e-6),
                    }) + "\n")

        train_throughput = n_seen / max(time.time() - t0, 1e-6)
        val_metrics = eval_loop(model, val_loader, device, autocast_dtype)
        logging.info(
            "Epoch %d | train_loss=%.4f | %.1f samp/sec | val %s | %.1fs",
            epoch + 1, running / max(1, len(train_loader)), train_throughput,
            val_metrics, time.time() - t0,
        )
        with log_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "phase": "val",
                                "train_throughput_samp_per_sec": train_throughput,
                                **val_metrics}) + "\n")

        if val_metrics["mean_auroc"] > best_mean_auroc:
            best_mean_auroc = val_metrics["mean_auroc"]
            best_epoch = epoch
            no_improve = 0
            model.vision_model.save_pretrained(out_dir / "adapter")
            heads_dict = {
                "head_sidewalk":  model.head_sidewalk.state_dict(),
                "head_crosswalk": model.head_crosswalk.state_dict(),
            }
            if cfg.multitask:
                heads_dict["head_intersection"] = model.head_intersection.state_dict()
                heads_dict["head_building"] = model.head_building.state_dict()
            torch.save(heads_dict, out_dir / "classification_heads.pt")
            logging.info("  ↳ new best val mean_auroc=%.4f — saved adapter+heads", best_mean_auroc)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logging.info("Early stopping at epoch %d (patience=%d)", epoch + 1, cfg.patience)
                break

    logging.info("Reloading best adapter (epoch %d) for test eval", best_epoch + 1)
    from peft import PeftModel
    base = AutoModel.from_pretrained(MODEL_ID).vision_model
    model.vision_model = PeftModel.from_pretrained(base, str(out_dir / "adapter")).to(device)
    heads = torch.load(out_dir / "classification_heads.pt", map_location=device)
    model.head_sidewalk.load_state_dict(heads["head_sidewalk"])
    model.head_crosswalk.load_state_dict(heads["head_crosswalk"])
    if cfg.multitask and "head_intersection" in heads:
        model.head_intersection.load_state_dict(heads["head_intersection"])
        model.head_building.load_state_dict(heads["head_building"])

    test_metrics = eval_loop(model, test_loader, device, autocast_dtype)
    logging.info("Test metrics: %s", test_metrics)

    summary = {
        "best_epoch": best_epoch + 1,
        "best_val_mean_auroc": best_mean_auroc,
        "test": test_metrics,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "cache_manifest": manifest,
        "config": asdict(cfg),
        "env": env_fingerprint(),
    }
    (out_dir / "val_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps({**asdict(cfg), "env": env_fingerprint()}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune SigLIP-SO400M on heading-level Overture labels (cached)")
    p.add_argument("--cache-dir", default=str(Path.home() / "walkclip_cache"),
                   help="Directory containing siglip_pixels_street_384.npy + index. "
                        "Build via cache_pixel_values.py.")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=64,
                   help="With cached tensors on a 5070 Ti / A100, 64 fits easily; "
                        "raise on larger GPUs.")
    p.add_argument("--lr",          type=float, default=1e-4, help="LR on LoRA adapters")
    p.add_argument("--head-lr",     type=float, default=1e-3, help="LR on classification heads")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lora-r",      type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience",    type=int,   default=3)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=SEED)
    p.add_argument("--no-flip",     action="store_true",
                   help="Disable random horizontal flip augmentation (default: enabled).")
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Disable gradient checkpointing (saves ~30%% compute, costs ~2x activation memory).")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches before optimizer.step(). "
                        "Effective batch = batch_size * grad_accum_steps.")
    p.add_argument("--output", default=str(REPO_ROOT / "project/artifacts/models/siglip_lora_v1"))
    p.add_argument("--holdout-city", choices=["none", "msp", "seattle", "dc", "pittsburgh"],
                   default="none",
                   help="Leave-one-city-out: exclude this city from the LoRA fine-tune entirely. "
                        "Use this to produce a backbone whose cross-city evaluation on the held-out "
                        "city is honestly held-out at backbone level. Default 'none' = all cities. "
                        "'pittsburgh' trains on MSP+Seattle+DC only (Pittsburgh is never in the "
                        "pixel cache, so this is structurally equivalent to the 3-city training set).")
    p.add_argument("--multitask", action="store_true",
                   help="Train 4 heads (sidewalk, crosswalk, intersection_count_log1p, "
                        "building_footprint_frac) instead of 2. Also adds fc1/fc2 to LoRA targets.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = LoRAConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        warmup_frac=args.warmup_frac, patience=args.patience,
        num_workers=args.num_workers, seed=args.seed,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        horizontal_flip=not args.no_flip,
        grad_clip=args.grad_clip,
        grad_checkpoint=not args.no_grad_checkpoint,
        grad_accum_steps=args.grad_accum_steps,
        holdout_city=args.holdout_city,
        multitask=args.multitask,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
