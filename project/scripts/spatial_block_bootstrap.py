"""
Phase 3 of the WalkBench reality-check: spatial-block bootstrap CIs + paired
equivalence tests over every reality-check prediction bundle.

Answers the Corley 2026 spatial-autocorrelation flag: instead of resampling test
*points* (which are spatially correlated, so a naive bootstrap under-estimates the
CI), we snap each test point to a ~1 km grid cell and resample *cells* with
replacement (Karasiak 2022). The naive point bootstrap is kept as a cross-check and
the CI-width ratio quantifies how much autocorrelation inflates the naive interval.

For every complex rung we run a *paired* spatial-block bootstrap against the linear
probe of the SAME encoder (the encoder-confound rule from REALITY_CHECK_PLAN.md S3):
  - siglip2 (frozen MLP)        vs siglip2_linprobe
  - siglip_lora / _lora_loco_v2 vs siglip_linprobe
  - siglip_coral / _subspace    vs siglip_linprobe
Each Delta = metric(complex) - metric(linprobe) gets a spatial-block CI, a paired
DeLong p (binary), Holm-Bonferroni correction across the whole (method x target x
direction) family, and a pre-registered equivalence verdict at epsilon = 0.02.
"Not better" is reported as a positive equivalence result, not merely p > 0.05.

Inputs  : project/artifacts/predictions/*.npz  (native rungs from run_reality_check.py
          + MLP/LoRA rungs from train_multitarget.py --save-predictions)
          project/data/processed/labels/features_labels_agreement.csv  (lat/lon)
Outputs : project/artifacts/reports/reality_check_summary.json   (headline, per S8)
          project/artifacts/reports/reality_check_cis.jsonl       (per rung x dir x target)
          project/artifacts/reports/reality_check_paired.jsonl    (per paired test)

LOCO (6 directions among MSP/Seattle/DC) and Pittsburgh zero-shot are reported as
separate blocks. Run AFTER the GPU arm (run_reality_check_gpu.sh) finishes, so the
siglip2-MLP and LoRA-LOCO Pittsburgh rungs are present.

Run (Windows venv, CPU; repo ROOT), ~10 min at the default n_boot=1000:
    & ".venv\\Scripts\\python.exe" project\\scripts\\spatial_block_bootstrap.py
    & ".venv\\Scripts\\python.exe" project\\scripts\\spatial_block_bootstrap.py --n-boot 200   # quick
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the vetted metric + parsing helpers from the existing bootstrap harness so
# AUROC / Spearman / run-id parsing stay identical across the two tools.
from bootstrap_cis import auroc, spearman, parse_run_id, paired_delong

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "project/artifacts/predictions"
REPORT_DIR = REPO_ROOT / "project/artifacts/reports"
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"

EPS = 0.02                       # pre-registered equivalence margin (AUROC / rho)
N_BOOT = 1000
CELL_M = 1000.0                  # ~1 km spatial blocks
SEEDS = tuple(range(41, 51))     # the 10 reality-check seeds
SPATIAL_RNG = 20260628
NAIVE_RNG = 20260629


@dataclass(frozen=True)
class Target:
    name: str
    kind: str          # "binary" | "regression"
    metric: str        # "auroc" | "spearman_rho"


TARGETS: tuple[Target, ...] = (
    Target("osm_marked_crossing_present", "binary", "auroc"),
    Target("overture_sidewalk_present", "binary", "auroc"),
    Target("overture_intersection_count_200m", "regression", "spearman_rho"),
    Target("overture_building_footprint_frac_100m", "regression", "spearman_rho"),
)
TARGET_BY_NAME = {t.name: t for t in TARGETS}


@dataclass(frozen=True)
class Rung:
    virtual: str       # stable tag used in outputs
    encoder: str       # siglip | siglip2 | dinov2  (for the encoder-confound rule)
    rung: str          # linprobe | mlp | lora | lora_loco | coral | subspace
    kind: str          # "baseline" | "complex"
    baseline: str | None   # virtual tag of the same-encoder linear probe (complex only)


def classify(backbone: str) -> Rung | None:
    """Map a raw run-id backbone tag to its ladder rung, or None if not a rung."""
    if backbone.startswith("siglip_lora_loco_v2_"):
        return Rung("siglip_lora_loco_v2", "siglip", "lora_loco", "complex", "siglip_linprobe")
    return {
        "siglip_lora":      Rung("siglip_lora", "siglip", "lora", "complex", "siglip_linprobe"),
        "siglip2":          Rung("siglip2_mlp", "siglip2", "mlp", "complex", "siglip2_linprobe"),
        "siglip_coral":     Rung("siglip_coral", "siglip", "coral", "complex", "siglip_linprobe"),
        "siglip_subspace":  Rung("siglip_subspace", "siglip", "subspace", "complex", "siglip_linprobe"),
        "siglip_linprobe":  Rung("siglip_linprobe", "siglip", "linprobe", "baseline", None),
        "siglip2_linprobe": Rung("siglip2_linprobe", "siglip2", "linprobe", "baseline", None),
        "dinov2_linprobe":  Rung("dinov2_linprobe", "dinov2", "linprobe", "baseline", None),
    }.get(backbone)


LOCO_CITIES = {"msp", "seattle", "dc"}
PGH_SRC = "dc+msp+seattle"


def direction_block(src: str, dst: str) -> str | None:
    """'loco' for single-city LOCO among the trio, 'pittsburgh' for the zero-shot, else None."""
    if dst == "pittsburgh" and src == PGH_SRC:
        return "pittsburgh"
    if src in LOCO_CITIES and dst in LOCO_CITIES and src != dst:
        return "loco"
    return None


def metric_fn_for(target: Target):
    return auroc if target.kind == "binary" else spearman


def to_score(target: Target, logit: np.ndarray) -> np.ndarray:
    """AUROC/Spearman are rank-based, so a monotonic map is harmless; expit keeps
    binary scores in [0, 1] for DeLong. Regression predictions pass through."""
    if target.kind == "binary":
        return 1.0 / (1.0 + np.exp(-logit))
    return logit


# --------------------------------------------------------------------------- coords / cells

def load_coords() -> dict[str, dict[int, tuple[float, float]]]:
    """(city -> {point_id -> (lat, lon)}) from the master file."""
    df = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", "lat", "lon"], low_memory=False)
    df = df.dropna(subset=["lat", "lon"]).drop_duplicates(["city", "point_id"])
    df["point_id"] = df["point_id"].astype(int)
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for city, g in df.groupby("city"):
        out[city] = dict(zip(g["point_id"], zip(g["lat"], g["lon"])))
    return out


def cell_codes(city: str, point_ids: np.ndarray, coords: dict, cell_m: float = CELL_M) -> np.ndarray:
    """Integer ~1 km cell code per point. Local equirectangular meters about the
    city centroid (true metres to <0.1% over a city-sized extent) snapped to cell_m.
    Codes are namespaced per city so pooled blocks never merge cells across cities.
    Points with no coordinate get a unique singleton cell (never co-resampled)."""
    city_coords = coords.get(city, {})
    lats = np.array([city_coords.get(int(p), (np.nan, np.nan))[0] for p in point_ids])
    lons = np.array([city_coords.get(int(p), (np.nan, np.nan))[1] for p in point_ids])
    finite = np.isfinite(lats) & np.isfinite(lons)
    lat0 = float(np.nanmean(lats)) if finite.any() else 0.0
    lon0 = float(np.nanmean(lons)) if finite.any() else 0.0
    x = (lons - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lats - lat0) * 110_540.0
    cx = np.floor(np.where(finite, x, 0.0) / cell_m).astype(np.int64)
    cy = np.floor(np.where(finite, y, 0.0) / cell_m).astype(np.int64)
    city_offset = (hash(city) & 0xFFFF) * 10_000_000
    codes = city_offset + (cx + 5000) * 20_000 + (cy + 5000)
    # Unique singleton code for coordinate-less points so they cannot anchor a block.
    codes = codes.astype(np.int64)
    for i in np.where(~finite)[0]:
        codes[i] = -(int(i) + 1)
    return codes


def _cell_groups(cells: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(cells, kind="mergesort")
    cs = cells[order]
    groups, start = [], 0
    for i in range(1, len(cs) + 1):
        if i == len(cs) or cs[i] != cs[start]:
            groups.append(order[start:i])
            start = i
    return groups


# --------------------------------------------------------------------------- bootstraps

def _pct_ci(boots: np.ndarray) -> tuple[float, float, float]:
    boots = boots[np.isfinite(boots)]
    if boots.size < max(50, len(boots) // 10):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), float(boots.std(ddof=1))


def spatial_ci(y, score, cells, metric_fn, n_boot, rng) -> dict:
    valid = np.isfinite(y) & np.isfinite(score)
    y, score, cells = y[valid], score[valid], cells[valid]
    if len(y) < 30:
        return {"point": float("nan"), "n": int(len(y))}
    groups = _cell_groups(cells)
    n_cells = len(groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        idx = np.concatenate([groups[i] for i in pick])
        boots[b] = metric_fn(y[idx], score[idx])
    lo, hi, sd = _pct_ci(boots)
    return {"point": float(metric_fn(y, score)), "ci_lo": lo, "ci_hi": hi,
            "boot_std": sd, "n": int(len(y)), "n_cells": int(n_cells)}


def naive_ci(y, score, metric_fn, n_boot, rng) -> dict:
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = y[valid], score[valid]
    if len(y) < 30:
        return {"point": float("nan"), "n": int(len(y))}
    n = len(y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = metric_fn(y[idx], score[idx])
    lo, hi, sd = _pct_ci(boots)
    return {"point": float(metric_fn(y, score)), "ci_lo": lo, "ci_hi": hi, "boot_std": sd, "n": int(n)}


def paired_spatial_delta(y, s_complex, s_base, cells, metric_fn, n_boot, rng) -> dict:
    valid = np.isfinite(y) & np.isfinite(s_complex) & np.isfinite(s_base)
    y, s_complex, s_base, cells = y[valid], s_complex[valid], s_base[valid], cells[valid]
    if len(y) < 30:
        return {"delta": float("nan"), "n": int(len(y))}
    groups = _cell_groups(cells)
    n_cells = len(groups)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        idx = np.concatenate([groups[i] for i in pick])
        deltas[b] = metric_fn(y[idx], s_complex[idx]) - metric_fn(y[idx], s_base[idx])
    lo, hi, sd = _pct_ci(deltas)
    deltas_f = deltas[np.isfinite(deltas)]
    p = float(2 * min((deltas_f >= 0).mean(), (deltas_f <= 0).mean())) if deltas_f.size else float("nan")
    return {"delta": float(metric_fn(y, s_complex) - metric_fn(y, s_base)),
            "ci_lo": lo, "ci_hi": hi, "boot_std": sd, "boot_p": p,
            "n": int(len(y)), "n_cells": int(n_cells)}


def verdict(ci_lo: float, ci_hi: float, eps: float = EPS) -> str:
    """Equivalence verdict for Delta = complex - linprobe at margin eps."""
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "inconclusive"
    if ci_lo > eps:
        return "better"          # complex beats baseline by >= eps (reportable exception)
    if ci_hi < -eps:
        return "worse"
    if ci_lo > -eps and ci_hi < eps:
        return "equivalent"
    if ci_hi < eps:
        return "not_better"      # CI may dip below -eps but never clears +eps
    return "inconclusive"


def holm_bonferroni(pvals: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    adj = [float("nan")] * len(pvals)
    if not finite:
        return adj
    finite.sort(key=lambda t: t[1])
    m = len(finite)
    running = 0.0
    for rank, (i, p) in enumerate(finite):
        running = max(running, (m - rank) * p)
        adj[i] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- assembly

@dataclass
class Group:
    """All seeds of one (virtual rung, src, dst); per target a seed-averaged score."""
    rung: Rung
    src: str
    dst: str
    block: str
    point_ids: np.ndarray | None = None
    sum_score: dict[str, np.ndarray] = field(default_factory=dict)
    seed_count: dict[str, int] = field(default_factory=dict)
    y: dict[str, np.ndarray] = field(default_factory=dict)
    per_seed_metric: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def mean_score(self, target: str) -> np.ndarray:
        return self.sum_score[target] / max(self.seed_count.get(target, 0), 1)


def collect_groups() -> dict[tuple[str, str, str], Group]:
    groups: dict[tuple[str, str, str], Group] = {}
    n_files = 0
    for path in sorted(PRED_DIR.glob("*.npz")):
        meta = parse_run_id(path.stem)
        if meta is None or meta["tier"] != 1 or meta["ablation"] != "full":
            continue
        if meta["seed"] not in SEEDS:
            continue
        rung = classify(meta["backbone"])
        if rung is None:
            continue
        block = direction_block(meta["src"], meta["dst"])
        if block is None:
            continue
        key = (rung.virtual, meta["src"], meta["dst"])
        grp = groups.get(key)
        if grp is None:
            grp = groups[key] = Group(rung, meta["src"], meta["dst"], block)
        data = np.load(path)
        ids = data["point_ids"].astype(np.int64)
        if grp.point_ids is None:
            grp.point_ids = ids
        elif not np.array_equal(grp.point_ids, ids):
            # Seeds of one rung+direction share a test set by construction; if a stray
            # ordering differs, realign onto the first-seen ids.
            pos = {int(p): i for i, p in enumerate(ids)}
            sel = np.array([pos.get(int(p), -1) for p in grp.point_ids])
            if (sel < 0).any():
                logging.warning("Seed point-set mismatch in %s; skipping non-overlapping points.", path.name)
        for target in TARGETS:
            lk, yk = f"logit__{target.name}", f"y__{target.name}"
            if lk not in data.files or yk not in data.files:
                continue
            score = to_score(target, data[lk].astype(np.float64))
            y = data[yk].astype(np.float64)
            if not np.array_equal(ids, grp.point_ids):
                pos = {int(p): i for i, p in enumerate(ids)}
                sel = np.array([pos.get(int(p), -1) for p in grp.point_ids])
                ok = sel >= 0
                full_score = np.full(len(grp.point_ids), np.nan)
                full_y = np.full(len(grp.point_ids), np.nan)
                full_score[ok] = score[sel[ok]]
                full_y[ok] = y[sel[ok]]
                score, y = full_score, full_y
            grp.sum_score.setdefault(target.name, np.zeros(len(grp.point_ids)))
            grp.sum_score[target.name] = grp.sum_score[target.name] + np.nan_to_num(score, nan=0.0)
            grp.seed_count[target.name] = grp.seed_count.get(target.name, 0) + 1
            grp.y[target.name] = y
            grp.per_seed_metric[target.name].append(metric_fn_for(target)(y, score))
        n_files += 1
    logging.info("Consumed %d reality-check npz into %d (rung,direction) groups.", n_files, len(groups))
    return groups


def aligned_scores(grp: Group, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(point_ids, seed-mean score, y) for a target, only where this group has it."""
    ids = grp.point_ids
    score = grp.mean_score(target)
    y = grp.y[target]
    return ids, score, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 spatial-block bootstrap + equivalence.")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--cell-m", type=float, default=CELL_M)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coords = load_coords()
    groups = collect_groups()
    if not groups:
        logging.error("No reality-check prediction bundles found in %s.", PRED_DIR)
        return 2

    by_block_dir_rung: dict[tuple[str, str], dict[str, Group]] = defaultdict(dict)
    for grp in groups.values():
        by_block_dir_rung[(grp.block, f"{grp.src}->{grp.dst}")][grp.rung.virtual] = grp

    rng_sp = np.random.default_rng(SPATIAL_RNG)
    rng_na = np.random.default_rng(NAIVE_RNG)

    ci_rows: list[dict] = []
    paired_rows: list[dict] = []

    for (block, direction), rungs_here in sorted(by_block_dir_rung.items()):
        dst = direction.split("->")[1]
        for vtag, grp in sorted(rungs_here.items()):
            for target in TARGETS:
                if grp.seed_count.get(target.name, 0) == 0:
                    continue
                ids, score, y = aligned_scores(grp, target.name)
                cells = cell_codes(dst, ids, coords, args.cell_m)
                mfn = metric_fn_for(target)
                sp = spatial_ci(y, score, cells, mfn, args.n_boot, rng_sp)
                na = naive_ci(y, score, mfn, args.n_boot, rng_na)
                seedvals = np.array(grp.per_seed_metric[target.name], dtype=float)
                seedvals = seedvals[np.isfinite(seedvals)]
                width_sp = sp.get("ci_hi", np.nan) - sp.get("ci_lo", np.nan)
                width_na = na.get("ci_hi", np.nan) - na.get("ci_lo", np.nan)
                ci_rows.append({
                    "block": block, "direction": direction, "rung": vtag,
                    "encoder": grp.rung.encoder, "rung_kind": grp.rung.kind,
                    "target": target.name, "metric": target.metric,
                    "point": sp.get("point"),
                    "seed_mean": float(seedvals.mean()) if seedvals.size else float("nan"),
                    "seed_std": float(seedvals.std(ddof=1)) if seedvals.size > 1 else 0.0,
                    "n_seeds": int(seedvals.size), "n": sp.get("n"), "n_cells": sp.get("n_cells"),
                    "spatial_ci_lo": sp.get("ci_lo"), "spatial_ci_hi": sp.get("ci_hi"),
                    "naive_ci_lo": na.get("ci_lo"), "naive_ci_hi": na.get("ci_hi"),
                    "ci_width_ratio": float(width_sp / width_na) if np.isfinite(width_sp) and np.isfinite(width_na) and width_na > 0 else float("nan"),
                })

            if grp.rung.kind != "complex":
                continue
            base = rungs_here.get(grp.rung.baseline)
            if base is None:
                logging.info("No baseline %s for %s in %s; skipping paired test.",
                             grp.rung.baseline, vtag, direction)
                continue
            for target in TARGETS:
                if grp.seed_count.get(target.name, 0) == 0 or base.seed_count.get(target.name, 0) == 0:
                    continue
                ids_c, sc_c, y_c = aligned_scores(grp, target.name)
                ids_b, sc_b, y_b = aligned_scores(base, target.name)
                pos_b = {int(p): i for i, p in enumerate(ids_b)}
                common = np.array([i for i, p in enumerate(ids_c) if int(p) in pos_b])
                if common.size < 30:
                    continue
                sel_b = np.array([pos_b[int(ids_c[i])] for i in common])
                y = y_c[common]
                sc_complex = sc_c[common]
                sc_baseline = sc_b[sel_b]
                cells = cell_codes(dst, ids_c[common], coords, args.cell_m)
                mfn = metric_fn_for(target)
                pr = paired_spatial_delta(y, sc_complex, sc_baseline, cells, mfn, args.n_boot, rng_sp)
                delong = {}
                if target.kind == "binary":
                    m = np.isfinite(y) & np.isfinite(sc_complex) & np.isfinite(sc_baseline)
                    if m.sum() >= 30 and len(np.unique(y[m])) == 2:
                        delong = paired_delong(y[m], sc_complex[m], sc_baseline[m])
                primary_p = delong.get("p_value") if target.kind == "binary" else pr.get("boot_p")
                paired_rows.append({
                    "block": block, "direction": direction,
                    "rung": vtag, "encoder": grp.rung.encoder, "rung_name": grp.rung.rung,
                    "baseline": grp.rung.baseline, "target": target.name, "metric": target.metric,
                    "delta": pr.get("delta"), "delta_ci_lo": pr.get("ci_lo"), "delta_ci_hi": pr.get("ci_hi"),
                    "boot_p": pr.get("boot_p"), "delong_p": delong.get("p_value"),
                    "primary_p": primary_p, "n_common": pr.get("n"),
                    "verdict": verdict(pr.get("ci_lo", float("nan")), pr.get("ci_hi", float("nan")), args.eps),
                })

    # Holm-Bonferroni across the entire (method x target x direction) paired family.
    holm = holm_bonferroni([r["primary_p"] for r in paired_rows])
    for r, h in zip(paired_rows, holm):
        r["holm_p"] = h

    summary = build_summary(ci_rows, paired_rows, args)
    (REPORT_DIR / "reality_check_cis.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ci_rows), encoding="utf-8")
    (REPORT_DIR / "reality_check_paired.jsonl").write_text(
        "\n".join(json.dumps(r) for r in paired_rows), encoding="utf-8")
    (REPORT_DIR / "reality_check_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote reality_check_summary.json (%d CI rows, %d paired rows).",
                 len(ci_rows), len(paired_rows))
    _log_headline(summary)
    return 0


def build_summary(ci_rows: list[dict], paired_rows: list[dict], args) -> dict:
    blocks: dict[str, dict] = {}
    for block in ("loco", "pittsburgh"):
        rungs: dict[str, dict] = {}
        block_ci = [r for r in ci_rows if r["block"] == block]
        block_pr = [r for r in paired_rows if r["block"] == block]
        for vtag in sorted({r["rung"] for r in block_ci}):
            rung_ci = [r for r in block_ci if r["rung"] == vtag]
            tgt_summary: dict[str, dict] = {}
            for target in TARGETS:
                trows = [r for r in rung_ci if r["target"] == target.name]
                if not trows:
                    continue
                # Headline point = per-seed mean (the Corley/prior-table convention);
                # the ensemble point (metric on seed-averaged scores) is what the spatial
                # CI brackets and is kept per-direction for the figure.
                seedwise = np.array([r["seed_mean"] for r in trows
                                     if r["seed_mean"] is not None and np.isfinite(r["seed_mean"])])
                ens = np.array([r["point"] for r in trows if r["point"] is not None and np.isfinite(r["point"])])
                ratios = np.array([r["ci_width_ratio"] for r in trows
                                   if np.isfinite(r.get("ci_width_ratio", np.nan))])
                entry = {
                    "metric": target.metric,
                    "n_directions": int(len(trows)),
                    "mean": float(seedwise.mean()) if seedwise.size else float("nan"),
                    "std": float(seedwise.std(ddof=1)) if seedwise.size > 1 else 0.0,
                    "best": float(seedwise.max()) if seedwise.size else float("nan"),
                    "mean_ensemble": float(ens.mean()) if ens.size else float("nan"),
                    "mean_ci_width_ratio": float(ratios.mean()) if ratios.size else float("nan"),
                    "per_direction": {r["direction"]: {
                        "seed_mean": r["seed_mean"], "seed_std": r["seed_std"], "n_seeds": r["n_seeds"],
                        "point_ensemble": r["point"],
                        "spatial_ci": [r["spatial_ci_lo"], r["spatial_ci_hi"]],
                        "naive_ci": [r["naive_ci_lo"], r["naive_ci_hi"]],
                        "ci_width_ratio": r["ci_width_ratio"], "n": r["n"], "n_cells": r["n_cells"],
                    } for r in trows},
                }
                prows = [r for r in block_pr if r["rung"] == vtag and r["target"] == target.name]
                if prows:
                    deltas = np.array([r["delta"] for r in prows if r["delta"] is not None and np.isfinite(r["delta"])])
                    counts: dict[str, int] = defaultdict(int)
                    for r in prows:
                        counts[r["verdict"]] += 1
                    holm_sig = [r for r in prows if np.isfinite(r.get("holm_p", np.nan)) and r["holm_p"] < 0.05
                                and (r["delta"] or 0) > 0]
                    entry["paired_vs_baseline"] = {
                        "baseline": prows[0]["baseline"],
                        "delta_mean": float(deltas.mean()) if deltas.size else float("nan"),
                        "delta_min": float(deltas.min()) if deltas.size else float("nan"),
                        "delta_max": float(deltas.max()) if deltas.size else float("nan"),
                        "verdict_counts": dict(counts),
                        "overall_verdict": _overall_verdict(counts),
                        "n_holm_sig_better": int(len(holm_sig)),
                        "per_direction": {r["direction"]: {
                            "delta": r["delta"], "delta_ci": [r["delta_ci_lo"], r["delta_ci_hi"]],
                            "delong_p": r["delong_p"], "boot_p": r["boot_p"], "holm_p": r.get("holm_p"),
                            "verdict": r["verdict"],
                        } for r in prows},
                    }
                tgt_summary[target.name] = entry
            enc = rung_ci[0]["encoder"]
            kind = rung_ci[0]["rung_kind"]
            rungs[vtag] = {"encoder": enc, "kind": kind, "targets": tgt_summary}
        blocks[block] = {"rungs": rungs}
    return {
        "meta": {
            "eps": args.eps, "n_boot": args.n_boot, "cell_size_m": args.cell_m,
            "seeds": list(SEEDS), "n_targets": len(TARGETS),
            "spatial_bootstrap": "resample ~1km cells with replacement (Karasiak 2022)",
            "point_convention": "rung 'mean' = mean over directions of the per-seed-mean metric "
                                "(Corley convention); 'mean_ensemble' and the spatial/naive CIs are "
                                "computed on the seed-averaged score vector. Paired Delta = complex - "
                                "same-encoder linear probe, on the seed-averaged scores over shared points.",
            "claim": "complex cross-city adaptation does not beat frozen + linear probe; "
                     "equivalence at |delta| < eps, paired vs the same-encoder linear probe.",
        },
        "blocks": blocks,
    }


def _overall_verdict(counts: dict[str, int]) -> str:
    if counts.get("better", 0) > 0:
        return f"mixed: {counts['better']} direction(s) beat baseline (>+eps)"
    n_not_better = counts.get("not_better", 0) + counts.get("equivalent", 0) + counts.get("worse", 0)
    if counts.get("inconclusive", 0) == 0 and n_not_better > 0:
        return "not_better"     # no direction clears +eps; supports the reality-check claim
    if n_not_better > 0:
        return "not_better_with_inconclusive"
    return "inconclusive"


def _log_headline(summary: dict) -> None:
    for block, bdata in summary["blocks"].items():
        logging.info("---- %s ----", block.upper())
        for vtag, rdata in bdata["rungs"].items():
            cells = []
            for t in TARGETS:
                e = rdata["targets"].get(t.name)
                if not e:
                    continue
                tag = t.name.split("_")[1][:5]
                v = e.get("paired_vs_baseline", {}).get("overall_verdict", rdata["kind"])
                cells.append(f"{tag}={e['mean']:.3f}({v.split(':')[0] if isinstance(v,str) else v})")
            logging.info("  %-22s %s", vtag, "  ".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
