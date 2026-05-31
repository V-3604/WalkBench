"""
Bootstrap confidence intervals + paired DeLong tests on saved test predictions.

Reads .npz files written by train_multitarget.py --save-predictions and
computes:

  1. Per-(backbone, direction, ablation, target, seed) percentile bootstrap
     95% CI for AUROC (binary targets) and Spearman ρ (regression targets).
     n_boot=1000, resampling test points with replacement, fixed seed=12345.

  2. Cross-seed mean ± std AUROC where multiple seeds exist for the same
     (backbone, direction, ablation, target). Reports a downstream-MLP
     variance estimate.

  3. Paired bootstrap test for ΔAUROC between two backbones evaluated on the
     same (direction, ablation, target). Pairs predictions by point_id
     (the LOCO and frozen runs have identical test_point_ids per direction
     by construction). Reports two-sided p-value (fraction of bootstraps
     where the sign of Δ flips).

  4. Paired DeLong test (Sun & Xu 2014, fast O(N log N) implementation) for
     AUROC differences. Used as the canonical statistical test for binary
     targets; bootstrap is the cross-check.

Outputs (project/artifacts/reports/):
  - bootstrap_cis.jsonl           one record per (run_id, target)
  - paired_tests.jsonl            one record per (direction, ablation, target,
                                   backbone_a, backbone_b)
  - bootstrap_summary.json        per-(backbone, direction, target) seed-aggregated
                                   mean ± std + bootstrap 95 % CI

Run:
    python project/scripts/bootstrap_cis.py
    python project/scripts/bootstrap_cis.py --pairs siglip,siglip_lora_loco \\
        --pairs siglip,clip --pairs siglip_lora,siglip_lora_loco
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR  = REPO_ROOT / "project/artifacts/predictions"
REPORT    = REPO_ROOT / "project/artifacts/reports"

BINARY_TARGETS = {"overture_sidewalk_present", "overture_crosswalk_present"}
REGRESSION_TARGETS = {
    "overture_intersection_count_200m",
    "overture_building_footprint_frac_100m",
    "walkability_index_v1",
    "walkability_index_v2",
}

# [a-z_0-9+] includes literal '+' so siglip2+remoteclip filenames parse correctly.
# src allows '+'-joined multi-city tokens (e.g. dc+msp+seattle for Pittsburgh runs).
RUN_RE = re.compile(
    r"^(?P<backbone>[a-z_0-9+]+?)__(?P<eval_mode>cross_city|in_city)__"
    r"(?P<src>[a-z+]+)_to_(?P<dst>[a-z]+)__"
    r"tier(?P<tier>[0-9]+)__(?P<ablation>[a-z_]+)__seed(?P<seed>[0-9]+)$"
)


def parse_run_id(run_id: str) -> dict | None:
    m = RUN_RE.match(run_id)
    if m is None:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    d["tier"] = int(d["tier"])
    # Collapse leave-one-city-out adapter tag to a single virtual backbone.
    if d["backbone"].startswith("siglip_lora_loco_v2_"):
        d["backbone_virtual"] = "siglip_lora_loco_v2"
    elif d["backbone"].startswith("siglip_lora_loco_"):
        d["backbone_virtual"] = "siglip_lora_loco"
    else:
        d["backbone_virtual"] = d["backbone"]
    return d


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    y_v, s_v = y[valid], score[valid]
    if valid.sum() < 2 or len(np.unique(y_v)) < 2:
        return float("nan")
    npos = int((y_v == 1).sum())
    nneg = len(y_v) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    # Wilcoxon-Mann-Whitney estimator: same result as sklearn, O(n log n),
    # no sklearn Array-API dispatch overhead inside the bootstrap loop.
    ranks = rankdata(s_v)
    return float((ranks[y_v == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    if valid.sum() < 2:
        return float("nan")
    rho, _ = spearmanr(score[valid], y[valid])
    return float(rho)


def bootstrap_metric(
    y: np.ndarray,
    score: np.ndarray,
    metric_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    valid = np.isfinite(y) & np.isfinite(score)
    y_v = y[valid]
    s_v = score[valid]
    n = len(y_v)
    if n < 30:
        return float("nan"), float("nan"), float("nan"), float("nan")
    point = metric_fn(y_v, s_v)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = metric_fn(y_v[idx], s_v[idx])
        except Exception:
            boots[b] = np.nan
    boots = boots[np.isfinite(boots)]
    if boots.size < 100:
        return point, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), float(boots.std(ddof=1))


def fast_delong_var(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Sun-Xu 2014 fast DeLong covariance for paired AUROC tests.
    Returns (auroc, structural variance)."""
    y = y.astype(int)
    pos = scores[y == 1]
    neg = scores[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    a = roc_auc_score(y, scores)
    return float(a), float(a * (1 - a) / min(m, n))  # crude fallback variance


def paired_delong(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> dict:
    """Paired DeLong (Sun-Xu) for AUROC_a - AUROC_b on the same labels.
    Implementation follows the public DeLong fast variant."""
    y = y.astype(int)
    pos_mask = y == 1
    m = int(pos_mask.sum())
    n = int((~pos_mask).sum())
    if m == 0 or n == 0:
        return {"auroc_a": float("nan"), "auroc_b": float("nan"),
                "delta": float("nan"), "p_value": float("nan")}

    def midrank(x):
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        ranks = np.empty(len(x), dtype=np.float64)
        i = 0
        while i < len(x_sorted):
            j = i
            while j < len(x_sorted) and x_sorted[j] == x_sorted[i]:
                j += 1
            ranks[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty_like(ranks)
        out[order] = ranks
        return out

    def fast_components(scores):
        pos = scores[pos_mask]
        neg = scores[~pos_mask]
        tx = midrank(pos)
        ty = midrank(neg)
        tz = midrank(np.concatenate([pos, neg]))
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        auroc = float(((tz[:m] - tx).sum() / m + 1) / n - 1.0 / (2 * n)
                      ) if False else float(roc_auc_score(y, scores))
        return v01, v10, auroc

    v01_a, v10_a, auroc_a = fast_components(score_a)
    v01_b, v10_b, auroc_b = fast_components(score_b)

    sx = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    sy = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s = sx / m + sy / n
    var = float(s[0, 0] + s[1, 1] - 2 * s[0, 1])
    delta = auroc_a - auroc_b
    if var <= 0:
        p = float("nan")
        z = float("nan")
    else:
        z = delta / np.sqrt(var)
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "auroc_a": float(auroc_a),
        "auroc_b": float(auroc_b),
        "delta": float(delta),
        "z": float(z) if np.isfinite(z) else None,
        "p_value": p,
        "var": var,
    }


def paired_bootstrap_metric(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    valid = np.isfinite(y) & np.isfinite(score_a) & np.isfinite(score_b)
    y_v, a_v, b_v = y[valid], score_a[valid], score_b[valid]
    n = len(y_v)
    if n < 30:
        return {"delta": float("nan"), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            deltas[b] = metric_fn(y_v[idx], a_v[idx]) - metric_fn(y_v[idx], b_v[idx])
        except Exception:
            deltas[b] = np.nan
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size < 100:
        return {"delta": float("nan"), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    delta_point = float(metric_fn(y_v, a_v) - metric_fn(y_v, b_v))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float(2 * min((deltas >= 0).mean(), (deltas <= 0).mean()))
    return {
        "delta": delta_point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value_two_sided_bootstrap": p,
        "n": int(n),
    }


def index_runs() -> dict[str, list[Path]]:
    runs = list(PRED_DIR.glob("*.npz"))
    grouped: dict[str, list[Path]] = defaultdict(list)
    for p in runs:
        meta = parse_run_id(p.stem)
        if meta is None:
            logging.warning("Skipping unparseable run_id: %s", p.name)
            continue
        # Group key: ignore seed and ignore loco-tag suffix so we can pair runs.
        key = (
            f"{meta['backbone_virtual']}__{meta['src']}_to_{meta['dst']}"
            f"__{meta['ablation']}__tier{meta['tier']}"
        )
        grouped[key].append(p)
    return grouped


def per_run_cis(runs: dict[str, list[Path]], n_boot: int) -> list[dict]:
    rng = np.random.default_rng(12345)
    rows: list[dict] = []
    for key, paths in sorted(runs.items()):
        for p in paths:
            meta = parse_run_id(p.stem)
            data = np.load(p)
            for k in data.files:
                if not k.startswith("logit__"):
                    continue
                target = k[len("logit__"):]
                logits = data[k]
                y = data[f"y__{target}"]
                if target in BINARY_TARGETS:
                    metric_fn = auroc
                    metric_name = "auroc"
                    score = 1.0 / (1.0 + np.exp(-logits))
                elif target in REGRESSION_TARGETS:
                    metric_fn = spearman
                    metric_name = "spearman_rho"
                    score = logits
                else:
                    continue
                point, lo, hi, sd = bootstrap_metric(y, score, metric_fn, n_boot, rng)
                rows.append({
                    "run_id": p.stem,
                    "backbone": meta["backbone"],
                    "backbone_virtual": meta["backbone_virtual"],
                    "src": meta["src"], "dst": meta["dst"],
                    "ablation": meta["ablation"], "tier": meta["tier"],
                    "seed": meta["seed"],
                    "target": target,
                    "metric": metric_name,
                    "point": point,
                    "ci_lo": lo, "ci_hi": hi, "boot_std": sd,
                    "n_boot": n_boot,
                })
    return rows


def seed_aggregate(rows: list[dict]) -> dict:
    """Aggregate across seeds for the same (backbone_virtual, dir, abl, target)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        k = (r["backbone_virtual"], r["src"], r["dst"], r["ablation"], r["target"])
        groups[k].append(r)
    out: dict = {}
    for k, recs in groups.items():
        pts = np.array([r["point"] for r in recs if np.isfinite(r["point"])])
        if pts.size == 0:
            continue
        # Pool bootstrap CIs across seeds via averaging the per-seed CI bounds
        # (a quick approximation; per-seed bootstrap is the primary CI).
        los = np.array([r["ci_lo"] for r in recs if np.isfinite(r["ci_lo"])])
        his = np.array([r["ci_hi"] for r in recs if np.isfinite(r["ci_hi"])])
        out[" / ".join(k)] = {
            "backbone": k[0],
            "direction": f"{k[1]}->{k[2]}",
            "ablation": k[3],
            "target": k[4],
            "metric": recs[0]["metric"],
            "n_seeds": int(pts.size),
            "mean": float(pts.mean()),
            "std": float(pts.std(ddof=1)) if pts.size > 1 else 0.0,
            "ci_lo_mean": float(los.mean()) if los.size else float("nan"),
            "ci_hi_mean": float(his.mean()) if his.size else float("nan"),
        }
    return out


def paired_tests(
    runs: dict[str, list[Path]],
    pairs: list[tuple[str, str]],
    n_boot: int,
) -> list[dict]:
    rng = np.random.default_rng(67890)
    out: list[dict] = []
    # Build {(backbone_virtual, src, dst, ablation, tier) -> [paths]}
    by_key: dict[tuple, list[Path]] = defaultdict(list)
    for paths in runs.values():
        for p in paths:
            m = parse_run_id(p.stem)
            by_key[(m["backbone_virtual"], m["src"], m["dst"],
                    m["ablation"], m["tier"])].append(p)
    # For each pair (a, b) and each direction/ablation/tier where both exist,
    # use seed=42 records (or earliest seed) as the canonical pair.
    for a, b in pairs:
        # All directions where both a and b have at least one run.
        a_keys = {k for k in by_key if k[0] == a}
        b_keys = {k for k in by_key if k[0] == b}
        for ka in a_keys:
            kb = (b,) + ka[1:]
            if kb not in b_keys:
                continue
            pa = sorted(by_key[ka], key=lambda p: parse_run_id(p.stem)["seed"])[0]
            pb = sorted(by_key[kb], key=lambda p: parse_run_id(p.stem)["seed"])[0]
            da = np.load(pa); db = np.load(pb)
            ids_a = da["point_ids"]; ids_b = db["point_ids"]
            if not np.array_equal(ids_a, ids_b):
                # Should not happen for cross-city since test set = test_city locked points;
                # but join on point_id just in case ordering differs.
                set_a = {int(x): i for i, x in enumerate(ids_a)}
                common = [int(x) for x in ids_b if int(x) in set_a]
                idx_a = np.array([set_a[i] for i in common])
                idx_b = np.array([np.where(ids_b == i)[0][0] for i in common])
            else:
                idx_a = idx_b = np.arange(len(ids_a))
            for k in da.files:
                if not k.startswith("logit__"):
                    continue
                target = k[len("logit__"):]
                if target not in BINARY_TARGETS and target not in REGRESSION_TARGETS:
                    continue
                logits_a = da[k][idx_a]
                logits_b = db[k][idx_b]
                y = da[f"y__{target}"][idx_a]
                if target in BINARY_TARGETS:
                    score_a = 1.0 / (1.0 + np.exp(-logits_a))
                    score_b = 1.0 / (1.0 + np.exp(-logits_b))
                    delong = paired_delong(y, score_a, score_b)
                    boot = paired_bootstrap_metric(y, score_a, score_b, auroc, n_boot, rng)
                    metric = "auroc"
                else:
                    score_a, score_b = logits_a, logits_b
                    delong = {}
                    boot = paired_bootstrap_metric(y, score_a, score_b, spearman, n_boot, rng)
                    metric = "spearman_rho"
                out.append({
                    "backbone_a": a, "backbone_b": b,
                    "src": ka[1], "dst": ka[2],
                    "ablation": ka[3], "tier": ka[4],
                    "target": target, "metric": metric,
                    "delong": delong,
                    "bootstrap": boot,
                    "run_a": pa.stem, "run_b": pb.stem,
                })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CIs + paired DeLong on saved predictions.")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--pairs", action="append", default=[],
                   help="Paired backbone test, format 'a,b' (e.g. siglip,siglip_lora_loco). "
                        "Use the virtual backbone tag — siglip_lora_loco aggregates the 3 LOCO adapters.")
    p.add_argument("--require-pairs", action="store_true",
                   help="Exit non-zero if any requested pair has no overlapping runs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    REPORT.mkdir(parents=True, exist_ok=True)
    if not PRED_DIR.exists():
        logging.error("No predictions directory: %s. Re-run trainer with --save-predictions.", PRED_DIR)
        return 2
    runs = index_runs()
    logging.info("Indexed %d run-groups, %d total prediction files.",
                 len(runs), sum(len(v) for v in runs.values()))

    rows = per_run_cis(runs, args.n_boot)
    out_jsonl = REPORT / "bootstrap_cis.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    logging.info("Wrote %d per-run CI records → %s", len(rows), out_jsonl.name)

    summary = seed_aggregate(rows)
    (REPORT / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote seed-aggregate summary (n_keys=%d).", len(summary))

    pairs: list[tuple[str, str]] = []
    for s in args.pairs:
        a, b = s.split(",")
        pairs.append((a.strip(), b.strip()))
    if not pairs:
        pairs = [
            ("siglip_lora_loco", "siglip"),
            ("siglip_lora_loco", "clip"),
            ("siglip_lora", "siglip_lora_loco"),
            ("siglip", "clip"),
            # SRC backbone comparisons
            ("siglip2", "siglip"),
            ("dinov2", "siglip"),
            ("siglip2+remoteclip", "siglip2"),
        ]
    paired = paired_tests(runs, pairs, args.n_boot)
    out_paired = REPORT / "paired_tests.jsonl"
    out_paired.write_text("\n".join(json.dumps(r) for r in paired), encoding="utf-8")
    logging.info("Wrote %d paired-test records → %s", len(paired), out_paired.name)

    if args.require_pairs:
        seen = {(r["backbone_a"], r["backbone_b"]) for r in paired}
        missing = [p for p in pairs if p not in seen]
        if missing:
            logging.error("Missing paired tests for: %s", missing)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
