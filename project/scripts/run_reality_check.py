"""
Reality-check orchestrator: the NATIVE rungs of the complexity ladder.

Runs the frozen-encoder rungs that need no GPU training -- linear probe (the
baseline-to-beat), CORAL, and subspace alignment -- on the 4 active targets, under
the cross-city protocol (6 LOCO directions among MSP/Seattle/DC + Pittsburgh
zero-shot), for one or more frozen backbones and seeds.

Each (method, backbone, direction, seed) writes a prediction bundle whose schema is
consumed unchanged by bootstrap_cis.py / the Phase-3 spatial bootstrap:

    point_ids                 int64    (n_test,)
    logit__<target>           float32  log-odds (binary) | predicted value (regression)
    y__<target>               float32  raw ground truth (NaN where unlabeled)

Filename / run_id (parseable by bootstrap_cis.RUN_RE):
    {backbone}_{method}__cross_city__{src}_to_{dst}__tier1__full__seed{seed}.npz
e.g. siglip2_linprobe__cross_city__seattle_to_dc__tier1__full__seed42.npz
The MLP / LoRA rungs are produced separately by train_multitarget.py --save-predictions.

Featurization (per modality, then concatenated):
  linprobe   PCA-128(street) + PCA-128(naip), StandardScaler, fit on train city only.
  coral      same PCA + scaler, then CORAL-align the source to the target city's
             feature moments (unsupervised) before the probe.
  subspace   per-domain PCA-128 + subspace alignment, then StandardScaler.
Probes: logistic regression (class_weight balanced, C tuned on a source split) for
binary targets; RidgeCV for regression. Intersection count is fit in log1p space
(Spearman rho is rank-invariant, so the stored prediction space does not change it).

Run (Windows venv, CPU; repo ROOT):
    # full native matrix (3 methods x backbones x 7 directions x 10 seeds)
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_reality_check.py
    # Phase-1 smoke: one direction, one seed
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_reality_check.py \\
        --methods linprobe coral subspace --backbones siglip2 \\
        --directions seattle_to_dc --seeds 42 --out-jsonl reality_check_smoke.jsonl
    # plan only, no compute
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_reality_check.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from methods_da import coral_align, subspace_align

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "project/artifacts/embeddings"
LABELS_CSV = REPO_ROOT / "project/data/processed/labels/features_labels_agreement.csv"
OSM_CSV = REPO_ROOT / "project/data/processed/labels/osm_crossing_labels.csv"
PRED_DIR = REPO_ROOT / "project/artifacts/predictions"
REPORT_DIR = REPO_ROOT / "project/artifacts/reports"

PCA_DIM = 128
TRIO = ["msp", "seattle", "dc"]
LOGREG_CS = (0.1, 1.0, 10.0)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_SEEDS = list(range(41, 51))
NATIVE_METHODS = ("linprobe", "coral", "subspace")


@dataclass(frozen=True)
class Target:
    name: str
    kind: str            # "binary" | "regression"
    metric: str          # "auroc" | "spearman_rho"
    log1p: bool = False


TARGETS: tuple[Target, ...] = (
    Target("osm_marked_crossing_present", "binary", "auroc"),
    Target("overture_sidewalk_present", "binary", "auroc"),
    Target("overture_intersection_count_200m", "regression", "spearman_rho", log1p=True),
    Target("overture_building_footprint_frac_100m", "regression", "spearman_rho"),
)


@dataclass(frozen=True)
class Direction:
    src: tuple[str, ...]   # train cities
    dst: str               # test city

    @property
    def src_token(self) -> str:
        return "+".join(self.src)

    @property
    def label(self) -> str:
        return f"{self.src_token}->{self.dst}"


def all_directions() -> list[Direction]:
    loco = [Direction((a,), b) for a in TRIO for b in TRIO if a != b]
    pittsburgh = Direction(tuple(sorted(TRIO)), "pittsburgh")
    return loco + [pittsburgh]


@dataclass
class CityBlock:
    street: np.ndarray
    naip: np.ndarray
    point_ids: np.ndarray
    labels: pd.DataFrame   # aligned row-for-row with street/naip


def load_embedding(backbone: str, source: str, city: str) -> tuple[np.ndarray, np.ndarray]:
    emb = np.load(EMBED_DIR / f"{backbone}_{source}_{city}.npy").astype(np.float32)
    ids = np.load(EMBED_DIR / f"{backbone}_{source}_{city}_ids.npy").astype(np.int64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    return emb, ids


def load_city_block(backbone: str, city: str, labels: pd.DataFrame) -> CityBlock:
    """Street + NAIP embeddings aligned with labels for points present in all three."""
    street_emb, street_ids = load_embedding(backbone, "street", city)
    naip_emb, naip_ids = load_embedding(backbone, "naip", city)
    street_index = {int(p): i for i, p in enumerate(street_ids)}
    naip_index = {int(p): i for i, p in enumerate(naip_ids)}
    city_labels = labels[labels["city"] == city]
    street_rows, naip_rows, kept = [], [], []
    for point_id in city_labels["point_id"].astype(int):
        if point_id in street_index and point_id in naip_index:
            street_rows.append(street_emb[street_index[point_id]])
            naip_rows.append(naip_emb[naip_index[point_id]])
            kept.append(point_id)
    frame = city_labels.set_index("point_id").loc[kept].reset_index()
    return CityBlock(
        street=np.asarray(street_rows, dtype=np.float32),
        naip=np.asarray(naip_rows, dtype=np.float32),
        point_ids=np.asarray(kept, dtype=np.int64),
        labels=frame,
    )


def load_labels() -> pd.DataFrame:
    overture_cols = [t.name for t in TARGETS if t.name.startswith("overture_")]
    fla = pd.read_csv(LABELS_CSV, usecols=["point_id", "city", *overture_cols], low_memory=False)
    fla["point_id"] = fla["point_id"].astype(int)
    osm = pd.read_csv(OSM_CSV, usecols=["point_id", "city", "osm_marked_crossing_present"])
    osm["point_id"] = osm["point_id"].astype(int)
    return fla.merge(osm, on=["point_id", "city"], how="left")


def featurize(
    method: str,
    street_train: np.ndarray,
    naip_train: np.ndarray,
    street_test: np.ndarray,
    naip_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-modality featurization for a rung; returns (X_train, X_test) concatenated.

    For coral/subspace the *test* embeddings stand in for the target city's unlabeled
    images used by the unsupervised alignment.
    """
    train_parts, test_parts = [], []
    for train_mod, test_mod in ((street_train, street_test), (naip_train, naip_test)):
        if method == "subspace":
            aligned_train, projected_test = subspace_align(train_mod, test_mod, PCA_DIM, random_state=seed)
            scaler = StandardScaler().fit(aligned_train)
            train_parts.append(scaler.transform(aligned_train))
            test_parts.append(scaler.transform(projected_test))
            continue
        n_components = min(PCA_DIM, train_mod.shape[1], train_mod.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=seed).fit(train_mod)
        scaler = StandardScaler().fit(pca.transform(train_mod))
        train_scaled = scaler.transform(pca.transform(train_mod))
        test_scaled = scaler.transform(pca.transform(test_mod))
        if method == "coral":
            train_scaled = coral_align(train_scaled, test_scaled)
        train_parts.append(train_scaled)
        test_parts.append(test_scaled)
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def tune_logistic(features: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    """Pick C on a stratified source hold-out (AUROC), then refit on all source data."""
    best_c, best_score = LOGREG_CS[0], -np.inf
    if len(np.unique(y)) == 2 and np.bincount(y).min() >= 5:
        x_fit, x_val, y_fit, y_val = train_test_split(
            features, y, test_size=0.2, random_state=seed, stratify=y)
        for c in LOGREG_CS:
            clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=c,
                                     random_state=seed).fit(x_fit, y_fit)
            score = roc_auc_score(y_val, clf.decision_function(x_val))
            if score > best_score:
                best_score, best_c = score, c
    return LogisticRegression(class_weight="balanced", max_iter=2000, C=best_c,
                              random_state=seed).fit(features, y)


def fit_predict_target(
    method: str,
    target: Target,
    street_train: np.ndarray,
    naip_train: np.ndarray,
    y_train: np.ndarray,
    block_test: CityBlock,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a rung's probe and score every test point. Returns (logit/pred, y_raw, metric)."""
    y_test_raw = block_test.labels[target.name].to_numpy(dtype=np.float64)
    train_mask = np.isfinite(y_train)
    features_train, features_test = featurize(
        method, street_train[train_mask], naip_train[train_mask],
        block_test.street, block_test.naip, seed)
    test_mask = np.isfinite(y_test_raw)

    if target.kind == "binary":
        probe = tune_logistic(features_train, y_train[train_mask].astype(int), seed)
        scores = probe.decision_function(features_test).astype(np.float32)
        metric = float("nan")
        if test_mask.sum() >= 30 and len(np.unique(y_test_raw[test_mask])) == 2:
            metric = float(roc_auc_score(y_test_raw[test_mask].astype(int), expit(scores[test_mask])))
    else:
        target_values = np.log1p(y_train[train_mask]) if target.log1p else y_train[train_mask]
        probe = RidgeCV(alphas=RIDGE_ALPHAS).fit(features_train, target_values)
        scores = probe.predict(features_test).astype(np.float32)
        metric = float("nan")
        if test_mask.sum() >= 30:
            metric = float(spearmanr(scores[test_mask], y_test_raw[test_mask]).correlation)
    return scores, y_test_raw.astype(np.float32), metric


def run_unit(
    method: str,
    backbone: str,
    direction: Direction,
    seed: int,
    blocks: dict[str, CityBlock],
) -> tuple[str, list[dict]]:
    """One (method, backbone, direction, seed): write the npz, return (run_id, jsonl rows)."""
    block_test = blocks[direction.dst]
    street_train = np.concatenate([blocks[c].street for c in direction.src])
    naip_train = np.concatenate([blocks[c].naip for c in direction.src])

    tag = f"{backbone}_{method}"
    run_id = f"{tag}__cross_city__{direction.src_token}_to_{direction.dst}__tier1__full__seed{seed}"
    npz_payload: dict[str, np.ndarray] = {"point_ids": block_test.point_ids.astype(np.int64)}
    rows: list[dict] = []
    for target in TARGETS:
        y_train = np.concatenate([blocks[c].labels[target.name].to_numpy(dtype=np.float64)
                                  for c in direction.src])
        scores, y_raw, metric = fit_predict_target(
            method, target, street_train, naip_train, y_train, block_test, seed)
        npz_payload[f"logit__{target.name}"] = scores
        npz_payload[f"y__{target.name}"] = y_raw
        rows.append({
            "backbone": backbone, "method": method, "tag": tag,
            "eval_mode": "cross_city", "src": direction.src_token, "dst": direction.dst,
            "direction": direction.label, "tier": 1, "ablation": "full", "seed": seed,
            "target": target.name, "metric": target.metric,
            "n_test": int(np.isfinite(y_raw).sum()),
            "value": round(metric, 4) if np.isfinite(metric) else None,
            "run_id": run_id,
        })
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_DIR / f"{run_id}.npz", **npz_payload)
    return run_id, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native reality-check rungs (linear probe, CORAL, subspace).")
    parser.add_argument("--methods", nargs="+", default=list(NATIVE_METHODS), choices=NATIVE_METHODS)
    parser.add_argument("--backbones", nargs="+", default=["siglip2"],
                        help="Frozen embedding backbones (e.g. siglip2 siglip dinov2).")
    parser.add_argument("--directions", nargs="+", default=None,
                        help="Subset like 'seattle_to_dc' or 'dc+msp+seattle_to_pittsburgh'. Default: all 7.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of run units.")
    parser.add_argument("--dry-run", action="store_true", help="List planned run units and exit.")
    parser.add_argument("--out-jsonl", default="reality_check_results.jsonl",
                        help="Results filename under project/artifacts/reports/.")
    return parser.parse_args()


def select_directions(requested: list[str] | None) -> list[Direction]:
    available = {d.label.replace("->", "_to_"): d for d in all_directions()}
    if requested is None:
        return list(available.values())
    chosen: list[Direction] = []
    for name in requested:
        if name not in available:
            raise SystemExit(f"Unknown direction '{name}'. Choices: {sorted(available)}")
        chosen.append(available[name])
    return chosen


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    directions = select_directions(args.directions)

    units = [(m, b, d, s)
             for b in args.backbones for m in args.methods
             for d in directions for s in args.seeds]
    if args.limit is not None:
        units = units[: args.limit]
    logging.info("Planned run units: %d (methods=%s backbones=%s directions=%d seeds=%s)",
                 len(units), args.methods, args.backbones, len(directions), args.seeds)

    if args.dry_run:
        for method, backbone, direction, seed in units:
            logging.info("  %s_%s  %s  seed%d", backbone, method, direction.label, seed)
        return 0

    labels = load_labels()
    block_cache: dict[tuple[str, str], CityBlock] = {}
    all_rows: list[dict] = []
    for index, (method, backbone, direction, seed) in enumerate(units, start=1):
        for city in (*direction.src, direction.dst):
            key = (backbone, city)
            if key not in block_cache:
                block_cache[key] = load_city_block(backbone, city, labels)
        blocks = {city: block_cache[(backbone, city)] for city in (*direction.src, direction.dst)}
        run_id, rows = run_unit(method, backbone, direction, seed, blocks)
        all_rows.extend(rows)
        summary = "  ".join(
            f"{r['target'].split('_')[1][:5]}={r['value']}" for r in rows)
        logging.info("[%d/%d] %s  %s", index, len(units), run_id, summary)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / args.out_jsonl
    out_path.write_text("\n".join(json.dumps(r) for r in all_rows), encoding="utf-8")
    logging.info("Wrote %d result rows -> %s", len(all_rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
