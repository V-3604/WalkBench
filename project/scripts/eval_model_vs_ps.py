"""
Model-vs-Project-Sidewalk evaluation for Seattle test points.

For each .npz prediction bundle in project/artifacts/predictions/ where the
test city is Seattle, derive PS labels at each lock point (using the same
25 m match radius as audit_label_noise.py) and compute AUROC against PS for
sidewalk_present and crosswalk_present. Outputs a side-by-side comparison
to the matched model-vs-Overture AUROC.

This is the headline robustness check that turns the Seattle κ=0.135
label-noise finding into a positive number for the same trained model.

Inputs:
    project/artifacts/predictions/*.npz
    project/data/external/sidewalk-equity-study/seattle/datasets/01-project-sidewalk-labels/
    project/data/processed/points/seattle_points_v2_final_lock.csv

Output:
    project/artifacts/reports/model_vs_ps.json
    project/artifacts/reports/model_vs_ps.csv  (one row per (run_id, target))
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Re-use the PS loader and label deriver from audit_label_noise.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_label_noise import (
    load_ps, derive_ps_point_labels, CITY_UTM_EPSG,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR  = REPO_ROOT / "project/artifacts/predictions"
POINTS_DIR = REPO_ROOT / "project/data/processed/points"
REPORT    = REPO_ROOT / "project/artifacts/reports"

RUN_ID_RE = re.compile(
    r"^(?P<bb>[a-z0-9_]+?)__(?P<mode>[a-z_]+)__(?P<src>[a-z]+)_to_(?P<dst>[a-z]+)"
    r"__tier(?P<tier>\d+)__(?P<abl>[a-z_]+)__seed(?P<seed>\d+)$"
)


def parse_run_id(stem: str) -> dict | None:
    m = RUN_ID_RE.match(stem)
    return m.groupdict() if m else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--match-radius-m", type=float, default=25.0)
    p.add_argument("--test-city", default="seattle",
                   help="PS only ships Seattle in this clone; override with care.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    pts = pd.read_csv(POINTS_DIR / f"{args.test_city}_points_v2_final_lock.csv")
    pts["city"] = args.test_city
    pts["point_id"] = pts["point_id"].astype(int)

    ps = load_ps(args.test_city)
    if ps.empty:
        raise SystemExit(f"No PS labels for {args.test_city}.")
    logging.info("[%s] PS labels loaded: %d", args.test_city, len(ps))
    ps_pts = derive_ps_point_labels(ps, pts, args.test_city, args.match_radius_m)
    ps_pts["point_id"] = ps_pts["point_id"].astype(int)

    rows = []
    for npz in sorted(PRED_DIR.glob("*.npz")):
        meta = parse_run_id(npz.stem)
        if meta is None:
            logging.debug("skip (unparseable): %s", npz.name)
            continue
        if meta["dst"] != args.test_city:
            continue
        d = np.load(npz, allow_pickle=False)
        pids = d["point_ids"].astype(int)
        df = pd.DataFrame({"point_id": pids})
        df = df.merge(ps_pts[["point_id", "ps_sidewalk_present",
                               "ps_crosswalk_present"]],
                      on="point_id", how="left")

        for short, ps_col in [("overture_sidewalk_present",  "ps_sidewalk_present"),
                              ("overture_crosswalk_present", "ps_crosswalk_present")]:
            logit_key = f"logit__{short}"
            label_key = f"y__{short}"
            if logit_key not in d.files:
                continue
            logits = d[logit_key]
            y_over = d[label_key].astype(int)
            y_ps = df[ps_col].to_numpy(dtype=float)
            mask = np.isfinite(y_ps) & np.isfinite(logits)
            n = int(mask.sum())
            if n < 50:
                continue
            yp = y_ps[mask].astype(int)
            yo = y_over[mask].astype(int)
            lp = logits[mask]
            try:
                auroc_ps   = float(roc_auc_score(yp, lp)) if len(np.unique(yp)) > 1 else float("nan")
                auroc_over = float(roc_auc_score(yo, lp)) if len(np.unique(yo)) > 1 else float("nan")
                ap_ps      = float(average_precision_score(yp, lp)) if len(np.unique(yp)) > 1 else float("nan")
                ap_over    = float(average_precision_score(yo, lp)) if len(np.unique(yo)) > 1 else float("nan")
            except Exception as e:
                logging.warning("metric fail on %s/%s: %s", npz.stem, short, e)
                continue
            rows.append({
                "run_id":    npz.stem,
                "backbone":  meta["bb"],
                "src":       meta["src"],
                "dst":       meta["dst"],
                "ablation":  meta["abl"],
                "seed":      int(meta["seed"]),
                "target":    short,
                "n":         n,
                "ps_prevalence":       float(yp.mean()),
                "overture_prevalence": float(yo.mean()),
                "auroc_vs_ps":        auroc_ps,
                "auroc_vs_overture":  auroc_over,
                "ap_vs_ps":           ap_ps,
                "ap_vs_overture":     ap_over,
                "delta_auroc":        auroc_ps - auroc_over if np.isfinite(auroc_ps) and np.isfinite(auroc_over) else float("nan"),
            })

    if not rows:
        raise SystemExit("No matching .npz prediction files found. "
                         "Run train_multitarget.py with --save-predictions first "
                         "(see Tier S queue in v2_RUNBOOK_2026.md).")

    df = pd.DataFrame(rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT / "model_vs_ps.csv", index=False)

    # Aggregate.
    agg = df.groupby(["backbone", "target"]).agg(
        n_runs=("run_id", "size"),
        auroc_vs_ps_mean=("auroc_vs_ps", "mean"),
        auroc_vs_ps_std=("auroc_vs_ps", "std"),
        auroc_vs_overture_mean=("auroc_vs_overture", "mean"),
        delta_auroc_mean=("delta_auroc", "mean"),
        ap_vs_ps_mean=("ap_vs_ps", "mean"),
        ap_vs_overture_mean=("ap_vs_overture", "mean"),
    ).reset_index()
    out = {
        "match_radius_m": args.match_radius_m,
        "test_city": args.test_city,
        "summary": agg.to_dict(orient="records"),
    }
    (REPORT / "model_vs_ps.json").write_text(json.dumps(out, indent=2),
                                              encoding="utf-8")
    logging.info("Wrote %s and model_vs_ps.csv (%d rows)",
                 REPORT / "model_vs_ps.json", len(df))
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
