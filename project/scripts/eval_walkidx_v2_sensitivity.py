"""Sensitivity check: re-score cross-city walkability predictions against the
cleaner 5-feature composite index (no EPA features). Models were trained
against the 8-feature v1 index, so this measures whether the rank order of
predictions still correlates strongly with the cleaner target.

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\eval_walkidx_v2_sensitivity.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "project" / "artifacts" / "predictions"
V2_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "walkability_index_v2_5feat.csv"
OUT_JSON = REPO_ROOT / "project" / "artifacts" / "reports" / "walkidx_v2_sensitivity.json"

PATTERN = re.compile(
    r"^(?P<backbone>siglip|clip)__cross_city__(?P<src>\w+?)_to_(?P<dst>\w+?)__tier1__full__seed(?P<seed>\d+)\.npz$"
)


def main() -> int:
    v2 = pd.read_csv(V2_CSV)
    v2_lookup = {
        (int(row.point_id), row.city): float(row.walkability_index_v2)
        for row in v2.itertuples(index=False)
    }

    rows: list[dict] = []
    for path in sorted(PRED_DIR.glob("*__cross_city__*__tier1__full__seed*.npz")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        d = np.load(path)
        if "logit__walkability_index_v1" not in d:
            continue
        pred = d["logit__walkability_index_v1"]
        pids = d["point_ids"]
        dst = m.group("dst")
        y2 = np.array([v2_lookup.get((int(pid), dst), np.nan) for pid in pids])
        mask = ~np.isnan(y2)
        if mask.sum() < 100:
            continue
        rho_v1 = float(spearmanr(pred[mask], d["y__walkability_index_v1"][mask]).statistic)
        rho_v2 = float(spearmanr(pred[mask], y2[mask]).statistic)
        rows.append(
            {
                "backbone": m.group("backbone"),
                "src": m.group("src"),
                "dst": dst,
                "seed": int(m.group("seed")),
                "n": int(mask.sum()),
                "rho_v1_8feat": rho_v1,
                "rho_v2_5feat": rho_v2,
                "delta": rho_v2 - rho_v1,
            }
        )

    df = pd.DataFrame(rows)
    summary_by_backbone = (
        df.groupby("backbone")[["rho_v1_8feat", "rho_v2_5feat", "delta"]]
        .agg(["mean", "std"])
        .round(4)
        .to_dict()
    )
    overall = {
        "rho_v1_mean": float(df["rho_v1_8feat"].mean()),
        "rho_v1_std": float(df["rho_v1_8feat"].std()),
        "rho_v2_mean": float(df["rho_v2_5feat"].mean()),
        "rho_v2_std": float(df["rho_v2_5feat"].std()),
        "delta_mean": float(df["delta"].mean()),
        "n_cells": int(len(df)),
    }
    OUT_JSON.write_text(
        json.dumps({"per_cell": rows, "overall": overall, "by_backbone": {str(k): v for k, v in summary_by_backbone.items()}}, indent=2)
    )
    print(f"Cells: {len(df)}")
    print(f"v1 (8-feat) mean rho: {overall['rho_v1_mean']:.4f} +- {overall['rho_v1_std']:.4f}")
    print(f"v2 (5-feat) mean rho: {overall['rho_v2_mean']:.4f} +- {overall['rho_v2_std']:.4f}")
    print(f"Delta mean: {overall['delta_mean']:+.4f}")
    print(df.groupby("backbone")[["rho_v1_8feat", "rho_v2_5feat", "delta"]].mean().round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
