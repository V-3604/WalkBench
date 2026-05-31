"""
Mean-logit ensemble across {siglip, clip, siglip_lora_loco_v2_<dst>} for
the same (src, dst, seed, target). Recomputes AUROC/AP/rho on the average.

Inputs:
    project/artifacts/predictions/*.npz

Output:
    project/artifacts/reports/ensemble_results.json

Run:
    python project/scripts/ensemble_logits.py
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "project/artifacts/predictions"
OUT = REPO_ROOT / "project/artifacts/reports/ensemble_results.json"

RUN_RE = re.compile(
    r"^(?P<bb>[a-z0-9_]+?)__(?P<mode>[a-z_]+)__(?P<src>[a-z]+)_to_(?P<dst>[a-z]+)"
    r"__tier(?P<tier>\d+)__(?P<abl>[a-z_]+)__seed(?P<seed>\d+)$"
)
ENSEMBLE_BB = {"siglip", "clip", "siglip_lora_loco_v2"}


def collapse_bb(bb: str) -> str:
    if bb.startswith("siglip_lora_loco_v2_"):
        return "siglip_lora_loco_v2"
    if bb.startswith("siglip_lora_loco_"):
        return "siglip_lora_loco"
    return bb


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bundles: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for npz in sorted(PRED_DIR.glob("*.npz")):
        m = RUN_RE.match(npz.stem)
        if not m:
            continue
        d = m.groupdict()
        col_bb = collapse_bb(d["bb"])
        if col_bb not in ENSEMBLE_BB:
            continue
        key = (d["src"], d["dst"], int(d["seed"]), d["abl"])
        z = np.load(npz, allow_pickle=False)
        bundles[key][col_bb] = {f: z[f] for f in z.files}

    rows: list[dict] = []
    for (src, dst, seed, abl), bb_map in bundles.items():
        if not ENSEMBLE_BB.issubset(set(bb_map)):
            continue
        ref = next(iter(bb_map.values()))
        for f in ref:
            if not f.startswith("logit__"):
                continue
            target = f.replace("logit__", "")
            stack = np.stack([bb_map[bb][f] for bb in ENSEMBLE_BB], axis=0)
            avg = stack.mean(axis=0)
            y = ref[f"y__{target}"]
            mask = np.isfinite(y) & np.isfinite(avg)
            if mask.sum() < 30:
                continue
            yt, yp = y[mask], avg[mask]
            row: dict = {
                "src": src, "dst": dst, "seed": seed, "abl": abl,
                "target": target, "n": int(mask.sum()),
            }
            uniq = np.unique(yt[~np.isnan(yt)])
            if set(uniq.astype(int).tolist()) <= {0, 1} and len(uniq) == 2:
                row["auroc"] = float(roc_auc_score(yt.astype(int), yp))
                row["ap"] = float(average_precision_score(yt.astype(int), yp))
            else:
                rho, _ = spearmanr(yt, yp)
                row["spearman_rho"] = float(rho)
                row["mae"] = float(mean_absolute_error(yt, yp))
            rows.append(row)

    if not rows:
        raise SystemExit("No complete (siglip, clip, lora_loco_v2) triples found.")

    df = pd.DataFrame(rows)
    out_data: dict = {"n_rows": len(df), "rows": df.to_dict(orient="records")}

    # Per-target summary.
    summary_rows: list[dict] = []
    for target, grp in df.groupby("target"):
        entry: dict = {"target": target, "n_runs": len(grp)}
        if "auroc" in grp.columns and grp["auroc"].notna().any():
            entry["auroc_mean"] = float(grp["auroc"].mean())
            entry["auroc_std"] = float(grp["auroc"].std())
        if "ap" in grp.columns and grp["ap"].notna().any():
            entry["ap_mean"] = float(grp["ap"].mean())
        if "spearman_rho" in grp.columns and grp["spearman_rho"].notna().any():
            entry["rho_mean"] = float(grp["spearman_rho"].mean())
            entry["rho_std"] = float(grp["spearman_rho"].std())
        summary_rows.append(entry)
    out_data["summary"] = summary_rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    for s in summary_rows:
        logging.info("  %-45s %s", s["target"],
                     " ".join(f"{k}={v:.4f}" for k, v in s.items() if isinstance(v, float)))
    logging.info("Wrote %s (%d rows)", OUT, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
