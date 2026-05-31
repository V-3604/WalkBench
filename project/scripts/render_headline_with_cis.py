"""
Render the paper headline table (Tier-1 cross-city, ablation=full) with
seed-aggregated mean ± std and bootstrap 95% CIs from bootstrap_cis.py.

Reads:
    project/artifacts/reports/bootstrap_summary.json   (seed aggregate)
    project/artifacts/reports/paired_tests.jsonl       (DeLong p-values)

Writes:
    project/artifacts/reports/headline_table_with_cis.tex

Output format: per-backbone block, one row per direction with
"AUROC mean ± std [bootstrap_95_lo, bootstrap_95_hi]" cells. Adds a star
on cells where the paired DeLong test against frozen SigLIP rejects at
p<0.05, and a separate row for the cross-direction mean.

Run:
    python project/scripts/render_headline_with_cis.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT    = REPO_ROOT / "project/artifacts/reports"

BACKBONES = ["siglip", "clip", "siglip_lora", "siglip_lora_loco", "siglip_lora_loco_v2"]
LABELS = {
    "siglip":              "SigLIP-SO400M (frozen)",
    "clip":                "CLIP ViT-L/14 (frozen)",
    "siglip_lora":         "SigLIP+LoRA (in-corpus)",
    "siglip_lora_loco":    "SigLIP+LoRA (LOCO)",
    "siglip_lora_loco_v2": "SigLIP+LoRA (LOCO v2, multi-task)",
}
TARGETS = [
    ("overture_sidewalk_present",            "Sidewalk AUROC",      "auroc"),
    ("overture_crosswalk_present",           "Crosswalk AUROC",     "auroc"),
    ("overture_intersection_count_200m",     "Intersect ρ",         "spearman_rho"),
    ("overture_building_footprint_frac_100m","Bldg-frac ρ",         "spearman_rho"),
    ("walkability_index_v1",                 "Walk Index ρ",        "spearman_rho"),
]
DIRECTIONS = ["msp->seattle", "msp->dc", "seattle->msp", "seattle->dc",
              "dc->msp", "dc->seattle"]


def load() -> tuple[dict, list]:
    summary_path = REPORT / "bootstrap_summary.json"
    paired_path  = REPORT / "paired_tests.jsonl"
    if not summary_path.exists():
        raise SystemExit(
            f"Missing {summary_path}. Run: python project/scripts/bootstrap_cis.py "
            "(after running train_multitarget.py with --save-predictions on the headline grid).")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paired: list[dict] = []
    if paired_path.exists():
        for line in paired_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                paired.append(json.loads(line))
    return summary, paired


def cell(summary: dict, backbone: str, direction: str, target: str) -> str:
    key = f"{backbone} / {direction.split('->')[0]} / {direction.split('->')[1]} / full / {target}"
    rec = summary.get(key)
    if rec is None:
        return "---"
    mean = rec["mean"]
    std  = rec["std"]
    lo   = rec.get("ci_lo_mean", float("nan"))
    hi   = rec.get("ci_hi_mean", float("nan"))
    if not np.isfinite(lo):
        return f"{mean:.3f}"
    if rec["n_seeds"] > 1:
        return f"{mean:.3f}$\\pm${std:.3f} [{lo:.3f},{hi:.3f}]"
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"


def is_significant_vs_frozen(paired: list, backbone: str, direction: str, target: str) -> bool:
    src, dst = direction.split("->")
    for r in paired:
        if (r["backbone_a"] == backbone and r["backbone_b"] == "siglip"
            and r["src"] == src and r["dst"] == dst
            and r["target"] == target):
            p = r.get("delong", {}).get("p_value")
            if p is None:
                p = r.get("bootstrap", {}).get("p_value_two_sided_bootstrap")
            return bool(p is not None and np.isfinite(p) and p < 0.05)
    return False


def render() -> str:
    summary, paired = load()
    out: list[str] = []
    out.append("% Auto-generated headline table with bootstrap CIs and DeLong stars.")
    out.append("% '*' = paired DeLong p<0.05 vs SigLIP-frozen on the same direction/target.")
    out.append("\\begin{table}[t]")
    out.append("\\centering\\small")
    out.append("\\caption{Cross-city Tier-1 results (ablation=full). Cell = "
               "mean$\\pm$std across seeds [percentile bootstrap 95\\% CI]. "
               "Stars indicate paired DeLong test vs SigLIP-frozen $p<0.05$.}")
    out.append("\\label{tab:cross-city-cis}")
    out.append("\\begin{tabular}{l " + "l " * len(TARGETS) + "}")
    out.append("\\toprule")
    out.append("Backbone / Direction & " + " & ".join(t[1] for t in TARGETS) + " \\\\")
    out.append("\\midrule")
    for bb in BACKBONES:
        out.append(f"\\multicolumn{{{1+len(TARGETS)}}}{{l}}{{\\textit{{{LABELS[bb]}}}}} \\\\")
        for d in DIRECTIONS:
            row = [d]
            for tgt, _, _ in TARGETS:
                c = cell(summary, bb, d, tgt)
                star = "*" if (bb != "siglip" and is_significant_vs_frozen(paired, bb, d, tgt)) else ""
                row.append(c + star)
            out.append(" & ".join(row) + " \\\\")
        # Cross-direction mean
        means = []
        for tgt, _, _ in TARGETS:
            vals = []
            for d in DIRECTIONS:
                key = f"{bb} / {d.split('->')[0]} / {d.split('->')[1]} / full / {tgt}"
                if key in summary:
                    vals.append(summary[key]["mean"])
            means.append(f"\\textbf{{{np.mean(vals):.3f}}}" if vals else "---")
        out.append("\\textit{mean across directions} & " + " & ".join(means) + " \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    text = render()
    out = REPORT / "headline_table_with_cis.tex"
    out.write_text(text, encoding="utf-8")
    logging.info("Wrote %s (%d chars)", out, len(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
