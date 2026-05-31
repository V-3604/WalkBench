"""
Label-ceiling analysis (Stage 6).

For each (city, target ∈ {sidewalk, crosswalk}) computes the theoretical
upper-bound AUROC that any model trained on Overture labels can achieve when
evaluated against Project Sidewalk (PS) ground truth, given the observed
inter-source Cohen's κ.

The ceiling is derived from the confusion matrix between Overture and PS:
a model that perfectly predicts PS labels achieves AUROC(PS → Overture) when
scored against Overture as the reference.  For binary labels this resolves to:

    a       = (κ*(1-P_e) + P_e + p_o + p_s - 1) / 2          # P(O=1, T=1)
    ceiling = 0.5 * (1 + a/p_o - (p_s - a)/(1 - p_o))        # AUROC(PS→Overture)

where p_o = Overture prevalence, p_s = PS prevalence,
P_e = p_o*p_s + (1-p_o)*(1-p_s) (chance agreement).

Observed model AUROCs vs Overture are pulled from multitarget_results.jsonl
(src_backbones tag, full ablation, siglip and siglip2 frozen backbones).
Model AUROCs vs PS are pulled from model_vs_ps.json (Seattle only).

Inputs:
    project/artifacts/reports/label_noise_audit.json
    project/artifacts/reports/model_vs_ps.json
    project/artifacts/reports/multitarget_results.jsonl

Outputs:
    project/artifacts/reports/label_ceiling_analysis.json
    project/artifacts/reports/figures/label_ceiling.pdf

Run (Windows, repo root):
    & ".venv\\Scripts\\python.exe" project\\scripts\\label_ceiling_analysis.py
    & ".venv\\Scripts\\python.exe" project\\scripts\\label_ceiling_analysis.py --cities seattle pittsburgh
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "project/artifacts/reports"
AUDIT_JSON = REPORT / "label_noise_audit.json"
MODEL_PS_JSON = REPORT / "model_vs_ps.json"
RESULTS_JSONL = REPORT / "multitarget_results.jsonl"
FIGURES_DIR = REPORT / "figures"

TARGET_COLS = {
    "sidewalk": "overture_sidewalk_present",
    "crosswalk": "overture_crosswalk_present",
}
# Backbones to include when computing mean observed model AUROC vs Overture
FROZEN_BACKBONES = {"siglip", "siglip2", "clip", "dinov2"}


def auroc_ps_vs_overture_from_kappa(
    kappa: float, p_o: float, p_s: float
) -> float:
    """Theoretical AUROC(PS→Overture) given κ and marginal prevalences.

    Derived from the unique confusion matrix consistent with the observed
    κ, p_o, p_s.  Clamps a to the feasible range before computing.
    """
    p_e = p_o * p_s + (1 - p_o) * (1 - p_s)
    a = (kappa * (1 - p_e) + p_e + p_o + p_s - 1) / 2
    a = float(np.clip(a, max(0.0, p_o + p_s - 1), min(p_o, p_s)))
    if p_o <= 0 or p_o >= 1:
        return float("nan")
    sensitivity_ps = a / p_o          # P(PS=1 | Overture=1)
    fpr_ps = (p_s - a) / (1 - p_o)   # P(PS=1 | Overture=0)
    fpr_ps = float(np.clip(fpr_ps, 0.0, 1.0))
    return float(np.clip(0.5 * (1 + sensitivity_ps - fpr_ps), 0.0, 1.0))


def theoretical_curve(
    p_o: float, p_s: float, n_pts: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """AUROC(PS→Overture) vs κ curve at fixed marginals."""
    kappas = np.linspace(0.0, 1.0, n_pts)
    aurocs = np.array([auroc_ps_vs_overture_from_kappa(k, p_o, p_s) for k in kappas])
    return kappas, aurocs


def load_model_auroc_vs_overture(target_col: str) -> dict[str, float]:
    """Mean observed model AUROC vs Overture per backbone from multitarget_results.jsonl.

    Uses src_backbones rows (3-city LOCO, full ablation, frozen backbones).
    Falls back to all cross_city rows if src_backbones is empty.
    """
    records = [json.loads(l) for l in RESULTS_JSONL.open(encoding="utf-8")]
    src = [
        r for r in records
        if r.get("experiment_tag") == "src_backbones"
        and r.get("ablation") == "full"
        and r.get("backbone") in FROZEN_BACKBONES
    ]
    if not src:
        src = [
            r for r in records
            if r.get("ablation") == "full"
            and r.get("eval_mode") == "cross_city"
            and r.get("backbone") in FROZEN_BACKBONES
        ]
    by_bb: dict[str, list[float]] = {}
    for r in src:
        bb = r.get("backbone", "")
        entry = r.get("results_per_target", {}).get(target_col, {})
        auroc = entry.get("auroc")
        if auroc is not None:
            by_bb.setdefault(bb, []).append(auroc)
    return {bb: float(np.mean(vals)) for bb, vals in by_bb.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-ceiling analysis: κ → AUROC upper bound.")
    p.add_argument("--cities", nargs="+", default=["seattle", "pittsburgh"],
                   help="Cities to include (must have entries in label_noise_audit.json).")
    p.add_argument("--match-radius-m", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    model_ps = json.loads(MODEL_PS_JSON.read_text(encoding="utf-8")) if MODEL_PS_JSON.exists() else {}

    output: dict = {"match_radius_m": args.match_radius_m, "by_city_target": {}}

    for city in args.cities:
        city_data = audit.get("by_city", {}).get(city)
        if city_data is None or city_data.get("skipped"):
            logging.warning("[%s] Not in label_noise_audit.json or skipped — skip.", city)
            continue

        for target_short, target_col in TARGET_COLS.items():
            entry = city_data.get(f"{target_short}_present")
            if entry is None or np.isnan(entry.get("kappa", float("nan"))):
                logging.warning("[%s/%s] No valid κ — skip.", city, target_short)
                continue

            kappa = entry["kappa"]
            p_o = entry["overture_prevalence"]
            p_s = entry["ps_prevalence"]
            auroc_o_vs_ps = entry["auroc_overture_vs_ps"]   # empirical: Overture predicts PS
            auroc_ps_vs_o = entry["auroc_ps_vs_overture"]   # empirical: PS predicts Overture

            ceiling_theoretical = auroc_ps_vs_overture_from_kappa(kappa, p_o, p_s)

            # Model AUROC vs Overture (observed, from results JSONL)
            model_vs_o = load_model_auroc_vs_overture(target_col)
            model_vs_o_mean = float(np.mean(list(model_vs_o.values()))) if model_vs_o else float("nan")

            # Model AUROC vs PS from model_vs_ps.json (Seattle only, pre-retrain)
            model_vs_ps_vals = [
                rec["auroc_vs_ps_mean"]
                for rec in model_ps.get("summary", [])
                if rec.get("target") == target_col
                and rec.get("backbone") in FROZEN_BACKBONES
            ]
            model_vs_ps_mean = float(np.mean(model_vs_ps_vals)) if model_vs_ps_vals else float("nan")

            key = f"{city}/{target_short}"
            output["by_city_target"][key] = {
                "city": city,
                "target": target_short,
                "n": entry["n"],
                "kappa": kappa,
                "overture_prevalence": p_o,
                "ps_prevalence": p_s,
                "auroc_overture_vs_ps": auroc_o_vs_ps,
                "auroc_ps_vs_overture_empirical": auroc_ps_vs_o,
                "auroc_ps_vs_overture_theoretical": ceiling_theoretical,
                "model_auroc_vs_overture_mean": model_vs_o_mean,
                "model_auroc_vs_overture_per_backbone": model_vs_o,
                "model_auroc_vs_ps_mean": model_vs_ps_mean,
                "interpretation": (
                    "Model AUROC vs Overture exceeds ceiling — fitting Overture-specific noise."
                    if model_vs_o_mean > ceiling_theoretical + 0.02
                    else "Model AUROC vs Overture is at or below the label-ceiling — near-signal regime."
                ),
            }
            logging.info(
                "[%s/%s] κ=%.3f ceiling=%.3f model_vs_o=%.3f model_vs_ps=%.3f",
                city, target_short, kappa, ceiling_theoretical, model_vs_o_mean, model_vs_ps_mean,
            )

    out_path = REPORT / "label_ceiling_analysis.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logging.info("Wrote: %s", out_path)

    _plot(output)
    return 0


def _plot(output: dict) -> None:
    """One figure, two panels (crosswalk / sidewalk)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    titles = {"crosswalk": "Crosswalk", "sidewalk": "Sidewalk"}
    colors = {"seattle": "#1f77b4", "pittsburgh": "#ff7f0e", "msp": "#2ca02c", "dc": "#d62728"}
    markers = ["o", "s", "^", "D"]

    for ax, target_short in zip(axes, ["crosswalk", "sidewalk"]):
        # Collect entries for this target
        entries = [
            v for v in output["by_city_target"].values()
            if v["target"] == target_short and not np.isnan(v["kappa"])
        ]
        if not entries:
            ax.set_visible(False)
            continue

        # Theoretical ceiling curve at reference marginals (first available city)
        ref = entries[0]
        kappas, aurocs = theoretical_curve(ref["overture_prevalence"], ref["ps_prevalence"])
        ax.plot(kappas, aurocs, "k--", linewidth=1.2, label="Ceiling: AUROC(PS→Overture)", zorder=1)

        # Data points
        for i, entry in enumerate(entries):
            city = entry["city"]
            k = entry["kappa"]
            c = colors.get(city, "#7f7f7f")
            mk = markers[i % len(markers)]
            # Empirical ceiling (Overture as PS predictor)
            ax.scatter([k], [entry["auroc_ps_vs_overture_empirical"]], color=c, marker=mk,
                       s=70, zorder=3, label=f"{city.title()} ceiling (empirical)")
            # Observed model AUROC vs Overture
            if not np.isnan(entry["model_auroc_vs_overture_mean"]):
                ax.scatter([k], [entry["model_auroc_vs_overture_mean"]], color=c, marker=mk,
                           s=70, facecolors="none", linewidths=1.5, zorder=3,
                           label=f"{city.title()} model vs Overture")
            # Model AUROC vs PS (if available)
            if not np.isnan(entry["model_auroc_vs_ps_mean"]):
                ax.scatter([k], [entry["model_auroc_vs_ps_mean"]], color=c, marker="x",
                           s=60, zorder=3, linewidths=1.5,
                           label=f"{city.title()} model vs PS")

        ax.set_xlabel("Cohen's κ (Overture ↔ PS)", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(titles[target_short], fontsize=11)
        ax.set_xlim(-0.05, 1.0)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Label-ceiling analysis: inter-source agreement bounds model performance",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "label_ceiling.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote: %s", out)


if __name__ == "__main__":
    raise SystemExit(main())
