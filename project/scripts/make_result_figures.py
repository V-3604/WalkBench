"""Generate WalkBench result figures directly from the locked aggregate.

Every number is read at run time from
``project/artifacts/reports/bootstrap_summary.json`` (the seed-averaged file with
per-cell mean / std / bootstrap CI). Nothing is hard-coded, so the figures cannot
drift from the results. Re-run this after any results change and the PDFs update.

Each figure is one function; edit the function (colors, order, labels, size) and
re-run. Outputs go to ``project/artifacts/reports/figures/`` as PDF (vector, for the
paper), PNG (300 dpi, for preview / WhatsApp), and SVG (text-as-text, editable in
Inkscape / Illustrator / Figma).

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\make_result_figures.py
    & ".venv\\Scripts\\python.exe" project\\scripts\\make_result_figures.py --only transfer
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "project" / "artifacts" / "reports"
SUMMARY_PATH = REPORTS / "bootstrap_summary.json"
LABEL_CEILING_PATH = REPORTS / "label_ceiling_analysis.json"
FIG_DIR = REPORTS / "figures"

# Grayscale palette (print-safe). Series are distinguished by shade + a thin
# black edge on every bar. Titles live in the LaTeX \caption, not in the figure.
INK = "#000000"
GRAY_DARK = "#3A3A3A"
GRAY_MID = "#7F7F7F"
GRAY_LIGHT = "#BFBFBF"
GREY = "#666666"
# Light-to-dark ramp for the 5-backbone Pittsburgh figure. Even ~50-step jumps
# so each series is clearly distinct in print.
GRAY5 = ["#E2E2E2", "#B0B0B0", "#7E7E7E", "#4C4C4C", "#1A1A1A"]
EDGE = {"edgecolor": "black", "linewidth": 0.5}

LOCO_DIRECTIONS = {
    "dc->msp", "dc->seattle", "msp->dc",
    "msp->seattle", "seattle->dc", "seattle->msp",
}

# Pretty labels.
BACKBONE_LABEL = {
    "siglip": "SigLIP",
    "siglip2": "SigLIP 2",
    "clip": "CLIP",
    "dinov2": "DINOv2",
    "siglip2+remoteclip": "SigLIP 2 +\nRemoteCLIP",
    "siglip_lora": "SigLIP",
    "siglip_lora_loco": "SigLIP",
    "siglip_lora_loco_v2": "SigLIP",
}
TARGET_LABEL = {
    "overture_sidewalk_present": "Sidewalk\n(AUROC)",
    "overture_crosswalk_present": "Crosswalk\n(AUROC)",
    "overture_intersection_count_200m": "Intersection\n($\\rho$)",
    "overture_building_footprint_frac_100m": "Building frac.\n($\\rho$)",
    "walkability_index_v2": "Walk index\n($\\rho$)",
}
# Which metric is the headline for each target.
TARGET_METRIC = {
    "overture_sidewalk_present": "auroc",
    "overture_crosswalk_present": "auroc",
    "overture_intersection_count_200m": "spearman_rho",
    "overture_building_footprint_frac_100m": "spearman_rho",
    "walkability_index_v2": "spearman_rho",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,      # embed TrueType so the PDF is portable
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text for vector editing
        "figure.dpi": 120,
    })


def load_rows() -> list[dict]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run bootstrap_cis.py to regenerate the "
            "seed-averaged summary before building figures."
        )
    return list(json.load(open(SUMMARY_PATH)).values())


def loco_cells(rows: list[dict], backbone: str, target: str, metric: str) -> list[float]:
    """Per-direction means for the 6 leave-one-city-out directions (ablation=full)."""
    out = {}
    for r in rows:
        if (r.get("ablation") == "full"
                and r.get("backbone") == backbone
                and r.get("target") == target
                and r.get("metric") == metric
                and r.get("direction") in LOCO_DIRECTIONS):
            out[r["direction"]] = r["mean"]
    return list(out.values())


def loco_mean_std(rows, backbone, target, metric) -> tuple[float, float] | None:
    """Mean and SD over the LOCO directions, or None if this cell is absent from
    the locked summary (e.g. dinov2 3-city LOCO was never aggregated). We skip
    rather than invent a number."""
    vals = loco_cells(rows, backbone, target, metric)
    if not vals:
        return None
    return stats.mean(vals), (stats.pstdev(vals) if len(vals) > 1 else 0.0)


def pgh_cell(rows, backbone, target, metric) -> tuple[float, float, float]:
    """Pittsburgh zero-shot mean and bootstrap CI (lo, hi)."""
    for r in rows:
        if (r.get("ablation") == "full"
                and r.get("backbone") == backbone
                and r.get("target") == target
                and r.get("metric") == metric
                and "pittsburgh" in r.get("direction", "")):
            return r["mean"], r.get("ci_lo_mean", r["mean"]), r.get("ci_hi_mean", r["mean"])
    raise ValueError(f"no Pittsburgh cell for {backbone}/{target}/{metric}")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if ext == "png" else None)
        print(f"  wrote {path.relative_to(REPO_ROOT)}")
    plt.close(fig)


def fig_transfer(rows: list[dict]) -> None:
    """Cross-city LOCO transfer of the composite index, per frozen backbone."""
    backbones = ["siglip2", "siglip", "siglip2+remoteclip", "clip", "dinov2"]
    means, stds, labels = [], [], []
    for bb in backbones:
        res = loco_mean_std(rows, bb, "walkability_index_v2", "spearman_rho")
        if res is None:
            print(f"  skip {bb}: no 3-city LOCO walk-index cell in locked summary")
            continue
        m, s = res
        means.append(m); stds.append(s); labels.append(BACKBONE_LABEL[bb])

    order = np.argsort(means)[::-1]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, capsize=2.5, color=GRAY_MID, width=0.5,
           error_kw={"elinewidth": 0.8, "capthick": 0.8}, **EDGE)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.5)
    ax.set_ylabel("Walkability index Spearman $\\rho$")
    ax.set_ylim(0.0, 0.85)
    ax.axhline(0, color="black", lw=0.7)
    _save(fig, "fig_transfer")


def fig_finetune_collapse(rows: list[dict]) -> None:
    """HERO: in-corpus fine-tuning beats frozen only because it sees the test city.

    Drawn at true print size (~2.2 in wide) for inclusion at
    width=0.65\\columnwidth, so the fonts below render at face value on the
    page. Scaling it down further in LaTeX defeats that — adjust figsize here
    instead. Legend labels lean on the caption defining in-corpus and LOCO.
    """
    targets = ["overture_sidewalk_present", "overture_crosswalk_present"]
    conditions = [
        ("siglip", "Frozen", GRAY_LIGHT),
        ("siglip_lora", "Fine-tuned (in-corpus)", GRAY_DARK),
        ("siglip_lora_loco_v2", "Fine-tuned (LOCO)", GRAY_MID),
    ]
    fig, ax = plt.subplots(figsize=(2.2, 1.5))
    n_cond = len(conditions)
    group_w = 0.78
    bar_w = group_w / n_cond
    x = np.arange(len(targets))
    for j, (bb, label, color) in enumerate(conditions):
        ms, ss = [], []
        for tgt in targets:
            res = loco_mean_std(rows, bb, tgt, "auroc")
            if res is None:
                raise ValueError(
                    f"collapse figure needs {bb}/{tgt}/auroc but it is missing "
                    "from the locked summary")
            m, s = res
            ms.append(m); ss.append(s)
        offset = (j - (n_cond - 1) / 2) * bar_w
        bars = ax.bar(x + offset, ms, bar_w * 0.9, yerr=ss, capsize=1.8,
                      color=color, label=label, **EDGE,
                      error_kw={"elinewidth": 0.7, "capthick": 0.7})
        for b, m, s in zip(bars, ms, ss):
            ax.text(b.get_x() + b.get_width() / 2, m + s + 0.012, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=4.5)
    # Chance reference at AUROC 0.5: thin dashed line, no inline label.
    # The caption states what it is (standard convention).
    ax.axhline(0.5, color=GRAY_MID, lw=0.7, ls=(0, (4, 2)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sidewalk", "Crosswalk"], fontsize=8)
    ax.set_ylabel("Cross-city AUROC", fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), frameon=False,
              ncol=1, fontsize=6.8, handlelength=0.9, handletextpad=0.4,
              labelspacing=0.2, borderaxespad=0.0)
    _save(fig, "fig_finetune_collapse")


def fig_pittsburgh(rows: list[dict]) -> None:
    """Pittsburgh zero-shot: per target, every frozen backbone, with bootstrap CIs.

    Drawn at true print size (~4.9 in wide) for inclusion at
    width=0.70\\textwidth — fonts render at face value on the page.
    """
    backbones = ["siglip2", "siglip", "clip", "dinov2", "siglip2+remoteclip"]
    colors = GRAY5
    targets = list(TARGET_METRIC.keys())

    fig, ax = plt.subplots(figsize=(4.9, 1.55))
    n_bb = len(backbones)
    group_w = 0.78
    bar_w = group_w / n_bb
    x = np.arange(len(targets))
    for j, (bb, color) in enumerate(zip(backbones, colors)):
        ms, lo_err, hi_err = [], [], []
        for tgt in targets:
            m, lo, hi = pgh_cell(rows, bb, tgt, TARGET_METRIC[tgt])
            ms.append(m); lo_err.append(m - lo); hi_err.append(hi - m)
        offset = (j - (n_bb - 1) / 2) * bar_w
        ax.bar(x + offset, ms, bar_w * 0.9, yerr=[lo_err, hi_err], capsize=1.2,
               color=color, label=BACKBONE_LABEL[bb].replace("\n", " "), **EDGE,
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABEL[t] for t in targets], fontsize=7)
    ax.set_ylabel("Zero-shot score\non Pittsburgh", fontsize=7.5)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower center", frameon=False, ncol=5, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 0.99),
              borderaxespad=0.0, fontsize=6.5)
    _save(fig, "fig_pittsburgh_zeroshot")


def fig_label_ceiling(_rows: list[dict]) -> None:
    """Why crosswalk transfers worst: the two label sources barely agree.

    Reads label_ceiling_analysis.json (NOT bootstrap_summary). Two panels of
    unambiguous facts only:
      A. Cohen's kappa between Overture (our training labels) and Project
         Sidewalk (independent expert labels), per (city, target).
      B. Positive-class prevalence under each source — the base-rate gap.

    Deliberately omits the kappa->AUROC ceiling value: its direction
    (AUROC(PS->Overture) vs AUROC(Overture->PS)) is still being settled with
    the advisor, so it stays out of the figure until confirmed.
    """
    if not LABEL_CEILING_PATH.exists():
        raise FileNotFoundError(
            f"{LABEL_CEILING_PATH} not found. Run label_ceiling_analysis.py first.")
    data = json.load(open(LABEL_CEILING_PATH))["by_city_target"]

    def nice(key: str) -> str:
        return (key.replace("/", "\n").replace("seattle", "Seattle")
                   .replace("pittsburgh", "Pittsburgh"))

    # Panel A shows kappa for crosswalk only. Sidewalk kappa looks low only
    # because ~98% of points have sidewalks (high-prevalence kappa artifact),
    # which would misread as "all labels bad" next to the real crosswalk
    # disagreement. Sidewalk still appears in Panel B, where it correctly shows
    # the sources agreeing.
    kappa_keys = [k for k in ["seattle/crosswalk"] if k in data]
    prev_pref = ["seattle/crosswalk", "seattle/sidewalk", "pittsburgh/sidewalk"]
    prev_keys = [k for k in prev_pref if k in data] + [k for k in data if k not in prev_pref]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   gridspec_kw={"wspace": 0.32, "width_ratios": [1, 2]})

    # Panel A: Cohen's kappa for crosswalk, against the Landis-Koch floor.
    xa = np.arange(len(kappa_keys))
    axA.axhline(0.20, color=INK, lw=0.7, ls=(0, (4, 2)))
    axA.text(0.0, 0.205, "'slight' agreement\nfloor ($\\kappa$ = 0.20)",
             fontsize=6, color=GREY, ha="center", va="bottom")
    axA.bar(xa, [data[k]["kappa"] for k in kappa_keys], width=0.45,
            color=GRAY_MID, **EDGE)
    for xi, k in zip(xa, kappa_keys):
        axA.text(xi, data[k]["kappa"] + 0.008, f"{data[k]['kappa']:.3f}",
                 ha="center", va="bottom", fontsize=8)
        axA.text(xi, 0.006, f"n={data[k]['n']:,}", ha="center", va="bottom",
                 fontsize=6, color="white")
    axA.set_xticks(xa); axA.set_xticklabels([nice(k) for k in kappa_keys], fontsize=7.5)
    axA.set_xlim(-0.7, 0.7)
    axA.set_ylabel("Cohen's $\\kappa$  (Overture vs Project Sidewalk)")
    axA.set_ylim(0, 0.40)

    # Panel B: prevalence under each source (the base-rate gap).
    xb = np.arange(len(prev_keys))
    bw = 0.36
    axB.bar(xb - bw / 2, [100 * data[k]["overture_prevalence"] for k in prev_keys],
            bw, color=GRAY_DARK, label="Overture (training)", **EDGE)
    axB.bar(xb + bw / 2, [100 * data[k]["ps_prevalence"] for k in prev_keys],
            bw, color=GRAY_LIGHT, label="Project Sidewalk (expert)", **EDGE)
    for xi, k in zip(xb, prev_keys):
        o = 100 * data[k]["overture_prevalence"]; p = 100 * data[k]["ps_prevalence"]
        axB.text(xi - bw / 2, o + 1.5, f"{o:.0f}%", ha="center", va="bottom", fontsize=6.4)
        axB.text(xi + bw / 2, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=6.4)
    axB.set_xticks(xb); axB.set_xticklabels([nice(k) for k in prev_keys], fontsize=7.5)
    axB.set_ylabel("Positive-class prevalence (%)")
    axB.set_ylim(0, 109)
    axB.legend(loc="upper center", frameon=False, fontsize=6.4,
               bbox_to_anchor=(0.5, -0.18), ncol=2, columnspacing=1.0)

    _save(fig, "fig_label_ceiling")


FIGURES = {
    "transfer": fig_transfer,
    "collapse": fig_finetune_collapse,
    "pittsburgh": fig_pittsburgh,
    "ceiling": fig_label_ceiling,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FIGURES), default=None,
                        help="build a single figure instead of all")
    args = parser.parse_args()

    set_style()
    rows = load_rows()
    targets = {args.only: FIGURES[args.only]} if args.only else FIGURES
    for name, fn in targets.items():
        print(f"[{name}]")
        fn(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
