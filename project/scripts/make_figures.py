"""Generate paper figures for WalkCLIP v2.

Figure 1: framework diagram (matplotlib boxes + arrows).
Figure 3: case study panel (4-heading Mapillary + NAIP) for three locations.
         Placeholder grid if specific case-study points have not been selected.

Output: paper/figures/fig1_framework.pdf, paper/figures/fig3_cases.pdf

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLOR_INPUT = "#E8F1F8"
COLOR_VIS = "#D6E8F5"
COLOR_PROC = "#FBE9D0"
COLOR_FUSE = "#E4D5F1"
COLOR_OUT = "#D7EBD0"
EDGE = "#333333"
TEXT = "#111111"


def add_box(ax, xy, w, h, text, fc, fs=8, weight="normal"):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.9, edgecolor=EDGE, facecolor=fc,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=TEXT, fontweight=weight, wrap=True)


def arrow(ax, xy1, xy2, lw=0.9):
    a = FancyArrowPatch(
        xy1, xy2, arrowstyle="->", mutation_scale=10,
        linewidth=lw, color=EDGE, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def make_framework():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.6, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Inputs (left column) — distributed evenly to leave clear margins
    add_box(ax, (0.2, 7.4), 2.8, 1.0, "Mapillary\nstreet-view (×4)", COLOR_INPUT, fs=8, weight="bold")
    add_box(ax, (0.2, 5.9), 2.8, 1.0, "NAIP aerial\n(0.6 m)",        COLOR_INPUT, fs=8, weight="bold")
    add_box(ax, (0.2, 4.4), 2.8, 1.0, "GPT-4o-mini\nstructured captions", COLOR_INPUT, fs=8, weight="bold")
    add_box(ax, (0.2, 2.9), 2.8, 1.0, "Overture wedge\nlabels (per heading)", COLOR_INPUT, fs=8, weight="bold")
    add_box(ax, (0.2, 1.4), 2.8, 1.0, "Pedestrian-graph\nkNN (Overture)",  COLOR_INPUT, fs=8, weight="bold")

    # Stage 1: frozen encoder
    add_box(ax, (3.6, 6.0), 3.2, 2.4,
            "Stage 1\nFrozen SigLIP-SO400M\n(or CLIP ViT-L/14)\n+ optional LoRA",
            COLOR_VIS, fs=8, weight="bold")

    # Stage 2: wedge supervision
    add_box(ax, (3.6, 3.4), 3.2, 1.8,
            "Stage 2\nHeading-conditioned\nwedge supervision\n(50 m sidewalk, 15 m crosswalk)",
            COLOR_PROC, fs=7.8)

    # Caption channel
    add_box(ax, (3.6, 1.4), 3.2, 1.4,
            "Caption embedding\nPCA-128", COLOR_VIS, fs=8)

    # Stage 2 -> Stage 1 (LoRA gradient path)
    a = FancyArrowPatch((5.2, 5.2), (5.2, 6.0), arrowstyle="->",
                        mutation_scale=10, linewidth=0.9,
                        linestyle=(0, (3, 2)), color=EDGE)
    ax.add_patch(a)

    # PCA box (between encoder and fusion)
    add_box(ax, (7.4, 6.4), 1.8, 1.6,
            "PCA-128\nper modality\n(fit on src city)",
            COLOR_PROC, fs=7.8)

    # Stage 3: Fusion
    add_box(ax, (9.8, 4.2), 3.0, 2.0,
            "Stage 3\nFusion\n$h_i$ = [street; NAIP;\ncaption; ped-graph]",
            COLOR_FUSE, fs=8, weight="bold")

    # Stage 4: Prediction head
    add_box(ax, (9.8, 1.6), 3.0, 2.0,
            "Stage 4\nShared-trunk MLP\n256$\\to$128$\\to$64\n+ per-target heads",
            COLOR_FUSE, fs=8)

    # Stage 5: logit ensemble
    add_box(ax, (13.2, 5.0), 2.6, 1.6,
            "Stage 5\nLogit ensemble\n(SigLIP + CLIP +\nLoRA-LOCO v2)",
            COLOR_FUSE, fs=7.5, weight="bold")

    # Outputs (right edge)
    add_box(ax, (13.2, 0.6), 2.6, 0.7, "Sidewalk AUROC",       COLOR_OUT, fs=7.5)
    add_box(ax, (13.2, 1.4), 2.6, 0.7, "Crosswalk AUROC",      COLOR_OUT, fs=7.5)
    add_box(ax, (13.2, 2.2), 2.6, 0.7, "Intersection $\\rho$", COLOR_OUT, fs=7.5)
    add_box(ax, (13.2, 3.0), 2.6, 0.7, "Building-frac $\\rho$",COLOR_OUT, fs=7.5)
    add_box(ax, (13.2, 3.8), 2.6, 0.7, "Walk index $\\rho$",   COLOR_OUT, fs=7.5, weight="bold")

    # Arrows: inputs -> Stage 1 (top three) and Stage 2 / Caption / Fusion (bottom two)
    arrow(ax, (3.0, 7.9), (3.6, 7.5))     # Mapillary -> Stage 1
    arrow(ax, (3.0, 6.4), (3.6, 7.0))     # NAIP -> Stage 1
    arrow(ax, (3.0, 4.9), (3.6, 2.1))     # captions -> Caption embedding
    arrow(ax, (3.0, 3.4), (3.6, 4.3))     # Overture wedge labels -> Stage 2
    arrow(ax, (3.0, 1.9), (9.8, 4.5))     # pedgraph -> Fusion (skips Stage 1)

    # Stage 1 -> PCA -> Fusion
    arrow(ax, (6.8, 7.2), (7.4, 7.2))
    arrow(ax, (9.2, 7.2), (9.8, 5.8))
    # Caption embedding -> Fusion
    arrow(ax, (6.8, 2.1), (9.8, 4.5))
    # Fusion -> Stage 4
    arrow(ax, (11.3, 4.2), (11.3, 3.6))
    # Stage 4 -> Stage 5 (ensemble)
    arrow(ax, (12.8, 2.8), (13.2, 5.2))
    # Stage 4 -> outputs (single-backbone direct path)
    for y_to in (0.95, 1.75, 2.55, 3.35, 4.15):
        arrow(ax, (12.8, 2.6), (13.2, y_to), lw=0.4)

    # Title strip
    ax.text(8.0, 9.30, "WalkCLIP v2 framework",
            ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(8.0, 8.85,
            "Frozen vision-language backbone, heading-conditioned supervision, pedestrian-graph context, multi-backbone ensemble.",
            ha="center", va="center", fontsize=7.8, style="italic", color="#555555")

    # Legend (bottom-left, in the free margin under the inputs column)
    ax.plot([0.4, 1.0], [-0.25, -0.25], color=EDGE, lw=0.9,
            linestyle=(0, (3, 2)))
    ax.text(1.1, -0.25, "Optional LoRA gradient (Stage 2 supervises Stage 1)",
            fontsize=7.0, va="center", color="#444444")

    out_pdf = FIG_DIR / "fig1_framework.pdf"
    out_png = FIG_DIR / "fig1_framework.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def make_cases_placeholder():
    """Placeholder case-study figure. Replace with real Mapillary+NAIP composites."""
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.0), gridspec_kw={"width_ratios": [3, 1]})
    titles = [
        ("(a) Case 1 -- MSP downtown", "Walk idx ref = +1.96\nEnsemble = +1.72"),
        ("(b) Case 2 -- Seattle Capitol Hill", "Walk idx ref = +0.42\nEnsemble = +0.51"),
        ("(c) Case 3 -- DC suburban edge", "Walk idx ref = -1.31\nEnsemble = -1.25"),
    ]
    for i, ((title, info), (ax_l, ax_r)) in enumerate(zip(titles, axes)):
        ax_l.set_facecolor("#F5F5F5")
        ax_l.text(0.5, 0.5, "4 x Mapillary headings\n(placeholder)",
                  ha="center", va="center", fontsize=9, color="#666666")
        ax_l.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_r.set_facecolor("#EEF3F8")
        ax_r.text(0.5, 0.5, "NAIP\n(placeholder)\n\n" + info,
                  ha="center", va="center", fontsize=8, color="#444444")
        ax_r.set_xticks([])
        ax_r.set_yticks([])
    fig.suptitle("Case-study locations (placeholder grid; replace with real images)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf = FIG_DIR / "fig3_cases.pdf"
    out_png = FIG_DIR / "fig3_cases.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def main() -> int:
    make_framework()
    # The case-study figure is generated by make_case_study_v2.py against real
    # imagery + prediction bundles. Do not regenerate the placeholder here, since
    # it would overwrite the real fig3_cases.{pdf,png}.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
