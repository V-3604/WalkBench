"""Build Figure 3: real case-study panel.

For each of three cities (MSP, Seattle, DC), pick one point at a chosen
walkability-index quantile that has all four Mapillary headings + NAIP on disk,
and compose its 4 street-view tiles + NAIP into a 3-row figure with model-vs-
reference annotations from the cross-city ensemble predictions.

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\make_case_study_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = REPO_ROOT / "project" / "data" / "processed" / "labels" / "walkability_index.csv"
IMG_ROOT = REPO_ROOT / "project" / "data" / "raw" / "imagery"
PRED_DIR = REPO_ROOT / "project" / "artifacts" / "predictions"
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CITY_DIRNAMES = {"msp": "Minneapolis", "seattle": "Seattle", "dc": "DC"}
CITY_LABELS = {"msp": "MSP", "seattle": "Seattle", "dc": "DC"}
QUANTILES = {"msp": 0.95, "seattle": 0.55, "dc": 0.05}


def has_all_images(city: str, point_id: int) -> bool:
    sv_dir = IMG_ROOT / "street_view_mapillary" / CITY_DIRNAMES[city]
    naip_dir = IMG_ROOT / "aerial_naip" / CITY_DIRNAMES[city]
    for h in (0, 90, 180, 270):
        if not (sv_dir / f"{point_id}_{h}.jpg").exists():
            return False
    return (naip_dir / f"{point_id}.jpg").exists()


def pick_point(city: str, df: pd.DataFrame, q: float) -> int:
    sub = df[df["city"] == city].copy().sort_values("walkability_index_v1").reset_index(drop=True)
    idx = int(round(q * (len(sub) - 1)))
    window = 200
    lo, hi = max(0, idx - window), min(len(sub), idx + window)
    cand = sub.iloc[lo:hi]
    target_z = sub.iloc[idx]["walkability_index_v1"]
    cand = cand.assign(d=(cand["walkability_index_v1"] - target_z).abs()).sort_values("d")
    for _, row in cand.iterrows():
        pid = int(row.point_id)
        if has_all_images(city, pid):
            return pid
    raise RuntimeError(f"No candidate with full imagery for {city} at q={q}")


def load_pred_for_point(city: str, point_id: int) -> dict[str, float]:
    """Collect ensemble logit for walkability_index_v1 by averaging available
    cross-city predictions where this point appears as test."""
    logits: list[float] = []
    y_ref: float | None = None
    for path in PRED_DIR.glob(f"*__cross_city__*_to_{city}__tier1__full__seed42.npz"):
        d = np.load(path)
        pids = d["point_ids"]
        if point_id in pids:
            i = int(np.where(pids == point_id)[0][0])
            logits.append(float(d["logit__walkability_index_v1"][i]))
            y_ref = float(d["y__walkability_index_v1"][i])
    return {
        "ensemble_pred": float(np.mean(logits)) if logits else float("nan"),
        "reference": y_ref if y_ref is not None else float("nan"),
        "n_backbones": len(logits),
    }


def make_figure():
    df = pd.read_csv(INDEX_CSV)
    cases: list[dict] = []
    for city, q in QUANTILES.items():
        pid = pick_point(city, df, q)
        pred = load_pred_for_point(city, pid)
        cases.append({"city": city, "point_id": pid, **pred})
        print(f"{city}: pid={pid} pred={pred['ensemble_pred']:+.2f} ref={pred['reference']:+.2f}")

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.05], wspace=0.05, hspace=0.18)

    titles = [
        f"(a) {CITY_LABELS['msp']} -- high walkability (q=0.95)",
        f"(b) {CITY_LABELS['seattle']} -- mid walkability (q=0.55)",
        f"(c) {CITY_LABELS['dc']} -- low walkability (q=0.05)",
    ]

    for row, (case, title) in enumerate(zip(cases, titles)):
        city = case["city"]
        pid = case["point_id"]
        sv_dir = IMG_ROOT / "street_view_mapillary" / CITY_DIRNAMES[city]
        naip_dir = IMG_ROOT / "aerial_naip" / CITY_DIRNAMES[city]
        for col, h in enumerate((0, 90, 180, 270)):
            ax = fig.add_subplot(gs[row, col])
            img = Image.open(sv_dir / f"{pid}_{h}.jpg").convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{h}deg", fontsize=8)
            if col == 0:
                ax.set_ylabel(title, fontsize=8.5, rotation=90, labelpad=4)
        ax = fig.add_subplot(gs[row, 4])
        naip = Image.open(naip_dir / f"{pid}.jpg").convert("RGB")
        ax.imshow(naip)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("NAIP", fontsize=8)
        info = f"ref = {case['reference']:+.2f}\npred = {case['ensemble_pred']:+.2f}"
        ax.text(
            1.02, 0.5, info, transform=ax.transAxes, fontsize=8,
            va="center", ha="left", family="monospace",
        )

    fig.suptitle(
        "Case-study locations: 4 Mapillary headings + NAIP, ensemble walkability-index prediction vs reference",
        fontsize=9, y=0.995,
    )
    out_pdf = FIG_DIR / "fig3_cases.pdf"
    out_png = FIG_DIR / "fig3_cases.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    meta = {"cases": cases}
    (FIG_DIR / "fig3_cases_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    raise SystemExit(make_figure())
