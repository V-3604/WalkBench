"""Figure 3 (Case Study C): one MSP point where the ensemble tracks the
reference tightly, one Seattle point near the model's worst-case miss, and one
DC residential-block point where the model misses on the low end. Uses real
Mapillary headings and NAIP. Filters to points that have all imagery on disk.

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\make_case_study_v2.py
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


def has_all_images(city: str, pid: int) -> bool:
    sv = IMG_ROOT / "street_view_mapillary" / CITY_DIRNAMES[city]
    nv = IMG_ROOT / "aerial_naip" / CITY_DIRNAMES[city]
    return all((sv / f"{pid}_{h}.jpg").exists() for h in (0, 90, 180, 270)) and (nv / f"{pid}.jpg").exists()


def collect_predictions(city: str) -> pd.DataFrame:
    """Mean ensemble logit per point on walkability_index_v1, averaging the
    two LOCO source cities and three seeds for SigLIP + CLIP."""
    rows: dict[int, dict] = {}
    for path in PRED_DIR.glob(f"*__cross_city__*_to_{city}__tier1__full__seed*.npz"):
        d = np.load(path)
        if "logit__walkability_index_v1" not in d:
            continue
        for pid, pred, y in zip(d["point_ids"], d["logit__walkability_index_v1"], d["y__walkability_index_v1"]):
            pid = int(pid)
            r = rows.setdefault(pid, {"point_id": pid, "preds": [], "y": float(y)})
            r["preds"].append(float(pred))
    df = pd.DataFrame([
        {"point_id": r["point_id"], "ensemble_pred": float(np.mean(r["preds"])),
         "reference": r["y"], "n_runs": len(r["preds"])}
        for r in rows.values()
    ])
    df["abs_err"] = (df["ensemble_pred"] - df["reference"]).abs()
    return df


def pick_msp_good_track(df: pd.DataFrame) -> int:
    """High-walkability point with the tightest pred-vs-ref match."""
    cand = df[df["reference"] > 1.0].sort_values("abs_err")
    for _, row in cand.iterrows():
        pid = int(row.point_id)
        if has_all_images("msp", pid):
            return pid
    raise RuntimeError("no candidate")


def pick_seattle_worst_miss(df: pd.DataFrame) -> int:
    """Largest absolute error among points with at least moderate reference."""
    cand = df[(df["reference"].abs() > 0.5)].sort_values("abs_err", ascending=False)
    for _, row in cand.iterrows():
        pid = int(row.point_id)
        if has_all_images("seattle", pid):
            return pid
    raise RuntimeError("no candidate")


def pick_dc_suburban_miss(df: pd.DataFrame) -> int:
    """Low-reference residential-block point with a moderate upward miss.
    Skip extreme outliers (|ref| > 4) which often pair with blown-out / dark
    Mapillary frames; pick a clearly-below-mean point in the [-2.5, -1.0] band
    that has the largest still-moderate upward miss."""
    cand = df[(df["reference"] < -1.0) & (df["reference"] > -3.0)].copy()
    cand["upward_miss"] = cand["ensemble_pred"] - cand["reference"]
    cand = cand[cand["upward_miss"] > 0.6]
    cand = cand.sort_values("upward_miss", ascending=False)
    for _, row in cand.iterrows():
        pid = int(row.point_id)
        if has_all_images("dc", pid):
            return pid
    raise RuntimeError("no candidate")


def render():
    cases: list[dict] = []
    pickers = [
        ("msp", "good track", pick_msp_good_track),
        ("seattle", "worst-case miss", pick_seattle_worst_miss),
        ("dc", "residential-block miss", pick_dc_suburban_miss),
    ]
    for city, label, picker in pickers:
        df = collect_predictions(city)
        pid = picker(df)
        row = df[df["point_id"] == pid].iloc[0]
        cases.append({
            "city": city, "label": label, "point_id": pid,
            "reference": float(row.reference),
            "ensemble_pred": float(row.ensemble_pred),
            "abs_err": float(row.abs_err),
            "n_runs": int(row.n_runs),
        })
        print(f"{city} ({label}): pid={pid} ref={row.reference:+.2f} pred={row.ensemble_pred:+.2f} |err|={row.abs_err:.2f}")

    fig = plt.figure(figsize=(9.0, 6.6))
    gs = fig.add_gridspec(
        3, 5,
        width_ratios=[1, 1, 1, 1, 1.05],
        left=0.04, right=0.83, top=0.86, bottom=0.03,
        wspace=0.06, hspace=0.45,
    )
    row_titles = [
        f"(a) {CITY_LABELS['msp']}: tight match",
        f"(b) {CITY_LABELS['seattle']}: high-tail miss",
        f"(c) {CITY_LABELS['dc']}: low-tail miss",
    ]
    for row_i, (case, row_title) in enumerate(zip(cases, row_titles)):
        city = case["city"]
        pid = case["point_id"]
        sv = IMG_ROOT / "street_view_mapillary" / CITY_DIRNAMES[city]
        nv = IMG_ROOT / "aerial_naip" / CITY_DIRNAMES[city]
        row_left_ax = None
        for col, h in enumerate((0, 90, 180, 270)):
            ax = fig.add_subplot(gs[row_i, col])
            if col == 0:
                row_left_ax = ax
            ax.imshow(Image.open(sv / f"{pid}_{h}.jpg").convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[row_i, 4])
        ax.imshow(Image.open(nv / f"{pid}.jpg").convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        info = (
            f"ref  = {case['reference']:+.3f}\n"
            f"pred = {case['ensemble_pred']:+.3f}\n"
            f"|err|= {case['abs_err']:.3f}"
        )
        ax.text(1.05, 0.5, info, transform=ax.transAxes, fontsize=9,
                va="center", ha="left", family="monospace")
        # Row title placed above the leftmost subplot, in figure-relative coords
        # so it never collides with subplot contents.
        bbox = row_left_ax.get_position()
        fig.text(bbox.x0, bbox.y1 + 0.012, row_title,
                 fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    fig.suptitle(
        "Case-study points: cross-city ensemble prediction vs. held-out walkability index $z_i$. "
        "Each row shows one point: four Mapillary street-view headings (0°, 90°, 180°, 270°) and the matched NAIP aerial tile; "
        "right-edge text gives reference value, ensemble prediction, and absolute error.",
        fontsize=8.5, y=0.97, wrap=True,
    )
    out_pdf = FIG_DIR / "fig3_cases.pdf"
    out_png = FIG_DIR / "fig3_cases.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    (FIG_DIR / "fig3_cases_meta.json").write_text(json.dumps({"cases": cases}, indent=2))


if __name__ == "__main__":
    raise SystemExit(render())
