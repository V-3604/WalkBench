"""
Auto-generate LaTeX result tables for WalkCLIP-3C paper.

Reads:
  project/artifacts/reports/multitarget_results.jsonl
  project/artifacts/reports/spatial_cv_results.jsonl  (if present)

Writes:
  project/artifacts/reports/tables_draft.tex
    - Table 2: Cross-city Tier 1 (per backbone)
    - Table 3: Modality ablation (per backbone, MSP -> Seattle)
    - Table 4: Backbone comparison (MSP -> Seattle)
    - Table 5: Spatial CV summary (5-fold KMeans)
  project/artifacts/reports/tables_provenance.json
    For every cell in every table: backbone, ablation, train/test city,
    timestamp, run_index in jsonl. So every number in the tex is auditable.
  project/artifacts/reports/multitarget_summary.json
    Real per-(backbone, direction, target) headline summary, NOT just the last
    appended run.

Backbone semantics:
  - "siglip"           : frozen SigLIP-SO400M (Google pretrained, no adaptation)
  - "siglip_lora"      : SigLIP-SO400M + LoRA, fine-tuned on union of all 3 cities
                         (in-corpus; cross-city numbers are NOT held-out at backbone)
  - "siglip_lora_loco" : virtual backbone composed of three LOCO adapters; for each
                         (src, dst) cross-city cell we accept records whose
                         backbone == f"siglip_lora_loco_{dst}" (the adapter that
                         held out the destination city). This IS held-out at backbone.
  - "clip"             : frozen CLIP ViT-L/14

Filtering:
  Records with n < MIN_N or with NaN in any reported metric are dropped (E5).
  When multiple records exist for the same (backbone, ablation, train, test)
  cell, the latest by timestamp is selected (E1).

Run:
    & ".venv\\Scripts\\python.exe" project/scripts/build_results_tables.py
    & ".venv\\Scripts\\python.exe" project/scripts/build_results_tables.py \\
        --require-backbones siglip,siglip_lora,clip
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "project/artifacts/reports"
MT_JSONL          = REPORT_DIR / "multitarget_results.jsonl"
CV_JSONL          = REPORT_DIR / "spatial_cv_results.jsonl"
DEFAULT_OUT       = REPORT_DIR / "tables_draft.tex"
DEFAULT_PROV      = REPORT_DIR / "tables_provenance.json"
DEFAULT_SUMMARY   = REPORT_DIR / "multitarget_summary.json"

CITIES = ["msp", "seattle", "dc"]
CITY_LABELS = {"msp": "MSP", "seattle": "Seattle", "dc": "DC"}

BACKBONE_LABELS = {
    "siglip":           "SigLIP-SO400M (frozen)",
    "siglip_lora":      "SigLIP-SO400M + LoRA (in-corpus)",
    "siglip_lora_loco": "SigLIP-SO400M + LoRA (held-out city)",
    "clip":             "CLIP ViT-L/14 (frozen)",
}
BACKBONE_ORDER = ["siglip", "siglip_lora_loco", "siglip_lora", "clip"]

SRC_BACKBONE_LABELS = {
    "siglip2":              "SigLIP-2 SO400m (frozen)",
    "dinov2":               "DINOv2 ViT-L/14 (frozen)",
    "siglip2+remoteclip":   "SigLIP-2 street + RemoteCLIP NAIP (frozen)",
}
# All backbones used in Pittsburgh zero-shot table (train=dc+msp+seattle → test=pittsburgh)
PITTSBURGH_BACKBONE_ORDER = [
    "siglip", "clip", "siglip2", "dinov2", "siglip2+remoteclip",
    "siglip_lora_loco_v2_pittsburgh",
]
PITTSBURGH_BACKBONE_LABELS = {
    **BACKBONE_LABELS,
    **SRC_BACKBONE_LABELS,
    "siglip_lora_loco_v2_pittsburgh": "SigLIP-SO400M + LoRA-LOCO-v2 (Pittsburgh held-out)",
}

TIER1_BINARY = ["overture_sidewalk_present", "overture_crosswalk_present"]
TIER1_REGR   = ["overture_intersection_count_200m", "overture_building_footprint_frac_100m"]
TIER1_ALL    = TIER1_BINARY + TIER1_REGR

TARGET_LABELS = {
    "overture_sidewalk_present":              "Sidewalk",
    "overture_crosswalk_present":             "Crosswalk",
    "overture_intersection_count_200m":       "Intersect. density",
    "overture_building_footprint_frac_100m":  "Building frac.",
}

ABLATION_ORDER = [
    "vision_only", "street_only", "naip_only", "caption_only", "spatial_only",
    "tabular_only", "vision_tabular", "vision_spatial", "caption_spatial",
    "no_spatial", "full",
]
ABLATION_LABELS = {
    "vision_only":      "Vision (street+NAIP)",
    "street_only":      "Street only",
    "naip_only":        "NAIP only",
    "caption_only":     "Caption only",
    "spatial_only":     "Spatial only",
    "tabular_only":     "Tabular only",
    "vision_tabular":   "Vision+Tabular",
    "vision_spatial":   "Vision+Spatial",
    "caption_spatial":  "Caption+Spatial",
    "no_spatial":       "Full (no spatial)",
    "full":             "\\textbf{Full}",
}

MIN_N = 100  # E5: drop runs with too few eval rows


# ---------------------------------------------------------------------------
# Loading + filtering
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for run_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_run_index"] = run_index  # for provenance
                out.append(rec)
            except json.JSONDecodeError:
                continue
    return out


def is_finite(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def has_valid_metrics(rec: dict, targets: Iterable[str]) -> bool:
    """E5: drop a record if any reported target has n < MIN_N or NaN AUROC/rho."""
    rpt = rec.get("results_per_target", {})
    for t in targets:
        if t not in rpt:
            return False
        cell = rpt[t]
        if cell.get("n", 0) < MIN_N:
            return False
        if cell.get("type") == "binary":
            if not is_finite(cell.get("auroc")):
                return False
        else:
            if not is_finite(cell.get("spearman_rho")):
                return False
    return True


def filter_records(records: list[dict]) -> list[dict]:
    keep = []
    for r in records:
        if not has_valid_metrics(r, TIER1_ALL):
            continue
        keep.append(r)
    return keep


def get_metric(record: dict, target: str, metric: str) -> float:
    return record.get("results_per_target", {}).get(target, {}).get(metric, float("nan"))


def fmt(val: float, decimals: int = 3) -> str:
    if not is_finite(val):
        return "---"
    return f"{val:.{decimals}f}"


def bold_max_in_col(col: list[float]) -> list[str]:
    finite = [v for v in col if is_finite(v)]
    if not finite:
        return [fmt(v) for v in col]
    best = max(finite)
    return [f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v) for v in col]


# ---------------------------------------------------------------------------
# Backbone selection (with virtual LOCO backbone)
# ---------------------------------------------------------------------------

def _records_for_backbone(records: list[dict], backbone: str, *, src: str = None, dst: str = None) -> list[dict]:
    """Return records belonging to a (possibly virtual) backbone.

    For 'siglip_lora_loco' a record matches when its backbone tag is
    'siglip_lora_loco_{dst}' — i.e. the adapter that holds out the test city.
    Caller must pass dst; if dst is None, all loco records pass through.
    """
    if backbone == "siglip":
        return [r for r in records if r.get("backbone", "siglip") in ("siglip", "")]
    if backbone == "siglip_lora_loco":
        if dst is not None:
            target = f"siglip_lora_loco_{dst}"
            return [r for r in records if r.get("backbone") == target]
        return [r for r in records if str(r.get("backbone", "")).startswith("siglip_lora_loco_")]
    return [r for r in records if r.get("backbone") == backbone]


def latest_by_ts(records: list[dict]) -> dict | None:
    """E1: pick the latest run by timestamp (string comparison; ISO 8601 sorts correctly)."""
    if not records:
        return None
    return max(records, key=lambda x: x.get("timestamp", ""))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class Provenance:
    """Records (table, row, col, backbone, ablation, train, test, timestamp, run_index)."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, *, table: str, row: str, col: str, rec: dict | None,
            backbone: str, ablation: str, src: str | None, dst: str | None) -> None:
        if rec is None:
            self.entries.append({
                "table": table, "row": row, "col": col,
                "backbone": backbone, "ablation": ablation,
                "train_city": src, "test_city": dst,
                "timestamp": None, "run_index": None, "value": None,
            })
            return
        self.entries.append({
            "table": table, "row": row, "col": col,
            "backbone": rec.get("backbone", backbone),
            "ablation": rec.get("ablation", ablation),
            "train_city": rec.get("train_city", src),
            "test_city": rec.get("test_city", dst),
            "timestamp": rec.get("timestamp"),
            "run_index": rec.get("_run_index"),
        })

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Table 2: Cross-city Tier 1 (per backbone, including LOCO virtual)
# ---------------------------------------------------------------------------

def _table_cross_city_one_backbone(records: list[dict], backbone: str, prov: Provenance) -> str:
    pairs = [(s, d) for s in CITIES for d in CITIES if s != d]

    # For LOCO, the right adapter depends on dst, so collect per cell.
    has_any = False
    cell_recs: dict[tuple[str, str], dict | None] = {}
    for src, dst in pairs:
        bb_recs = _records_for_backbone(records, backbone, src=src, dst=dst)
        full_cross = [
            r for r in bb_recs
            if r.get("ablation") == "full"
            and r.get("eval_mode") == "cross_city"
            and r.get("target_tier") in ("1", "all")
            and r.get("train_city") == src and r.get("test_city") == dst
        ]
        chosen = latest_by_ts(full_cross)
        cell_recs[(src, dst)] = chosen
        if chosen is not None:
            has_any = True
    if not has_any:
        return f"% No full cross-city Tier 1 runs for backbone={backbone}.\n"

    label_suffix = backbone.replace("_", "-")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Cross-city Tier 1 results --- backbone: {BACKBONE_LABELS[backbone]}. "
        f"AUROC for binary, Spearman~$\\rho$ for regression. "
        f"Training city $\\to$ test city. Ablation: \\texttt{{full}} with pedgraph spatial features.}}",
        f"\\label{{tab:cross_city_tier1_{label_suffix}}}",
        "\\small",
        "\\begin{tabular}{ll" + "r" * len(TIER1_ALL) + "}",
        "\\toprule",
        "Train & Test & " + " & ".join(TARGET_LABELS[t] for t in TIER1_ALL) + " \\\\",
        "\\midrule",
    ]
    for src, dst in pairs:
        rec = cell_recs[(src, dst)]
        if rec is None:
            vals = ["---"] * len(TIER1_ALL)
        else:
            vals = []
            for t in TIER1_BINARY:
                vals.append(fmt(get_metric(rec, t, "auroc")))
            for t in TIER1_REGR:
                vals.append(fmt(get_metric(rec, t, "spearman_rho")))
        for col_idx, t in enumerate(TIER1_ALL):
            prov.add(table=f"cross_city_tier1::{backbone}",
                     row=f"{src}->{dst}", col=TARGET_LABELS[t],
                     rec=rec, backbone=backbone, ablation="full", src=src, dst=dst)
        lines.append(
            f"{CITY_LABELS[src]} & {CITY_LABELS[dst]} & " + " & ".join(vals) + " \\\\"
        )
        if dst == [d for s2, d in pairs if s2 == src][-1] and src != pairs[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_cross_city_tier1(records: list[dict], prov: Provenance) -> str:
    parts = []
    for bb in BACKBONE_ORDER:
        # Quick presence check first
        has = False
        for src in CITIES:
            for dst in CITIES:
                if src == dst:
                    continue
                if any(
                    r.get("ablation") == "full" and r.get("eval_mode") == "cross_city"
                    and r.get("train_city") == src and r.get("test_city") == dst
                    and r.get("target_tier") in ("1", "all")
                    for r in _records_for_backbone(records, bb, src=src, dst=dst)
                ):
                    has = True
                    break
            if has:
                break
        if has:
            parts.append(_table_cross_city_one_backbone(records, bb, prov))
    if not parts:
        return "% Table 2: No full cross-city Tier 1 runs found in JSONL.\n"
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Table 3: Modality ablation (per backbone, one direction)
# ---------------------------------------------------------------------------

def table_modality_ablation(records: list[dict], direction: tuple[str, str],
                            backbone: str, prov: Provenance) -> str:
    src, dst = direction
    bb_recs = _records_for_backbone(records, backbone, src=src, dst=dst)
    ablation_records = [
        r for r in bb_recs
        if r.get("eval_mode") == "cross_city"
        and r.get("train_city") == src and r.get("test_city") == dst
        and r.get("target_tier") in ("1", "all")
    ]
    if not ablation_records:
        return f"% Table 3: No ablation runs found for {src}->{dst} backbone={backbone}.\n"

    # E1: latest-by-timestamp per ablation
    by_abl: dict[str, list[dict]] = {}
    for r in ablation_records:
        by_abl.setdefault(r.get("ablation", ""), []).append(r)
    ablation_latest = {a: latest_by_ts(rs) for a, rs in by_abl.items()}

    present_ablations = [a for a in ABLATION_ORDER if a in ablation_latest]
    if not present_ablations:
        return f"% Table 3: ablations missing for backbone={backbone}.\n"

    label_suffix = backbone.replace("_", "-")
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Modality ablation ({CITY_LABELS[src]}$\\to${CITY_LABELS[dst]}, Tier~1, "
        f"backbone: {BACKBONE_LABELS[backbone]}). "
        f"AUROC for sidewalk / crosswalk; Spearman~$\\rho$ for intersection~density / building~fraction.}}",
        f"\\label{{tab:ablation_{label_suffix}}}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(TIER1_ALL) + "}",
        "\\toprule",
        "Ablation & " + " & ".join(TARGET_LABELS[t] for t in TIER1_ALL) + " \\\\",
        "\\midrule",
    ]
    col_vals: dict[str, list[float]] = {t: [] for t in TIER1_ALL}
    for abl in present_ablations:
        r = ablation_latest[abl]
        for t in TIER1_BINARY:
            col_vals[t].append(get_metric(r, t, "auroc"))
        for t in TIER1_REGR:
            col_vals[t].append(get_metric(r, t, "spearman_rho"))
    col_bold = {t: bold_max_in_col(col_vals[t]) for t in TIER1_ALL}

    for i, abl in enumerate(present_ablations):
        rec = ablation_latest[abl]
        label = ABLATION_LABELS.get(abl, abl)
        vals = [col_bold[t][i] for t in TIER1_ALL]
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
        for t in TIER1_ALL:
            prov.add(table=f"ablation::{backbone}",
                     row=abl, col=TARGET_LABELS[t],
                     rec=rec, backbone=backbone, ablation=abl, src=src, dst=dst)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 4: Backbone comparison (MSP -> Seattle)
# ---------------------------------------------------------------------------

def table_backbone(records: list[dict], prov: Provenance) -> str:
    src, dst = "msp", "seattle"
    rows_by_backbone: dict[str, dict | None] = {}
    ablation_used: dict[str, str] = {}
    for bb in BACKBONE_ORDER:
        bb_recs = _records_for_backbone(records, bb, src=src, dst=dst)
        rec = None
        for abl in ("full", "vision_spatial"):  # CLIP grid lacks 'full'; fall back
            cand = [
                r for r in bb_recs
                if r.get("eval_mode") == "cross_city"
                and r.get("train_city") == src and r.get("test_city") == dst
                and r.get("target_tier") in ("1", "all")
                and r.get("ablation") == abl
            ]
            rec = latest_by_ts(cand)
            if rec is not None:
                ablation_used[bb] = abl
                break
        rows_by_backbone[bb] = rec
        if bb not in ablation_used:
            ablation_used[bb] = "n/a"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Backbone comparison (MSP$\\to$Seattle, Tier~1, \\texttt{full} ablation where available). "
        "\\textit{In-corpus} = LoRA fine-tuned on union of all 3 cities; \\textit{held-out city} = LoRA "
        "fine-tuned with Seattle excluded (LOCO).}",
        "\\label{tab:backbone}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(TIER1_BINARY) + "}",
        "\\toprule",
        "Backbone & " + " & ".join(TARGET_LABELS[t] + " AUROC" for t in TIER1_BINARY) + " \\\\",
        "\\midrule",
    ]
    for bb in BACKBONE_ORDER:
        rec = rows_by_backbone[bb]
        label = BACKBONE_LABELS[bb]
        suffix = "" if ablation_used.get(bb) in (None, "full") else f" [{ablation_used[bb]}]"
        if rec is None:
            lines.append(f"{label}{suffix} & " + " & ".join(["---"] * len(TIER1_BINARY)) + " \\\\")
        else:
            vals = [fmt(get_metric(rec, t, "auroc")) for t in TIER1_BINARY]
            lines.append(f"{label}{suffix} & " + " & ".join(vals) + " \\\\")
        for t in TIER1_BINARY:
            prov.add(table="backbone_comparison",
                     row=bb, col=TARGET_LABELS[t],
                     rec=rec, backbone=bb, ablation=ablation_used.get(bb, "n/a"),
                     src=src, dst=dst)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 5: Spatial CV
# ---------------------------------------------------------------------------

def table_spatial_cv(cv_records: list[dict], prov: Provenance) -> str:
    if not cv_records:
        return "% Table 5: No spatial CV results found.\n"

    by_city: dict[str, list[dict]] = {}
    for r in cv_records:
        city = r.get("city", "?")
        by_city.setdefault(city, []).append(r)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{5-fold spatial cross-validation (KMeans blocks). "
        "AUROC mean $\\pm$ std for \\texttt{overture\\_sidewalk\\_present}, Tier~1.}",
        "\\label{tab:spatial_cv}",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "City & Random split AUROC & Spatially blocked AUROC \\\\",
        "\\midrule",
    ]
    for city in CITIES:
        if city not in by_city:
            lines.append(f"{CITY_LABELS[city]} & --- & --- \\\\")
            prov.add(table="spatial_cv", row=city, col="auroc", rec=None,
                     backbone="n/a", ablation="full", src=city, dst=city)
            continue
        rec = latest_by_ts(by_city[city])
        fold_aurocs = []
        for fold in rec.get("per_fold_results", []):
            v = fold.get("metrics", {}).get("overture_sidewalk_present", {}).get("auroc")
            if v is not None and is_finite(v):
                fold_aurocs.append(float(v))
        if fold_aurocs:
            cv_mean = sum(fold_aurocs) / len(fold_aurocs)
            cv_std = statistics.pstdev(fold_aurocs) if len(fold_aurocs) > 1 else 0.0
            cv_str = f"{cv_mean:.3f} $\\pm$ {cv_std:.3f}"
        else:
            cv_str = "---"
        rand_auroc = rec.get("random_split_auroc", float("nan"))
        rand_str = fmt(rand_auroc) if is_finite(rand_auroc) else "---"
        lines.append(f"{CITY_LABELS[city]} & {rand_str} & {cv_str} \\\\")
        prov.add(table="spatial_cv", row=city, col="auroc",
                 rec=rec, backbone="n/a", ablation="full", src=city, dst=city)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# E4: real summary file (per backbone × direction × target)
# ---------------------------------------------------------------------------

def write_summary_json(records: list[dict], path: Path) -> None:
    pairs = [(s, d) for s in CITIES for d in CITIES if s != d]
    summary: dict[str, Any] = {"by_backbone_direction_target": {}}
    for bb in BACKBONE_ORDER:
        bb_summary: dict[str, Any] = {}
        for src, dst in pairs:
            bb_recs = _records_for_backbone(records, bb, src=src, dst=dst)
            cell = [
                r for r in bb_recs
                if r.get("ablation") == "full"
                and r.get("eval_mode") == "cross_city"
                and r.get("train_city") == src and r.get("test_city") == dst
                and r.get("target_tier") in ("1", "all")
            ]
            r = latest_by_ts(cell)
            if r is None:
                continue
            row = {
                "timestamp":  r.get("timestamp"),
                "ablation":   r.get("ablation"),
                "n":          r.get("results_per_target", {}).get("overture_sidewalk_present", {}).get("n"),
            }
            for t in TIER1_BINARY:
                row[f"{t}_auroc"] = get_metric(r, t, "auroc")
                row[f"{t}_ap"]    = get_metric(r, t, "avg_precision")
            for t in TIER1_REGR:
                row[f"{t}_spearman"] = get_metric(r, t, "spearman_rho")
                row[f"{t}_r2_recal"] = get_metric(r, t, "r2_recalibrated")
            bb_summary[f"{src}->{dst}"] = row

        # Cross-direction means
        if bb_summary:
            keys = next(iter(bb_summary.values())).keys()
            mean_row = {}
            for k in keys:
                vals = [v[k] for v in bb_summary.values() if isinstance(v.get(k), (int, float)) and is_finite(v.get(k))]
                if vals:
                    mean_row[k] = float(sum(vals) / len(vals))
            bb_summary["_mean_across_directions"] = mean_row
        summary["by_backbone_direction_target"][bb] = bb_summary

    summary["_meta"] = {
        "n_records_total":   len(records),
        "n_records_kept":    sum(1 for r in records if has_valid_metrics(r, TIER1_ALL)),
        "min_n_threshold":   MIN_N,
        "backbones_present": sorted({r.get("backbone", "siglip") for r in records}),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Table 6: Pittsburgh zero-shot (train=dc+msp+seattle → test=pittsburgh)
# ---------------------------------------------------------------------------

TIER1_WALK = TIER1_ALL + ["walkability_index_v2"]
TARGET_LABELS_WALK = {
    **TARGET_LABELS,
    "walkability_index_v2": "Walk-idx $\\rho$",
}


def table_pittsburgh_zeroshot(records: list[dict], prov: Provenance) -> str:
    """One row per backbone, all 5 targets including walkability_index_v2."""
    pgh_records = [
        r for r in records
        if r.get("test_city") == "pittsburgh"
        and r.get("ablation") == "full"
        and r.get("eval_mode") == "cross_city"
        and r.get("target_tier") in ("1", "all")
    ]
    if not pgh_records:
        return "% Table 6: No Pittsburgh zero-shot runs found in JSONL.\n"

    by_backbone: dict[str, list[dict]] = {}
    for r in pgh_records:
        bb = r.get("backbone", "")
        by_backbone.setdefault(bb, []).append(r)

    targets = TIER1_ALL + ["walkability_index_v2"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pittsburgh zero-shot transfer (train: MSP+Seattle+DC $\\to$ test: Pittsburgh). "
        "AUROC for binary targets; Spearman~$\\rho$ for regression. "
        "Seeds 41/42/43 averaged (latest-by-timestamp shown when multi-seed absent). "
        "No Pittsburgh data seen during training.}",
        "\\label{tab:pittsburgh_zeroshot}",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(targets) + "}",
        "\\toprule",
        "Backbone & " + " & ".join(TARGET_LABELS_WALK[t] for t in targets) + " \\\\",
        "\\midrule",
    ]
    for bb in PITTSBURGH_BACKBONE_ORDER:
        label = PITTSBURGH_BACKBONE_LABELS.get(bb, bb)
        recs = by_backbone.get(bb, [])
        rec = latest_by_ts(recs)
        if rec is None:
            vals = ["---"] * len(targets)
        else:
            vals = []
            for t in TIER1_BINARY:
                vals.append(fmt(get_metric(rec, t, "auroc")))
            for t in TIER1_REGR + ["walkability_index_v2"]:
                vals.append(fmt(get_metric(rec, t, "spearman_rho")))
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
        for t in targets:
            prov.add(table="pittsburgh_zeroshot",
                     row=bb, col=TARGET_LABELS_WALK[t],
                     rec=rec, backbone=bb, ablation="full",
                     src="dc+msp+seattle", dst="pittsburgh")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# E3: --require-backbones gate
# ---------------------------------------------------------------------------

def check_required_backbones(records: list[dict], required: list[str]) -> list[str]:
    """Return a list of human-readable issues, empty if all required backbones have full 6x8 grid."""
    issues = []
    pairs = [(s, d) for s in CITIES for d in CITIES if s != d]
    canonical_ablations = ["full", "vision_only", "street_only", "naip_only",
                           "spatial_only", "tabular_only", "vision_tabular", "vision_spatial"]
    for bb in required:
        if bb not in BACKBONE_LABELS:
            issues.append(f"Unknown required backbone: {bb}")
            continue
        for src, dst in pairs:
            bb_recs = _records_for_backbone(records, bb, src=src, dst=dst)
            for abl in canonical_ablations:
                hits = [r for r in bb_recs
                        if r.get("eval_mode") == "cross_city"
                        and r.get("train_city") == src and r.get("test_city") == dst
                        and r.get("target_tier") in ("1", "all")
                        and r.get("ablation") == abl]
                if not hits:
                    issues.append(f"missing: backbone={bb} {src}->{dst} ablation={abl}")
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate LaTeX result tables for WalkCLIP-3C paper")
    ap.add_argument("--output",     default=str(DEFAULT_OUT))
    ap.add_argument("--provenance", default=str(DEFAULT_PROV))
    ap.add_argument("--summary",    default=str(DEFAULT_SUMMARY))
    ap.add_argument("--ablation-direction", default="msp:seattle",
                    help="src:dst direction for ablation table (default: msp:seattle)")
    ap.add_argument("--require-backbones", default="",
                    help="Comma-separated list of backbones that MUST have a complete 6x8 cross-city "
                         "grid. Script exits with code 2 if any cell missing. "
                         "Example: --require-backbones siglip,siglip_lora,clip,siglip_lora_loco")
    ap.add_argument("--include-src-backbones", action="store_true",
                    help="Extend backbone tables to include SigLIP-2, DINOv2, SigLIP-2+RemoteCLIP "
                         "(src_backbones experiment tag).")
    ap.add_argument("--include-pittsburgh", action="store_true",
                    help="Add Table 6: Pittsburgh zero-shot results (train=dc+msp+seattle → "
                         "test=pittsburgh, all backbones, walkability_index_v2 included).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    raw_mt = load_jsonl(MT_JSONL)
    cv_records = load_jsonl(CV_JSONL)
    mt_records = filter_records(raw_mt)

    print(f"Loaded {len(raw_mt)} multitarget runs ({len(mt_records)} after E5 filter), "
          f"{len(cv_records)} CV runs", flush=True)

    if args.include_src_backbones:
        BACKBONE_LABELS.update(SRC_BACKBONE_LABELS)
        for bb in ["siglip2", "dinov2", "siglip2+remoteclip"]:
            if bb not in BACKBONE_ORDER:
                BACKBONE_ORDER.append(bb)
        print(f"--include-src-backbones: BACKBONE_ORDER={BACKBONE_ORDER}", flush=True)

    if args.require_backbones.strip():
        required = [b.strip() for b in args.require_backbones.split(",") if b.strip()]
        issues = check_required_backbones(mt_records, required)
        if issues:
            print("REQUIRED BACKBONES CHECK FAILED:", file=sys.stderr)
            for it in issues:
                print(f"  - {it}", file=sys.stderr)
            return 2
        print(f"Required backbones {required} all complete.", flush=True)

    src, dst = args.ablation_direction.split(":")
    prov = Provenance()

    # Per-backbone modality ablation tables
    ablation_sections: list[str] = []
    for bb in BACKBONE_ORDER:
        bb_recs = _records_for_backbone(mt_records, bb, src=src, dst=dst)
        if any(r.get("eval_mode") == "cross_city" and r.get("train_city") == src
               and r.get("test_city") == dst for r in bb_recs):
            ablation_sections.append(f"% --- Modality ablation, backbone={bb} ---")
            ablation_sections.append(table_modality_ablation(mt_records, (src, dst), bb, prov))

    sections = [
        "% WalkCLIP-3C auto-generated tables\n% Generated by build_results_tables.py\n"
        "% Provenance: every cell is logged in tables_provenance.json\n",
        "% ===== Table 2: Cross-city Tier 1 (per backbone) =====",
        table_cross_city_tier1(mt_records, prov),
        "% ===== Table 3: Modality ablation (per backbone) =====",
        *ablation_sections,
        "% ===== Table 4: Backbone comparison =====",
        table_backbone(mt_records, prov),
        "% ===== Table 5: Spatial CV =====",
        table_spatial_cv(cv_records, prov),
    ]
    if args.include_pittsburgh:
        sections += [
            "% ===== Table 6: Pittsburgh zero-shot =====",
            table_pittsburgh_zeroshot(mt_records, prov),
        ]
        print("--include-pittsburgh: Table 6 (Pittsburgh zero-shot) added.", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections), encoding="utf-8")
    prov.write(Path(args.provenance))
    write_summary_json(mt_records, Path(args.summary))
    print(f"Saved: {out_path}\nSaved: {args.provenance}\nSaved: {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
