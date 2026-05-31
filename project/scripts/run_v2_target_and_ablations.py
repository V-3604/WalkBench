"""Driver: retrain composite head against the 5-feature walkability_index_v2
target (Option 1C), and run the f_ped ablation (Option 2).

What this does, in order:

1. Composite-head retrain: 5 backbones x 6 LOCO directions x 3 seeds = 90 runs.
   Each run trains the full 6-target Tier-1 head (sidewalk, crosswalk,
   intersection, building-frac, walkability_v1, walkability_v2). The four
   non-composite head targets are unchanged so their numbers will be within
   seed noise of the existing bundles; walkability_index_v2 is the new
   meaningful output. Saves prediction bundles to project/artifacts/predictions
   tagged with --save-predictions.

2. f_ped ablation: SigLIP frozen x 6 LOCO directions x 3 seeds = 18 runs with
   --spatial-features none. All other settings match (1). Quantifies the
   contribution of the pedestrian-graph kNN channel to the cross-city
   walkability rho.

Run:
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_v2_target_and_ablations.py --phase all
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_v2_target_and_ablations.py --phase retrain
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_v2_target_and_ablations.py --phase ablation
    & ".venv\\Scripts\\python.exe" project\\scripts\\run_v2_target_and_ablations.py --dry-run --phase all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "project" / "scripts" / "train_multitarget.py"
SPATIAL_CV_SCRIPT = REPO_ROOT / "project" / "scripts" / "analyze_spatial_blocking.py"
CITIES = ["msp", "seattle", "dc"]
SEEDS = [41, 42, 43]
BACKBONES_RETRAIN = [
    ("siglip",          None),
    ("clip",            None),
    ("siglip_lora",     None),
    ("siglip_lora_loco_<test>",    "loco_v1"),   # placeholder substituted per direction
    ("siglip_lora_loco_v2_<test>", "loco_v2"),
]
# In-city spatial CV: per (city, backbone) on the v2 walkability target.
# LoRA-LOCO is matched to the city under test (the adapter has never seen that
# city's heading labels).
BACKBONES_INCITY = ["siglip", "clip", "siglip_lora", "siglip_lora_loco_v2_<city>"]


def build_pairs() -> list[tuple[str, str]]:
    return [(s, d) for s in CITIES for d in CITIES if s != d]


def cmd(extra: list[str], ablation: str = "full") -> list[str]:
    base = [sys.executable, str(TRAIN_SCRIPT),
            "--target-tier", "1",
            "--ablation", ablation,
            "--pca-dim", "128",
            "--heading-supervision", "on",
            "--save-predictions"]
    return base + extra


def runs_retrain() -> list[list[str]]:
    out: list[list[str]] = []
    for src, dst in build_pairs():
        for seed in SEEDS:
            common = ["--train-city", src, "--test-city", dst,
                      "--spatial-features", "pedgraph",
                      "--seed", str(seed)]
            for backbone_tmpl, _ in BACKBONES_RETRAIN:
                bb = backbone_tmpl.replace("<test>", dst)
                out.append(cmd(common + ["--backbone", bb]))
    return out


def runs_incity() -> list[list[str]]:
    """In-city 5-fold spatially blocked CV against the 5-feature walkability_index_v2.
    Reuses analyze_spatial_blocking.py; one run per (city, backbone)."""
    out: list[list[str]] = []
    for city in CITIES:
        for bb_tmpl in BACKBONES_INCITY:
            bb = bb_tmpl.replace("<city>", city)
            out.append([
                sys.executable, str(SPATIAL_CV_SCRIPT),
                "--city", city,
                "--backbone", bb,
                "--target", "walkability_index_v2",
                "--ablation", "full",
                "--pca-dim", "128",
                "--model-family", "mlp",
                "--block-buffer-m", "500",
                "--n-folds", "5",
            ])
    return out


def runs_ablation() -> list[list[str]]:
    """f_ped ablation: vision_only ablation drops the spatial (pedgraph kNN)
    feature group while keeping street + NAIP. Captions are dropped from all
    ablations by design in walkclip_stage_e_v2.py, so vision_only is the exact
    'no f_ped' configuration relative to the headline 'full' ablation."""
    out: list[list[str]] = []
    for src, dst in build_pairs():
        for seed in SEEDS:
            out.append(cmd([
                "--train-city", src, "--test-city", dst,
                "--spatial-features", "pedgraph",
                "--seed", str(seed),
                "--backbone", "siglip",
            ], ablation="vision_only"))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["retrain", "ablation", "incity", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    args = p.parse_args()

    queue: list[list[str]] = []
    if args.phase in ("retrain", "all"):
        queue += runs_retrain()
    if args.phase in ("ablation", "all"):
        queue += runs_ablation()
    if args.phase in ("incity", "all"):
        queue += runs_incity()

    print(f"[driver] {len(queue)} runs queued (phase={args.phase})", flush=True)
    if args.dry_run:
        for c in queue[:6]:
            print("  " + " ".join(c))
        if len(queue) > 6:
            print(f"  ... ({len(queue) - 6} more)")
        return 0

    failed: list[str] = []
    t0 = time.time()
    for i, c in enumerate(queue, 1):
        label = " ".join(c[3:])
        print(f"\n[driver {i}/{len(queue)}] {label}", flush=True)
        t = time.time()
        rc = subprocess.run(c).returncode
        elapsed = time.time() - t
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"[driver] {status} ({elapsed:.0f}s)", flush=True)
        if rc != 0:
            failed.append(label)
            if args.stop_on_fail:
                break
    total = time.time() - t0
    print(f"\n[driver] Done in {total:.0f}s. Failed: {len(failed)}", flush=True)
    for f in failed:
        print(f"  {f}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
