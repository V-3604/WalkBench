"""
Run the full cross-city training matrix for WalkCLIP v2.

Executes train_multitarget.py for all N×(N-1) directed city pairs,
optionally across multiple tiers. Logs each run's return code.

Run:
    # Full 6-direction Tier 1 sweep (per runbook W3 D18-19)
    & ".venv\Scripts\python.exe" project/scripts/run_full_matrix.py ^
        --tiers 1 --pca-dim 128 --heading-supervision on --spatial-features pedgraph

    # All-tier sweep (W3 D20)
    & ".venv\Scripts\python.exe" project/scripts/run_full_matrix.py ^
        --tiers 1 2 3 --pca-dim 128 --heading-supervision on

    # Dry run (print commands without executing)
    & ".venv\Scripts\python.exe" project/scripts/run_full_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "project/scripts/train_multitarget.py"
CITIES = ["msp", "seattle", "dc"]


def build_command(
    train_city: str,
    test_city: str,
    tier: int,
    ablation: str,
    pca_dim: int,
    heading_supervision: str,
    spatial_features: str,
    features_csv: str,
    loss_targets: str,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-city", train_city,
        "--test-city", test_city,
        "--target-tier", str(tier),
        "--ablation", ablation,
        "--pca-dim", str(pca_dim),
    ]
    if heading_supervision:
        cmd += ["--heading-supervision", heading_supervision]
    if spatial_features:
        cmd += ["--spatial-features", spatial_features]
    if features_csv:
        cmd += ["--features-csv", features_csv]
    if loss_targets and loss_targets != "all":
        cmd += ["--loss-targets", loss_targets]
    cmd += extra_args
    return cmd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run WalkCLIP cross-city training matrix")
    ap.add_argument("--cities", nargs="+", default=CITIES, choices=CITIES,
                    help="Cities to include (default: msp seattle dc)")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1],
                    choices=[1, 2, 3], help="Target tiers to run (default: 1)")
    ap.add_argument("--ablation", default="full",
                    help="Ablation mode passed to trainer (default: full)")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--heading-supervision", default="on", choices=["on", "off"])
    ap.add_argument("--spatial-features", default="pedgraph",
                    choices=["euclidean", "pedgraph", "none"])
    ap.add_argument("--features-csv", default="",
                    help="Override features CSV path (e.g. features_labels_agreement_v2_clean.csv)")
    ap.add_argument("--loss-targets", default="all", choices=["all", "tier1", "tier2"],
                    help="Loss targets forwarded to train_multitarget.py --loss-targets")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="Extra args forwarded verbatim to train_multitarget.py")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"Trainer not found: {TRAIN_SCRIPT}")

    pairs = [
        (src, dst)
        for src in args.cities
        for dst in args.cities
        if src != dst
    ]

    runs = [(src, dst, tier) for src, dst in pairs for tier in args.tiers]
    print(f"[matrix] {len(runs)} runs × {len(args.tiers)} tier(s) planned", flush=True)
    if args.dry_run:
        print("[matrix] DRY RUN — commands only:\n", flush=True)

    failed: list[str] = []
    t0_total = time.time()

    for src, dst, tier in runs:
        cmd = build_command(
            train_city=src, test_city=dst, tier=tier,
            ablation=args.ablation, pca_dim=args.pca_dim,
            heading_supervision=args.heading_supervision,
            spatial_features=args.spatial_features,
            features_csv=args.features_csv,
            loss_targets=args.loss_targets,
            extra_args=args.extra or [],
        )
        label = f"{src}→{dst} tier={tier} ablation={args.ablation}"
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        print(f"\n[matrix] START {label}", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
        print(f"[matrix] {status} {label}  ({elapsed:.0f}s)", flush=True)
        if result.returncode != 0:
            failed.append(label)

    if args.dry_run:
        return 0

    total_elapsed = time.time() - t0_total
    print(f"\n[matrix] Done. {len(runs)} runs in {total_elapsed:.0f}s", flush=True)
    if failed:
        print(f"[matrix] FAILED ({len(failed)}):", flush=True)
        for f in failed:
            print(f"  {f}", flush=True)
        return 1
    print("[matrix] All runs succeeded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
