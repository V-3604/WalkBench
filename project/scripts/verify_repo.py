"""
Verify that all critical v2 WalkCLIP artifacts are present.

Usage:
    & ".venv\Scripts\python.exe" project\scripts\verify_repo.py

Exits 0 if everything is present, 2 if any artifact is missing.
This script checks the v2 3-city benchmark layout, not the v1 layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

CITIES = ("msp", "seattle", "dc")


def repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "project").exists():
            return p
    raise RuntimeError("Could not locate repo root (expected a 'project/' folder).")


def check(path: Path, kind: str, problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"Missing {kind}: {path}")


def main() -> int:
    root = repo_root()
    processed = root / "project" / "data" / "processed"
    artifacts = root / "project" / "artifacts"
    problems: list[str] = []

    # --- lock files ---
    locks = processed / "locks"
    for city in CITIES:
        check(locks / f"{city}_v2_final_lock_ids.txt", "lock", problems)

    # --- labels ---
    labels = processed / "labels"
    check(labels / "features_labels_agreement.csv", "master-labels", problems)
    check(labels / "overture_targets.csv", "labels", problems)
    check(labels / "heading_overture_labels_v2.csv", "labels", problems)
    check(labels / "labels_joined_v2.csv", "labels", problems)

    # --- spatial context ---
    spatial = processed / "spatial_context"
    check(spatial / "pedgraph_features.csv", "spatial", problems)

    # --- per-city point tables ---
    points = processed / "points"
    for city in CITIES:
        check(points / f"{city}_points_v2_final_lock.csv", "points", problems)

    # --- caption features ---
    text = processed / "text"
    check(text / "caption_features_final_lock.csv", "captions", problems)

    # --- frozen backbone embeddings (siglip + clip, street + naip, 3 cities) ---
    emb = artifacts / "embeddings"
    for backbone in ("siglip", "clip"):
        for source in ("street", "naip"):
            for city in CITIES:
                stem = f"{backbone}_{source}_{city}"
                check(emb / f"{stem}.npy", "embedding", problems)
                check(emb / f"{stem}_ids.npy", "embedding", problems)

    # --- LoRA-LOCO v2 embeddings (siglip, street + naip, held-out city × inference city) ---
    for held_out in CITIES:
        for source in ("street", "naip"):
            for city in CITIES:
                stem = f"siglip_lora_loco_v2_{held_out}_{source}_{city}"
                check(emb / f"{stem}.npy", "embedding", problems)
                check(emb / f"{stem}_ids.npy", "embedding", problems)

    # --- LoRA-LOCO v2 adapter checkpoints ---
    models = artifacts / "models"
    for city in CITIES:
        adapter_dir = models / f"siglip_lora_loco_{city}_v2" / "adapter"
        check(adapter_dir / "adapter_config.json", "lora-adapter", problems)
        check(adapter_dir / "adapter_model.safetensors", "lora-adapter", problems)

    # --- key result files ---
    reports = artifacts / "reports"
    for fname in (
        "multitarget_results.jsonl",
        "bootstrap_summary.json",
        "ensemble_results.json",
        "spatial_cv_results.jsonl",
        "label_noise_audit.json",
    ):
        check(reports / fname, "report", problems)

    print("WalkCLIP v2 repo verification")
    print(f"  Repo root : {root}")
    print(f"  Cities    : {', '.join(CITIES)}")

    if problems:
        print(f"\n{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        return 2

    print("\nAll required v2 artifacts present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
