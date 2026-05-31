# WalkBench — Data Reference

This document covers every data file the pipeline reads or writes: what it is, where it lives, its schema, and its checksum.

**Scope.** Reported results cover five targets: sidewalk presence, near-buffer crosswalk presence, intersection density, building-footprint fraction, and the composite walkability index (`walkability_index_v2`). Training runs pull these with `--target-tier 1`. For an overview of the project, see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

**What is in git vs what you need to place:**

| Source | What it contains |
|---|---|
| `git clone` | All scripts, docs, label CSVs, lock files, point tables, spatial features, caption features, result JSONs/JSONL/LaTeX tables, and prediction bundles |
| Google Drive zip | Embeddings (`project/artifacts/embeddings/`, ~3.3 GB) and LoRA adapters (`project/artifacts/models/`, ~1.3 GB) |
| Raw imagery (`project/data/raw/`) | ~7 GB; only needed to re-run Stage E embedding extraction or regenerate figures |

After cloning, only the Drive zip needs to be extracted — everything else is already present.

**See § "Google Drive zip contents" at the bottom of this file** for the exact embedding and adapter file list.

---

## Benchmark overview

| City | Abbreviation | Locked points | FLA rows | Role |
|---|---|---:|---:|---|
| Minneapolis–St. Paul | `msp` | 4,847 | ~4,824 | 3-city LOCO training/eval |
| Seattle | `seattle` | 4,851 | ~4,824 | 3-city LOCO training/eval |
| Washington DC | `dc` | 4,994 | 4,994 | 3-city LOCO training/eval |
| Pittsburgh | `pittsburgh` | 4,982 | 4,982 | 4th external zero-shot city |
| **Total** | | **19,674** | **19,624** | |

The 50-row delta between lock IDs and FLA rows for the 3-city subset is documented in the discrepancy log at the bottom of this file (some points lost in multimodal join at SAM2/caption stage). Pittsburgh rows have NaN for the SAM2 and caption columns (captions and SAM2 not yet run for Pittsburgh).

---

## Directory layout

```
project/
├── data/
│   ├── raw/                                    ← imagery (place shared imagery here)
│   │   ├── streetview/
│   │   │   ├── msp/                            ← ~19,467 JPGs: {point_id}_{heading}.jpg
│   │   │   ├── seattle/                        ← ~19,461 JPGs
│   │   │   └── dc/                             ← ~19,995 JPGs
│   │   ├── aerial_naip/
│   │   │   ├── MSP/                            ← ~4,874 JPGs: {point_id}.jpg
│   │   │   ├── SEA/                            ← ~4,866 JPGs
│   │   │   └── DC/                             ← ~5,001 JPGs
│   │   └── labels/
│   │       └── gtfs/                           ← GTFS source tables
│   └── processed/
│       ├── locks/                              ← frozen point-ID sets (from bundle)
│       ├── points/                             ← per-city lat/lon metadata (from bundle)
│       ├── labels/                             ← master CSV + Overture/walkability tables (from bundle)
│       ├── spatial_context/                    ← pedgraph features (from bundle)
│       └── text/                               ← caption features CSV (from bundle)
└── artifacts/
    ├── embeddings/                             ← .npy arrays (from bundle)
    ├── models/                                 ← LoRA adapters + MLP checkpoints (from bundle)
    ├── predictions/                            ← .npz prediction bundles (from bundle)
    └── reports/                                ← result JSONL / JSON / LaTeX (from bundle)
```

All directories under `project/data/` and `project/artifacts/` are in `.gitignore`. They will be empty after `git clone` and need to be populated by extracting the shared bundle at the repo root.

---

## Placing shared data files

### Imagery

Street-view images follow the naming convention `{point_id}_{heading}.jpg` where `heading ∈ {0, 90, 180, 270}`. NAIP aerials are `{point_id}.jpg`. City folder names use the capitalized abbreviation for NAIP (`MSP`, `SEA`, `DC`) and lowercase for street-view (`msp`, `seattle`, `dc`).

```
project/data/raw/streetview/msp/0_0.jpg          ← point_id=0, heading=0°, MSP
project/data/raw/streetview/seattle/42_90.jpg
project/data/raw/aerial_naip/MSP/0.jpg
project/data/raw/aerial_naip/DC/99.jpg
```

### Embeddings

Place `.npy` arrays into `project/artifacts/embeddings/`. Each backbone produces a pair of files per city and modality. The `_ids.npy` sidecar maps row indices to `point_id` integers.

```
project/artifacts/embeddings/
├── siglip_street_msp.npy              (shape: N × 1152, float32)
├── siglip_street_msp_ids.npy          (shape: N, int64)
├── siglip_naip_msp.npy                (shape: N × 1152, float32)
├── siglip_naip_msp_ids.npy
│
├── clip_street_msp.npy                (shape: N × 768, float32)
├── clip_street_msp_ids.npy
├── clip_naip_msp.npy
├── clip_naip_msp_ids.npy
│
├── siglip_lora_street_msp.npy         ← LoRA-adapted SigLIP (in-corpus adapter)
├── siglip_lora_naip_msp.npy
│
└── siglip_lora_loco_{holdout}_{src}_{city}.npy   ← LOCO adapters (see §Backbones)
```

Repeat the `siglip_*`, `clip_*`, `siglip_lora_*` pattern for all three cities (`msp`, `seattle`, `dc`).

### LoRA model adapters

```
project/artifacts/models/
├── siglip_lora_v1/
│   ├── adapter/                       ← PEFT adapter weights
│   ├── classification_heads.pt
│   ├── train_log.jsonl
│   └── val_metrics.json
├── siglip_lora_loco_msp/
│   └── adapter/                       ← adapter trained on SEA + DC only
├── siglip_lora_loco_seattle/
│   └── adapter/                       ← adapter trained on MSP + DC only
├── siglip_lora_loco_dc/
│   └── adapter/                       ← adapter trained on MSP + SEA only
└── siglip_lora_loco_{msp,seattle,dc}_v2/
    └── adapter/                       ← multi-task v2 adapters (r=32, 4 heads)
```

---

## Label files

These CSVs live under `project/data/processed/` once the bundle is extracted. They are the authoritative source of truth for all training runs; do not regenerate them.

### `project/data/processed/labels/features_labels_agreement.csv`

**The master training table.** 19,624 rows × 46+ columns. Every training script reads this file.

SHA-256 (first 12 hex): `f34e55a74a8b`

| Column group | Columns | Source | Reported? |
|---|---|---|---|
| Identity | `point_id`, `city` | join key | yes (join key) |
| Coordinates | `lat`, `lon` | Mapillary centroid | yes (spatial CV) |
| SAM2 fractions | `sidewalk_frac_mean`, `crosswalk_frac_mean`, `vegetation_frac_mean`, `sky_frac_mean`, `building_frac_mean`, ... | `compute_audit_agreement.py` | no (calibration channel only; not in any reported run) |
| NAIP SAM2 | `naip_vegetation_frac`, `naip_building_frac`, `naip_road_frac`, ... | `build_sam2_features.py` | no |
| Channel agreement | `agreement_score`, `high_confidence` | `compute_audit_agreement.py` | no (diagnostic only) |
| Visually grounded targets | `overture_sidewalk_present`, `overture_crosswalk_present` (near-buffer), `overture_intersection_count_200m`, `overture_building_footprint_frac_100m` | `derive_overture_targets.py` | **yes (4 of the 5 targets)** |
| GTFS transit | `stops_400m`, `trips_per_hr_400m` | `stage_c_build_labels.py` | `stops_400m` only (5th composite component) |

**Do not edit or re-derive this file.** Run `sha256sum project/data/processed/labels/features_labels_agreement.csv | cut -c1-12` to verify the checksum.

### `project/data/processed/labels/overture_targets.csv`

19,674 rows. Overture Maps target labels. `crosswalk_present` has been replaced by the near-buffer version (15 m / ±30° wedge from `heading_overture_labels_v2.csv`); the original column is preserved as `crosswalk_present_legacy`.

SHA-256 (first 12): patched 2026-05-07 (checksum in `bootstrap_summary.json` header)

### `project/data/processed/labels/heading_overture_labels_v2.csv`

**The heading-level label table.** 78,696 rows = 19,674 points × 4 headings. Used by the LoRA fine-tune and heading-supervision path in the downstream trainer.

| Column | Description |
|---|---|
| `point_id` | Per-city integer |
| `city` | `msp` / `seattle` / `dc` |
| `heading` | 0 / 90 / 180 / 270 |
| `sidewalk_present` | Binary (Overture footway within 15 m, ±30° wedge) |
| `crosswalk_present` | Binary (Overture crosswalk within 15 m, ±30° wedge) |

Prevalences: MSP sidewalk 0.51 / crosswalk 0.266 · Seattle sidewalk 0.47 / crosswalk 0.148 · DC sidewalk 0.54 / crosswalk 0.213.

**v1 vs v2**: `heading_overture_labels.csv` is the old 100 m / ±45° version kept for rollback only. All headline runs use `_v2`.

### `project/data/processed/labels/walkability_index.csv`

19,624 rows. The composite **walkability index** (reported without the `_v2` suffix) is built as PC1 of five visually grounded components. This is the primary regression target for all headline runs.

**Reported name ↔ column name:** the "walkability index" I report is the `walkability_index_v2` column. The `_v2` suffix stays in code and artifacts (renaming would break frozen result files); only the reported name drops the version.

| Column | Description |
|---|---|
| `point_id` | Per-city integer |
| `city` | City abbreviation |
| `walkability_index_v2` | PC1 score (continuous, standardized) — the reported "walkability index" |
| `walkability_index_v1` | An earlier version of the index. Not used in any reported result. Leave it in place — a few scripts still read it. |

PCA details: `project/artifacts/reports/walkability_index_build.json` — PC1 explains 0.365 of variance in the five components; all loadings are positive. The five components are: `overture_sidewalk_present`, `crosswalk_present_near` (near-buffer), `log1p(intersection_count_200m)`, `building_footprint_frac_100m`, `log1p(stops_400m)`.

### `project/data/processed/labels/labels_joined_v2.csv`

19,674 rows. A legacy intermediate join table, superseded by `features_labels_agreement.csv` (the source of truth); not needed for any reported result. Uses `MSP_0`-style compound IDs (`city_prefix` + `_` + `int`); when merging, split on `_` and map `MSP`→`msp`, `SEA`→`seattle`, `DC`→`dc`. SHA-256 (first 12): `00a7a559f36f`.

### `project/data/processed/spatial_context/pedgraph_features.csv`

19,624 rows. OSM pedestrian-graph kNN features (k=8, radius ≤ 400 m). These are the `f_ped` channel in the full ablation.

SHA-256 (first 12): `c0fdc95cfafc`

| Column | Description |
|---|---|
| `point_id`, `city` | Join key |
| `knn_sidewalk_frac_mean` | Mean sidewalk fraction across k=8 neighbors |
| `knn_crosswalk_frac_mean` | Mean crosswalk fraction across k=8 neighbors |
| `knn_intersection_density` | Intersection density in neighborhood |
| `osmid_coverage` | Fraction of graph edges with matched OSM ID |
| ... | Additional pedestrian-graph statistics |

### `project/data/processed/locks/`

```
msp_v2_final_lock_ids.txt          (4,847 IDs)   SHA-256 first 12: cb9b4e4e4f64
seattle_v2_final_lock_ids.txt      (4,851 IDs)   SHA-256 first 12: 02f43617345d
dc_v2_final_lock_ids.txt           (4,994 IDs)   SHA-256 first 12: 1fe2b38f93ec
pittsburgh_v2_final_lock_ids.txt   (4,982 IDs)   SHA-256 first 12: 53ff0539c1d6
```

These are the canonical point-ID sets. All training uses only rows whose `point_id` appears in the corresponding lock file.

### `project/data/processed/points/{city}_points_v2_final_lock.csv`

Per-city lat/lon table with tract-level metadata. Four files: `msp_points_v2_final_lock.csv`, `seattle_points_v2_final_lock.csv`, `dc_points_v2_final_lock.csv`, `pittsburgh_points_v2_final_lock.csv`.

| Column | Description |
|---|---|
| `point_id` | Per-city integer (0-based) |
| `lat`, `lon` | WGS84 coordinates |
| `tract_geoid` | Census tract GEOID |
| `mapillary_coverage` | Whether point passed coverage gate |

### `project/data/processed/text/`

GPT-4o-mini structured captions (OpenAI Batch API, locked 2026-03-xx).

```
streetview_openai_msp_final_lock.jsonl       ← MSP captions
streetview_openai_seattle_final_lock.jsonl   ← Seattle captions
streetview_openai_dc_final_lock.jsonl        ← DC captions
caption_features_final_lock.csv              ← wide-form caption features (from build_caption_features.py)
```

Captions are not used as training features in the headline `full` ablation (they are the `caption_only` and `caption_spatial` ablation channels). See [`PIPELINE.md §Stage D1`](PIPELINE.md).

---

## Result files

These are written by the pipeline scripts and hold every reported number. They are included in the shared bundle under `project/artifacts/reports/`.

| File | Written by | Contents |
|---|---|---|
| `project/artifacts/reports/multitarget_results.jsonl` | `train_multitarget.py` | 656 runs, one JSON line each; append-only |
| `project/artifacts/reports/spatial_cv_results.jsonl` | `analyze_spatial_blocking.py` | 90 records; in-city 5-fold spatial CV |
| `project/artifacts/reports/bootstrap_summary.json` | `bootstrap_cis.py` | Seed-aggregated mean ± std, all 5 backbones |
| `project/artifacts/reports/bootstrap_cis.jsonl` | `bootstrap_cis.py` | Per-(run_id, target) bootstrap percentile CIs |
| `project/artifacts/reports/paired_tests.jsonl` | `bootstrap_cis.py` | Paired DeLong + bootstrap-Δ tests |
| `project/artifacts/reports/ensemble_results.json` | `ensemble_logits.py` | 3-backbone logit ensemble, 90 rows |
| `project/artifacts/reports/model_vs_ps.json` | `eval_model_vs_ps.py` | Model AUROC vs Project Sidewalk (Seattle) |
| `project/artifacts/reports/walkability_index_build.json` | `build_walkability_index.py` | PCA loadings, explained variance |
| `project/artifacts/reports/tables_draft.tex` | `build_results_tables.py` | LaTeX tables (auto-generated) |
| `project/artifacts/reports/tables_provenance.json` | `build_results_tables.py` | Per-cell source run audit trail |

See [`RESULTS.md`](RESULTS.md) for a full explanation of what every number means.

---

## Discrepancy log

| ID | Issue | Status |
|---|---|---|
| D1 | Lock IDs sum to 14,692 but training CSV has 14,642 rows. 50-row delta from multimodal join drops (points missing NAIP or no SAM2 masks). Not fixable without re-running the join. | Open — documented |
| D2 | `multitarget_summary.json` showed only the last run. | Resolved: `build_results_tables.py` writes the proper multi-run summary. Use `bootstrap_summary.json`. |
| D3 | SAM2 features were used as training features in the `full` ablation. | Resolved 2026-04-26: SAM2 demoted to calibration-channel only; not in `full`. |
| D4 | Captions included in the `full` ablation. | Resolved 2026-04-26: captions dropped from `full`. |
| D5 | Spatial CV used raw-degree KMeans without a buffer. | Resolved 2026-04-26: projected KMeans + 500 m buffer. |
| D6 | Crosswalk labels used a loose 100 m / ±45° wedge. | Resolved 2026-05-07: near-buffer (15 m / ±30°, Overture `subclass` column). |
| D7 | `calibration_audit_results.json` had empty PS/APE arms. | Resolved 2026-05-07: replaced by `model_vs_ps.json`. |

---

## Google Drive zip contents

The zip contains **only** the files that are gitignored: embeddings (~3.3 GB) and LoRA adapters (~1.3 GB). Everything else — label CSVs, lock files, point tables, spatial features, caption features, result JSONs, and prediction bundles — is in the git repo and present after `git clone`.

Extract the zip at the repo root. It should populate exactly these two directories:

### Embeddings (132 files → `project/artifacts/embeddings/`)

**Frozen baselines** — SigLIP SO400m (1152-d) and CLIP ViT-L/14 (768-d):
```
siglip_street_{msp,seattle,dc}.npy     + _ids.npy        (6 files)
siglip_naip_{msp,seattle,dc}.npy       + _ids.npy        (6 files)
siglip_street_{msp,seattle,dc}_per_heading.npy           (3 files)
clip_street_{msp,seattle,dc}.npy       + _ids.npy        (6 files)
clip_naip_{msp,seattle,dc}.npy         + _ids.npy        (6 files)
```

**In-corpus LoRA** (fine-tuned on all 3 cities; in-corpus, not a cross-city number):
```
siglip_lora_street_{msp,seattle,dc}.npy      + _ids.npy  (6 files)
siglip_lora_naip_{msp,seattle,dc}.npy        + _ids.npy  (6 files)
siglip_lora_street_{msp,seattle,dc}_per_heading.npy      (3 files)
```

**LoRA-LOCO v1** (HOLDOUT ∈ {msp, seattle, dc}; cross-city, test city unseen):
```
siglip_lora_loco_{HOLDOUT}_street_{msp,seattle,dc}.npy + _ids.npy   (18 files)
siglip_lora_loco_{HOLDOUT}_naip_{msp,seattle,dc}.npy   + _ids.npy   (18 files)
siglip_lora_loco_{HOLDOUT}_street_{msp,seattle,dc}_per_heading.npy   (9 files)
```

**LoRA-LOCO v2** (multi-task, r=32, attn+MLP; same structure as v1 with `_v2_` prefix):
```
siglip_lora_loco_v2_{HOLDOUT}_street_{msp,seattle,dc}.npy + _ids.npy  (18 files)
siglip_lora_loco_v2_{HOLDOUT}_naip_{msp,seattle,dc}.npy   + _ids.npy  (18 files)
siglip_lora_loco_v2_{HOLDOUT}_street_{msp,seattle,dc}_per_heading.npy  (9 files)
```

Each `_ids.npy` sidecar maps row index → `point_id` integer for that city. The `_per_heading.npy` arrays are shape `(N×4, dim)` and needed when `--heading-supervision on` is passed to the trainer.

Do **not** include `X_sat_augmented.npy` and `X_str_augmented.npy` — those are legacy files from a discarded augmentation experiment.

Sanity check after extraction (should print 132):
```bash
ls project/artifacts/embeddings/ | grep -v "augmented" | wc -l
```

### LoRA adapters (7 directories → `project/artifacts/models/`)

Only these adapter directories are needed. The `downstream/`, `downstream_v2/`, and `encoders/` model subdirectories are **not** needed (those are MLP downstream checkpoints; skip them).

```
project/artifacts/models/
├── siglip_lora_v1/              in-corpus adapter, r=16
├── siglip_lora_loco_msp/        LoRA-LOCO v1, holdout=MSP   (trained on Seattle + DC)
├── siglip_lora_loco_seattle/    LoRA-LOCO v1, holdout=Seattle
├── siglip_lora_loco_dc/         LoRA-LOCO v1, holdout=DC
├── siglip_lora_loco_msp_v2/     LoRA-LOCO v2, holdout=MSP
├── siglip_lora_loco_seattle_v2/ LoRA-LOCO v2, holdout=Seattle
└── siglip_lora_loco_dc_v2/      LoRA-LOCO v2, holdout=DC
```

Each directory contains `adapter/` (PEFT weights), `classification_heads.pt`, `config.json`, `train_log.jsonl`, and `val_metrics.json`.

Adapters are only needed to re-extract embeddings from scratch. If the embeddings are already in place, you can skip the adapters and go straight to `train_multitarget.py`.

### Building the zip (from the source machine)

```bash
# From the repo root — produces a ~4.5 GB compressed archive
tar --use-compress-program='zstd -19 -T0' -cf walkbench_embeddings.tar.zst \
    project/artifacts/embeddings \
    project/artifacts/models/siglip_lora_v1 \
    project/artifacts/models/siglip_lora_loco_msp \
    project/artifacts/models/siglip_lora_loco_seattle \
    project/artifacts/models/siglip_lora_loco_dc \
    project/artifacts/models/siglip_lora_loco_msp_v2 \
    project/artifacts/models/siglip_lora_loco_seattle_v2 \
    project/artifacts/models/siglip_lora_loco_dc_v2
```

If you also want to share raw imagery for Stage E re-extraction, add the three streetview and three NAIP directories (adds ~7 GB compressed to ~6–7 GB total).

### After extracting on the teammate's machine

```bash
cd WalkBench   # cloned repo root
tar --use-compress-program='zstd -d' -xf walkbench_embeddings.tar.zst

# Verify lock integrity
.venv/Scripts/python.exe project/scripts/audit_v2_lock.py
# Verify file presence
.venv/Scripts/python.exe project/scripts/verify_repo.py
```

After extraction at the teammate's repo root, run:

```bash
& $PY project\scripts\audit_v2_lock.py
& $PY project\scripts\verify_repo.py
```

These two scripts verify lock-id integrity and expected file presence respectively. A clean bundle should pass both with zero warnings.
