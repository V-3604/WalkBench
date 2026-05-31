# Repo map — where to find what

Where each file lives, and where every reported number comes from. For the project
itself, see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

## What's in the repo vs. what you download

In git (present right after you clone):

- All the code under `project/scripts/` (pipeline + training).
- The docs in `docs/`, plus `README.md`, `requirements.txt`, `.gitignore`.
- Label CSVs, lock files, point tables, and spatial features under
  `project/data/processed/`.
- Result JSON / JSONL / LaTeX tables under `project/artifacts/reports/`.
- Prediction bundles (`.npz`) under `project/artifacts/predictions/`.

Not in git — grab these from the Google Drive zip (see
[`DATA.md`](DATA.md), "Google Drive zip contents"):

- `project/artifacts/embeddings/` — the `.npy` embedding arrays (~3.3 GB).
- `project/artifacts/models/` — the LoRA adapter directories (~1.3 GB).
- `project/data/raw/` — raw imagery (~7 GB; only needed to re-extract embeddings).

## Docs (this folder)

- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) — the whole project, end to end.
- [`DATA.md`](DATA.md) — every data file: layout, schema, checksums, and what to download.
- [`PIPELINE.md`](PIPELINE.md) — the stages from raw geography to results, with the scripts for each.
- [`RESULTS.md`](RESULTS.md) — every result file explained, plus the headline numbers.
- [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) — the papers this builds on.

## Pipeline scripts (`project/scripts/`)

Sampling frame:
- `select_msp_points.py`, `select_seattle_points.py`, `select_dc_points.py`,
  `select_pittsburgh_points.py` — stratified sampling, writes per-city locks
- `evaluate_coverage.py` — Mapillary density probe per bounding box
- `verify_cities.py`, `audit_v2_lock.py`, `verify_repo.py` — integrity checks

Imagery:
- `download_mapillary_images.py` — 4-heading street imagery per point
- `fetch_naip_tiles.py` — NAIP aerial tiles

Labels:
- `derive_overture_targets.py`, `derive_heading_overture_labels.py` — Overture targets
- `stage_c_build_labels.py` → `features_labels_agreement.csv`
- `build_pedgraph_spatial_features.py` — pedestrian-graph context features

Embeddings + LoRA:
- `extract_siglip_embeddings.py`, `extract_clip_embeddings.py`,
  `extract_siglip2_embeddings.py`, `extract_dinov2_embeddings.py`,
  `extract_remoteclip_embeddings.py`
- `finetune_siglip_lora.py` → `extract_siglip_lora_embeddings.py`

Composite index:
- `build_walkability_index.py`, `build_walkability_index_src.py`

Training + analysis:
- `walkclip_stage_e_v2.py` — shared data loaders, scaler, ablation registry
- `train_multitarget.py` — the main trainer
- `run_full_matrix.py`, `run_v2_target_and_ablations.py` — sweep wrappers
- `analyze_spatial_blocking.py` — in-city spatial cross-validation
- `ensemble_logits.py` — logit ensemble
- `bootstrap_cis.py` — bootstrap CIs + paired tests
- `audit_label_noise.py`, `label_ceiling_analysis.py` — Project Sidewalk label audit
- `correlate_walkscore.py` — Walk Score construct-validity check
- `build_results_tables.py` → LaTeX tables

## Where every reported number comes from

| Result | File (under `project/artifacts/reports/`) | Built by |
|---|---|---|
| Cross-city LOCO headline (composite + component targets) | `bootstrap_summary.json` | `train_multitarget.py` → `bootstrap_cis.py` |
| Per-direction 6-way matrix | `multitarget_results.jsonl` (filter `seed==42`) | `train_multitarget.py` |
| Pittsburgh zero-shot | `bootstrap_summary.json` (`--include-pittsburgh`) | `train_multitarget.py` → `bootstrap_cis.py` |
| Ensemble | `ensemble_results.json` | `ensemble_logits.py` |
| In-city spatial CV | `spatial_cv_results.jsonl` (filter `walkability_index_v2`) | `analyze_spatial_blocking.py` |
| Bootstrap 95% CIs | `bootstrap_cis.jsonl` | `bootstrap_cis.py` |
| Paired tests | `paired_tests.jsonl` | `bootstrap_cis.py` |
| Label-ceiling audit (vs Project Sidewalk) | `label_noise_audit.json`, `label_ceiling_analysis.json` | `audit_label_noise.py`, `label_ceiling_analysis.py` |
| Composite index PCA loadings | `walkability_index_build.json` | `build_walkability_index.py` |
| LaTeX tables | `tables_draft_v3.tex` (+ `tables_provenance.json`) | `build_results_tables.py` |

`bootstrap_summary.json` is the authoritative seed-averaged source. The raw
`multitarget_results.jsonl` is an append-only log spanning many runs and schema
versions — use `bootstrap_summary.json` for anything you report. For what each metric
means, see [`RESULTS.md`](RESULTS.md).

## Environments

- **Windows venv** (`.venv\Scripts\python.exe`) — data pipeline and CSV/label scripts.
  New dependencies go in `requirements.txt`.
- **WSL venv** (`.venv_wsl/bin/python`) — CUDA 12.8 on the RTX 5070 Ti. Needed for the
  embedding extraction and LoRA fine-tuning. Run GPU jobs in native WSL, not over
  `/mnt/c`, to avoid the filesystem throughput hit.
