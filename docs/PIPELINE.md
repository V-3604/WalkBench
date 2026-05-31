# WalkBench — Pipeline Reference

End-to-end pipeline from raw geography to final results. All stages are complete for the current benchmark. This document describes what each stage does, which scripts run it, and what it produces.

**All stages are done.** The output files are already present. Commands are provided for reference and for extending the pipeline to new cities.

**Scope.** Reported results cover five targets (listed under Stage F). Reproduce them with `--target-tier 1`.

---

## Stage overview

| Stage | Name | Status |
|---|---|---|
| A | Sampling frame | Done |
| B1 | Mapillary street imagery | Done |
| B2 | NAIP aerial imagery | Done |
| C | Label join (Overture, GTFS) | Done |
| D1 | GPT-4o-mini structured captions | Done |
| D2 | SAM2 segmentation | Done (calibration channel only) |
| D3 | Channel A/B agreement scoring | Done |
| D4 | Overture targets + heading labels | Done |
| E | Embedding extraction (SigLIP, CLIP, LoRA variants) | Done |
| F | Cross-city 6-way sweep | Done |
| G | Composite walkability index + crosswalk relabeling | Done |
| H | LoRA-LOCO v2 + ensemble + spatial CV | Done |

---

## Stage A — Sampling frame

**Script**: `project/scripts/select_{msp,seattle,dc,pittsburgh}_points.py`  
**Purpose**: Stratified sampling of geo-points within each city bounding box. Points must pass a Mapillary coverage gate (at least one image within a short radius). Produces the locked point-ID sets.

**Outputs**:
- `project/data/processed/points/{city}_points_v2_final_lock.csv`
- `project/data/processed/locks/{msp,seattle,dc}_v2_final_lock_ids.txt`

**Lock sizes**: MSP 4,847 · Seattle 4,851 · DC 4,994 · Pittsburgh 4,982 (total 19,674).

```powershell
# Reference command (already done — don't re-run without a new city)
& $PY project\scripts\select_msp_points.py
& $PY project\scripts\select_seattle_points.py
& $PY project\scripts\select_dc_points.py
& $PY project\scripts\select_pittsburgh_points.py
```

---

## Stage B1 — Mapillary street imagery

**Script**: `project/scripts/download_mapillary_images.py`  
**Purpose**: For each locked point, fetch 4 Mapillary images at headings 0°, 90°, 180°, 270°. Requires `MAPILLARY_ACCESS_TOKEN` in `.env`.

**Output**: `project/data/raw/streetview/{msp,seattle,dc}/{point_id}_{heading}.jpg`  
**Counts**: MSP ~19,467 · Seattle ~19,461 · DC ~19,995 images

```bash
# Reference command (requires Mapillary token; already done)
$PY project/scripts/download_mapillary_images.py --cities msp seattle dc
```

---

## Stage B2 — NAIP aerial imagery

**Script**: `project/scripts/fetch_naip_tiles.py`  
**Purpose**: Fetch NAIP (National Agriculture Imagery Program) 1m aerial tiles centered on each locked point from the USDA NRCS API.

**Output**: `project/data/raw/aerial_naip/{MSP,SEA,DC}/{point_id}.jpg`  
**Counts**: MSP ~4,874 · Seattle ~4,866 · DC ~5,001 images

---

## Stage C — Label join

**Script**: `project/scripts/stage_c_build_labels.py`  
**Purpose**: Joins all label sources into the master training table. Sources:

| Source | What it provides |
|---|---|
| Overture Maps | Sidewalk, crosswalk, building footprint, intersection labels |
| GTFS (transit feeds) | Transit stop counts near each point: `stops_400m` |
| SAM2 outputs | Pixel-fraction columns from segmentation |

**Output**: `project/data/processed/labels/features_labels_agreement.csv` (19,624 rows)

Supporting scripts:
- `project/scripts/derive_overture_targets.py` — builds `overture_targets.csv`
- `project/scripts/compute_audit_agreement.py` — adds channel A/B agreement columns
- `project/scripts/build_pedgraph_spatial_features.py` — builds `pedgraph_features.csv`

---

## Stage D1 — GPT-4o-mini structured captions

**Script**: `project/scripts/generate_openai_structured_captions.py`  
**Purpose**: Calls the OpenAI Batch API (GPT-4o-mini) with a structured prompt on each Mapillary image to produce a JSON description of walkability features visible in the image. Requires `OPENAI_API_KEY`.

**Output**: `project/data/processed/text/streetview_openai_{city}_final_lock.jsonl`  
**Post-processing**: `project/scripts/build_caption_features.py` → `caption_features_final_lock.csv`

**Cost**: ~$2.40 total for all 3 cities (OpenAI Batch API pricing).

**Note**: Caption features are used only in the `caption_only` and `caption_spatial` ablation channels. They are **not** included in the headline `full` ablation.

---

## Stage D2 — SAM2 segmentation

**Script**: `project/scripts/segment_sam2.py`, `project/scripts/build_sam2_features.py`  
**Purpose**: Run SAM2 (Segment Anything Model 2) on street and NAIP images to produce pixel-fraction masks. SAM2 features are demoted to a **calibration channel only** — they are not used as training features in any headline run (resolved discrepancy D3 in [`DATA.md`](DATA.md)).

**Output**: `project/data/audit/` (raw mask stats, gitignored) → columns merged into `features_labels_agreement.csv`

---

## Stage D3 — Channel A/B agreement scoring

**Script**: `project/scripts/compute_audit_agreement.py`  
**Purpose**: Computes the agreement score between caption (channel A) and SAM2 (channel B) predictions for each point. Used as a diagnostic — high-confidence points have agreement ≥ 0.75.

**Output**: `agreement_score` and `high_confidence` columns in `features_labels_agreement.csv`  
**Mean agreement**: 0.774; 46.8% of points flagged as high-confidence.

---

## Stage D4 — Overture heading labels

**Script**: `project/scripts/derive_heading_overture_labels.py`  
**Purpose**: For each `(point_id, heading)` pair, determines whether an Overture Maps feature (sidewalk, crosswalk) falls within a 15 m / ±30° directional wedge. This produces the per-heading binary labels used by the LoRA fine-tune and heading supervision.

**Output**: `project/data/processed/labels/heading_overture_labels_v2.csv` (78,696 rows)

**v1 vs v2**: The original v1 labels used a loose 100 m / ±45° wedge and an incorrect Overture column. v2 uses 15 m / ±30° and the `subclass` column. All headline runs use v2.

```powershell
& $PY project\scripts\derive_heading_overture_labels.py --version v2
```

---

## Stage E — Embedding extraction

This stage extracts visual embeddings from all imagery using four backbone configurations. Each produces `.npy` arrays stored in `project/artifacts/embeddings/`. See [`DATA.md §Embeddings`](DATA.md) for the file naming convention.

### E1 — Frozen SigLIP SO400m

**Script**: `project/scripts/extract_siglip_embeddings.py`  
**Model**: `google/siglip-so400m-patch14-384` (HuggingFace)  
**Output dim**: 1152 per image  
**GPU required**: Yes (WSL `.venv_wsl`)

```bash
# WSL, GPU
python project/scripts/extract_siglip_embeddings.py --source all --city all
# Writes: siglip_{street,naip}_{msp,seattle,dc}.npy + _ids.npy
```

### E2 — Frozen CLIP ViT-L/14

**Script**: `project/scripts/extract_clip_embeddings.py`  
**Model**: `openai/clip-vit-large-patch14` (via open_clip_torch)  
**Output dim**: 768 per image  
**GPU required**: Yes (~2 h per city)

```bash
python project/scripts/extract_clip_embeddings.py --cities msp seattle dc
# Writes: clip_{street,naip}_{msp,seattle,dc}.npy + _ids.npy
```

### E3 — SigLIP + LoRA (in-corpus adapter)

**Scripts**: `project/scripts/cache_pixel_values.py` → `project/scripts/finetune_siglip_lora.py` → `project/scripts/extract_siglip_lora_embeddings.py`

This backbone is fine-tuned on the union of all 3 cities' heading-level Overture labels. Because the test city's labels were seen during backbone training, this is an **in-corpus** result — it sees the test city — and should not be reported as a cross-city transfer number.

**LoRA config**: r=16, α=32, dropout=0.1, target modules = `{q,k,v,out}_proj` of all attention blocks. Two binary classification heads (sidewalk, crosswalk). ~3–5M trainable / ~400M total (≈1%).

**Step 1 — Build pixel cache** (one-time, ~40 min, ~52 GB, writes to `~/walkclip_cache/`):
```bash
# WSL, .venv_wsl (native filesystem — mandatory for GPU utilization)
python project/scripts/cache_pixel_values.py \
    --output-dir ~/walkclip_cache \
    --batch-size 64 --num-workers 8
```

**Step 2 — Fine-tune** (~30–45 min):
```bash
python project/scripts/finetune_siglip_lora.py \
    --cache-dir ~/walkclip_cache \
    --epochs 10 --batch-size 64 \
    --output project/artifacts/models/siglip_lora_v1
```

**Step 3 — Re-extract embeddings** (~15–25 min):
```bash
python project/scripts/extract_siglip_lora_embeddings.py \
    --source all --city all --batch-size 64 \
    --adapter project/artifacts/models/siglip_lora_v1/adapter \
    --cache-dir ~/walkclip_cache
# Writes: siglip_lora_{street,naip}_{msp,seattle,dc}.npy
```

### E4 — SigLIP + LoRA-LOCO (leave-one-city-out adapters)

Three separate adapters, each trained on data from the **other two cities only**. The adapter for a holdout city `{dst}` has never seen any of `{dst}`'s images or labels at fine-tune time. This is the cross-city backbone — the test city is unseen.

- `siglip_lora_loco_msp` — trained on Seattle + DC only
- `siglip_lora_loco_seattle` — trained on MSP + DC only
- `siglip_lora_loco_dc` — trained on MSP + Seattle only

The downstream trainer enforces the LOCO contract: when testing on city `dst`, it selects `--backbone siglip_lora_loco_{dst}`.

**v2 (multi-task)**: The v2 LOCO adapters use r=32, attention+MLP layers, and 4 prediction heads instead of 2. These are in `project/artifacts/models/siglip_lora_loco_{city}_v2/`.

---

## Stage F — Cross-city sweep (Downstream training)

**Script**: `project/scripts/train_multitarget.py`  
**Library**: `project/scripts/walkclip_stage_e_v2.py` (shared constants, data loaders, scaler)

This is the primary training script. It trains a small fusion MLP on top of the pre-extracted embeddings + spatial features and evaluates cross-city generalization (leave-one-city-out protocol).

### Architecture (WalkCLIPMultiTarget)

- Modality-specific linear projectors: `street_proj` and `naip_proj` each map from backbone dim → 128
- PCA dimensionality reduction: PCA-128 applied to each modality separately **fit on the training city only**
- Optional spatial channel: `pedgraph_features.csv` (k=8 OSM neighbors, max 400 m)
- Fusion: concatenate projected modalities + spatial channel → 128-d hidden MLP
- Heads: one `nn.Linear(128, 1)` per target

### Targets (5 visually grounded targets only)

| Target | Type | Metric | Notes |
|---|---|---|---|
| `overture_sidewalk_present` | Binary | AUROC | Primary classification target |
| `crosswalk_present_near` | Binary | AUROC | Near-buffer (15 m / ±30°) Overture label |
| `overture_intersection_count_200m` | Regression | Spearman ρ | `log1p`-transformed during training; `expm1` at eval |
| `overture_building_footprint_frac_100m` | Regression | Spearman ρ | Aerial-derived; most transferable target |
| `walkability_index_v2` | Regression | Spearman ρ | PC1 of the 5 components above |

### Ablation channels (`--ablation`)

| Key | Channels included | Notes |
|---|---|---|
| `full` | SigLIP/CLIP street + NAIP embeddings + pedgraph spatial | **Headline ablation** |
| `vision_only` | Street + NAIP embeddings, no spatial | Used for f_ped ablation |
| `street_only` | Street embeddings only | Modality ablation |
| `naip_only` | NAIP embeddings only | Modality ablation |
| `spatial_only` | Pedgraph features only | Modality ablation |
| `caption_only` | GPT-4o-mini caption features only | Text-only baseline |
| `caption_spatial` | Caption + pedgraph | Text + spatial baseline |

### Training configuration

| Parameter | Value |
|---|---|
| PCA dim | 128 (street + NAIP separately; cumulative variance ~0.93) |
| Hidden dim | 128 |
| Max epochs | 300 with early stopping (patience 40) |
| Optimizer | AdamW with cosine LR schedule |
| Heading supervision | On by default (`heading_overture_labels_v2.csv`) |
| Seeds | 41, 42, 43 for multi-seed runs; 42 for single-seed |
| Calibration | Isotonic per (run, binary-target) on val logits → test logits |
| Device | CUDA (RTX 5070 Ti) or CPU |

### Running a single cross-city run

```bash
# WSL, GPU — train on MSP, test on Seattle, full ablation
PY=".venv/Scripts/python.exe"
$PY project/scripts/train_multitarget.py \
  --train-city msp --test-city seattle \
  --target-tier 1 --ablation full \
  --pca-dim 128 --heading-supervision on \
  --spatial-features pedgraph \
  --backbone siglip --seed 42 --save-predictions
```

Key flags:

| Flag | Description |
|---|---|
| `--train-city` | Source city for training data |
| `--test-city` | Target city (never seen by backbone in LOCO runs) |
| `--backbone` | `siglip`, `clip`, `siglip_lora`, `siglip_lora_loco_{dst}` |
| `--target-tier` | `1` |
| `--ablation` | See table above |
| `--seed` | Integer; pass `--seed 41 --seed 42 --seed 43` in a loop for multi-seed |
| `--save-predictions` | Writes a `.npz` bundle to `project/artifacts/predictions/` |
| `--spatial-features` | `pedgraph` (default for headline) or `none` (vision_only ablation) |

### Running all 6 cross-city directions (multi-seed)

```bash
SEEDS=(41 42 43)
CITIES=(msp seattle dc)
for seed in "${SEEDS[@]}"; do
  for src in "${CITIES[@]}"; do
    for dst in "${CITIES[@]}"; do
      [[ "$src" == "$dst" ]] && continue
      $PY project/scripts/train_multitarget.py \
        --train-city $src --test-city $dst \
        --target-tier 1 --ablation full \
        --pca-dim 128 --heading-supervision on \
        --spatial-features pedgraph \
        --backbone siglip --seed $seed --save-predictions
    done
  done
done
```

Results are appended to `project/artifacts/reports/multitarget_results.jsonl`.

---

## Stage G — Composite walkability index

### G1 — Build walkability index

**Script**: `project/scripts/build_walkability_index.py`  
**Purpose**: Fits PCA on the five visually grounded components and saves PC1 scores as the composite walkability index.

```powershell
& $PY project\scripts\build_walkability_index.py
# Writes: project/data/processed/labels/walkability_index.csv
# Writes: project/artifacts/reports/walkability_index_build.json (PCA loadings)
```

### G2 — Crosswalk near-buffer relabeling

**Script**: `project/scripts/relabel_crosswalk_near_buffer.py`  
**Purpose**: Regenerates heading-level crosswalk labels using the tighter 15 m / ±30° wedge.

```powershell
& $PY project\scripts\relabel_crosswalk_near_buffer.py
# Updates: heading_overture_labels_v2.csv
```

### G3 — Bootstrap CIs and paired tests

**Script**: `project/scripts/bootstrap_cis.py`  
**Purpose**: Given the `.npz` prediction bundles from multi-seed runs, computes 1000-sample bootstrap 95% CIs and paired DeLong tests (binary) / bootstrap-Δ tests (regression).

```powershell
& $PY project\scripts\bootstrap_cis.py --n-boot 1000 `
    --pairs siglip_lora_loco,siglip `
    --pairs siglip_lora_loco,clip `
    --pairs siglip_lora,siglip_lora_loco `
    --pairs siglip,clip
# Writes: bootstrap_cis.jsonl, bootstrap_summary.json, paired_tests.jsonl
```

---

## Stage H — LoRA-LOCO v2 + Ensemble + Spatial CV

### H1 — Multi-task LoRA-LOCO v2

Three independent LOCO adapters, each trained with r=32, 4 prediction heads (sidewalk + crosswalk + intersection + building-frac), and LoRA applied to both attention and MLP layers. Trained via the same `finetune_siglip_lora.py` script with different `--exclude-city` arguments.

```bash
# Train 3 adapters (one per holdout city) — ~45 min each on 5070 Ti
for holdout in msp seattle dc; do
  python project/scripts/finetune_siglip_lora.py \
    --exclude-city $holdout \
    --lora-r 32 --lora-alpha 64 \
    --n-heads 4 \
    --output project/artifacts/models/siglip_lora_loco_${holdout}_v2
done
```

After fine-tuning, re-extract embeddings for each LOCO adapter and run the 6-direction downstream grid with `--backbone siglip_lora_loco_{dst}`.

### H2 — Logit ensemble

**Script**: `project/scripts/ensemble_logits.py`  
**Purpose**: Averages raw logits from three backbones (SigLIP frozen + CLIP frozen + LoRA-LOCO v2) across all 90 prediction bundles.

```powershell
& $PY project\scripts\ensemble_logits.py
# Writes: project/artifacts/reports/ensemble_results.json
```

### H3 — In-city spatial CV

**Script**: `project/scripts/analyze_spatial_blocking.py`  
**Purpose**: 5-fold spatially-blocked cross-validation within each city. Folds are formed by KMeans clustering in projected metres. A 500 m buffer between train and test zones prevents leakage from the pedgraph kNN features (which aggregate data from neighbors within 400 m).

```powershell
foreach ($city in 'msp','seattle','dc') {
  foreach ($bb in 'siglip','clip','siglip_lora',"siglip_lora_loco_$city") {
    & $PY project\scripts\analyze_spatial_blocking.py `
        --city $city --backbone $bb `
        --target walkability_index_v2 `
        --block-buffer-m 500 --n-folds 5
  }
}
# Appends 12 records to: project/artifacts/reports/spatial_cv_results.jsonl
```

---

## Generating result tables

Once all runs are logged, regenerate LaTeX tables:

```powershell
& $PY project\scripts\build_results_tables.py `
    --output project/artifacts/reports/tables_draft.tex `
    --require-backbones siglip,siglip_lora,siglip_lora_loco,clip
```

The generated tables cover:
- Cross-city LOCO headline — 5 backbones + ensemble × 5 targets.
- In-city spatial CV — best backbone per city.
- f_ped ablation — SigLIP frozen vs SigLIP `vision_only`.

The cross-city headline and in-city spatial CV tables are both computed against `walkability_index_v2`. The generator also emits extra rows (per-target ablation grids, modality breakdowns) for diagnostic checking; they aren't part of the headline tables.

---

## Utility scripts

| Script | Purpose |
|---|---|
| `verify_repo.py` | Checks that all expected files are present |
| `audit_v2_lock.py` | Verifies per-city lock IDs against the master CSV |
| `eval_model_vs_ps.py` | Compares model AUROC vs Project Sidewalk ground truth (Seattle) |
| `run_full_matrix.py` | Convenience wrapper: runs the 6-direction matrix in one call |
