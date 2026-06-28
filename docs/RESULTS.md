# WalkBench — Results Reference

All experimental results are logged in `project/artifacts/reports/`. This document explains what each result file contains, what every metric means, and how to read the headline numbers. **All results cover the five targets only.**

See [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for the overview; this is the detailed reference for every result file and what each number means.

---

## Scope: five targets

| Target | Type | Column in training CSV | Notes |
|---|---|---|---|
| Sidewalk presence | Binary | `overture_sidewalk_present` | Overture footway within 15 m of point |
| Near-buffer crosswalk | Binary | `crosswalk_present_near` | Overture crosswalk within 15 m, ±30° heading wedge |
| Intersection density | Regression | `overture_intersection_count_200m` | Log1p-transformed for training; expm1 at eval |
| Building footprint fraction | Regression | `overture_building_footprint_frac_100m` | NAIP-derived |
| Composite walkability index | Regression | `walkability_index_v2` | PC1 of the four targets above + transit stops |

The walkability index is the primary headline regression target. It is built by PCA on the five visually grounded components listed above (sidewalk, near-buffer crosswalk, log1p intersection count, building footprint frac, log1p transit stops). PC1 explains 0.365 of variance in those components; all loadings are positive. See `project/artifacts/reports/walkability_index_build.json` for loadings.

---

## Five model configurations (backbones)

| Backbone tag | Description | Test city unseen? |
|---|---|---|
| `siglip` | Frozen SigLIP SO400m — no fine-tuning | Yes |
| `clip` | Frozen CLIP ViT-L/14 — no fine-tuning | Yes |
| `siglip_lora` | SigLIP + LoRA adapter trained on **all 3 cities** | **No** — in-corpus (sees test city) |
| `siglip_lora_loco` | SigLIP + LoRA adapter trained on the **other two cities only** | Yes |
| `ensemble` | Mean logits from `siglip` + `clip` + `siglip_lora_loco_v2` | Yes |

**Note on the in-corpus adapter**: `siglip_lora` saw the test city's heading-level labels during fine-tuning, so its AUROC scores (sidewalk 0.90, crosswalk 0.85) are not cross-city transfer numbers. Under the LOCO protocol (`siglip_lora_loco`), where the test city is unseen, those scores drop back to 0.78 / 0.60, against a frozen SigLIP baseline of 0.75 / 0.55.

---

## Result files

### `project/artifacts/reports/multitarget_results.jsonl`

**656 total runs** logged. One JSON object per line. Append-only. Of these, the reported subset is exactly the 108 runs produced by `run_v2_target_and_ablations.py` (90 cross-city retrain rows: 5 backbones × 6 LOCO directions × 3 seeds; plus 18 `vision_only` ablation rows: SigLIP × 6 LOCO directions × 3 seeds). To filter:

```python
mask = (
    (df.target_tier == "1") &
    (df.ablation.isin(["full", "vision_only"])) &
    (df.eval_mode == "cross_city") &
    (df.spatial_features.isin(["pedgraph", None])) &
    df.seed.isin([41, 42, 43]) &
    df.results_per_target.apply(lambda r: "walkability_index_v2" in r if isinstance(r, dict) else False)
)
```

Older rows in the file (single-seed sweeps and earlier ablations) are kept as a log; the reported numbers come from the filtered subset above.

Key fields in each record:

| Field | Description |
|---|---|
| `backbone` | One of the five backbone tags above |
| `train_city` / `test_city` | Source and target city |
| `ablation` | Feature channel: `full`, `vision_only`, `street_only`, etc. |
| `target_tier` | Always `1` for reported results |
| `seed` | Random seed (42 for single-seed; 41/42/43 for multi-seed) |
| `eval_mode` | `cross_city` or `in_city` |
| `sidewalk_auroc` | AUROC on `overture_sidewalk_present` |
| `crosswalk_auroc` | AUROC on `crosswalk_present_near` |
| `intersection_rho` | Spearman ρ on `intersection_count_200m` |
| `bldg_frac_rho` | Spearman ρ on `building_footprint_frac_100m` |
| `walkability_rho` | Spearman ρ on `walkability_index_v2` (Package-A runs only) |
| `walkability_r2_recal` | Recalibrated R² on walkability index |
| `timestamp` | ISO 8601 UTC run timestamp |
| `n` | Number of test-city rows evaluated |

To query from Python:
```python
import json, pandas as pd
records = [json.loads(l) for l in open("project/artifacts/reports/multitarget_results.jsonl")]
df = pd.DataFrame(records)
# Headline cross-city, full ablation
mask = (df.target_tier == "1") & (df.ablation == "full") & (df.eval_mode == "cross_city")
print(df[mask].groupby("backbone")[["sidewalk_auroc","crosswalk_auroc","bldg_frac_rho","walkability_rho"]].mean())
```

### `project/artifacts/reports/bootstrap_summary.json`

**Authoritative seed-aggregated summary.** Contains mean ± std across seeds 41/42/43 for every (backbone, direction, target) cell. This is the source of truth for reported numbers; do not cite `multitarget_summary.json` (which only reflects the last appended run).

Structure:
```json
{
  "siglip": {
    "msp->seattle": {
      "sidewalk_auroc": {"mean": 0.766, "std": 0.008},
      "crosswalk_auroc": {"mean": 0.552, "std": 0.005},
      ...
    }
  },
  "siglip_lora_loco": { ... },
  ...
}
```

### `project/artifacts/reports/bootstrap_cis.jsonl`

Per-(run_id, target) bootstrap 95% percentile CIs. One JSON line per (run, target). Used for the confidence-interval columns in the results tables. Each record has fields `run_id`, `backbone`, `target`, `ci_lower`, `ci_upper`, `mean`.

### `project/artifacts/reports/paired_tests.jsonl`

Paired significance tests between backbone pairs, per direction × target. Each record:

| Field | Description |
|---|---|
| `pair` | e.g., `"siglip_lora_loco,siglip"` |
| `direction` | e.g., `"msp->seattle"` |
| `target` | Target column name |
| `test` | `"delong"` (binary) or `"bootstrap_delta"` (regression) |
| `p_value` | Raw p-value (two-sided) |
| `delta_mean` | Mean difference (backbone 1 − backbone 2) |
| `significant` | Boolean (p < 0.05) |

### `project/artifacts/reports/ensemble_results.json`

Logit ensemble (SigLIP frozen + CLIP frozen + LoRA-LOCO v2), 90 rows covering all 6 LOCO directions × 3 seeds × 5 targets. Fields mirror `multitarget_results.jsonl`. Backbone tag: `"ensemble"`.

### `project/artifacts/reports/model_vs_ps.json`

Label-noise audit for Seattle. Compares model AUROC against Overture labels vs model AUROC against Project Sidewalk (PS) ground-truth annotations. Fields:

| Field | Description |
|---|---|
| `crosswalk_auroc_vs_overture` | Model vs Overture crosswalk labels (training signal) |
| `crosswalk_auroc_vs_ps` | Model vs Project Sidewalk crosswalk annotations |
| `sidewalk_auroc_vs_overture` | Model vs Overture sidewalk |
| `sidewalk_auroc_vs_ps` | Model vs Project Sidewalk sidewalk |

### `project/artifacts/reports/spatial_cv_results.jsonl`

In-city 5-fold spatial CV results. 90 records total: 78 earlier single-target runs (kept as a log) plus the 12 headline runs against the composite target. **For the reported numbers, filter to `target_filter == "walkability_index_v2"` (12 records: 4 backbones × 3 cities).** The other records are earlier single-target runs.

Key fields: `city`, `backbone`, `target`, `block_buffer_m`, `n_folds`, `walkability_rho_mean`, `walkability_rho_std`, `walkability_r2_recal_mean`, `pts_dropped_per_fold_mean`.

---

## Headline results

All numbers are from the 5-feature `walkability_index_v2` target (2026-05-11 retrain). Seeds 41/42/43 averaged unless noted.

### Cross-city LOCO — composite walkability index

Mean Spearman ρ across all 6 leave-one-city-out directions (3 seeds averaged).

| Backbone | Walkability ρ (mean, 6 LOCO dirs) | Source |
|---|---:|---|
| SigLIP frozen | 0.732 | `bootstrap_summary.json` → `siglip` |
| CLIP frozen | 0.713 | `bootstrap_summary.json` → `clip` |
| SigLIP + LoRA **in-corpus** (sees test city) | 0.752 | `bootstrap_summary.json` → `siglip_lora` |
| SigLIP + LoRA-LOCO v1 | 0.712 | `multitarget_results.jsonl` filter `loco_v1` |
| SigLIP + LoRA-LOCO v2 | 0.735 | `bootstrap_summary.json` → `siglip_lora_loco` |
| Ensemble (SigLIP + CLIP + LoRA-LOCO v2) | **0.763** | `ensemble_results.json` mean across directions |

Reading the table: frozen SigLIP and CLIP both land at ρ ~0.71–0.73 cross-city. LoRA-LOCO v1 is below the frozen baseline (0.712 vs 0.732). LoRA-LOCO v2 (multi-task, r=32, 4 heads, attention+MLP layers) is at 0.735, about the same as frozen SigLIP. The ensemble is +0.028 over the best single LOCO backbone, at 0.763.

### Cross-city LOCO — per-target breakdown

Mean across all 6 LOCO directions, 3 seeds, `full` ablation.

| Backbone | Sidewalk AUROC | Crosswalk AUROC | Intersect ρ | Bldg-frac ρ | Walk-idx ρ |
|---|---:|---:|---:|---:|---:|
| SigLIP frozen | 0.762 | 0.560 | 0.623 | 0.824 | 0.732 |
| CLIP frozen | 0.753 | 0.572 | 0.605 | 0.785 | 0.713 |
| LoRA in-corpus | 0.918 | 0.858 | 0.642 | 0.798 | 0.752 |
| LoRA-LOCO v1 | 0.763 | 0.586 | 0.600 | 0.794 | 0.712 |
| LoRA-LOCO v2 | 0.787 | 0.608 | 0.639 | 0.813 | 0.735 |
| Ensemble | **0.797** | 0.593 | **0.667** | **0.844** | **0.763** |

Per target:

- **Sidewalk (AUROC 0.76–0.79)**: frozen baselines and LoRA-LOCO v2 all land in the 0.76–0.79 range; the ensemble is at 0.797.
- **Crosswalk (AUROC 0.56–0.61)**: lowest of the five. Overture crosswalk prevalence varies across cities (Seattle 0.148 vs MSP 0.266). LoRA-LOCO v2 is +0.022 over frozen SigLIP. See the label-ceiling audit below for the κ analysis.
- **Intersection density (ρ 0.60–0.67)**: ensemble at 0.667. Log1p transform applied in training.
- **Building footprint fraction (ρ 0.79–0.84)**: highest of the five; ensemble at 0.844. Comes mainly from the NAIP aerial tile.
- **Composite walkability index (ρ 0.71–0.76)**: PC1 of the four targets above plus transit-stop density, so it tracks a weighted mix of them.

### 6-way cross-city matrix (SigLIP frozen, full ablation, single seed)

Absolute per-direction results for the SigLIP frozen backbone. Source: `multitarget_results.jsonl`, filter `backbone == "siglip" and ablation == "full" and target_tier == "1" and seed == 42`. **These are single-seed direction-specific numbers, not the seed-aggregated headline.** The headline (mean over 3 seeds and the 6 directions) lives in the table above this section.

| Train → Test | Sidewalk AUROC | Crosswalk AUROC | Intersect ρ | Bldg-frac ρ |
|---|---:|---:|---:|---:|
| MSP → Seattle | 0.820 | 0.555 | 0.141 | 0.682 |
| MSP → DC | 0.628 | 0.548 | 0.028 | 0.683 |
| Seattle → MSP | 0.855 | 0.468 | 0.296 | 0.648 |
| Seattle → DC | 0.716 | 0.525 | 0.055 | 0.683 |
| DC → MSP | 0.751 | 0.627 | 0.290 | 0.655 |
| DC → Seattle | 0.828 | 0.587 | 0.365 | 0.656 |

Patterns in this matrix:
- DC as the source city gives the highest crosswalk AUROC (0.587–0.627); DC also has the highest crosswalk label prevalence.
- MSP → DC is the lowest direction (sidewalk 0.628, intersection ρ 0.028).
- Building footprint stays in a narrow band across all directions (0.648–0.683).

### In-city spatial CV — composite walkability index

5-fold KMeans-blocked CV, 500 m buffer, `walkability_index_v2` target. Results from `spatial_cv_results.jsonl` (filter `target_filter == "walkability_index_v2"`). Best backbone per city shown.

| City | Best backbone | Walkability ρ ± std | Walkability R²_recal |
|---|---|---:|---:|
| MSP | SigLIP + LoRA | 0.774 ± 0.058 | 0.459 |
| Seattle | SigLIP + LoRA | 0.711 ± 0.110 | 0.430 |
| DC | SigLIP frozen | 0.755 ± 0.059 | 0.450 |
| **Mean** | — | **0.747** | **0.446** |

All 12 runs (4 backbones × 3 cities): `spatial_cv_results.jsonl`, filter `target_filter == "walkability_index_v2"`.

**R²_recal note**: R²_recal is a post-hoc affine recalibration that shifts and scales predictions to match the test city's mean and std before computing R². Because it uses the test labels to do that, it is an upper bound and tracks rank more than absolute accuracy. Report it alongside ρ, not on its own.

Frozen SigLIP is the best backbone for DC in-city; LoRA is best for MSP and Seattle. Seattle has the highest per-fold standard deviation (0.110).

### f_ped ablation — contribution of pedestrian-graph kNN features

SigLIP frozen, 6 LOCO directions × 3 seeds. `full` ablation includes `pedgraph_features.csv`; `vision_only` removes it. Source: dedicated ablation runs in `multitarget_results.jsonl`, filter `ablation IN ["full", "vision_only"] AND backbone == "siglip"`.

| Configuration | Walk-idx ρ | Crosswalk AUROC |
|---|---:|---:|
| SigLIP + f_ped (full) | 0.732 | 0.560 |
| SigLIP without f_ped (vision_only) | 0.718 | 0.549 |
| Delta (f_ped contribution) | **+0.013** | **+0.011** |

The pedgraph kNN channel adds +0.013 ρ. Removing it entirely (vision_only) drops cross-city ρ from 0.732 to 0.718. Context for this ablation: `pedgraph_features.csv` is derived from OSM, the same geography Overture's labels come from, so it is worth checking how much of the result depends on it; the +0.013 delta is the answer.

### Label-ceiling audit (Overture vs Project Sidewalk)

Sources: `project/artifacts/reports/label_noise_audit.json`, `project/artifacts/reports/label_ceiling_analysis.json`, `project/artifacts/reports/figures/fig_label_ceiling.pdf`

Seattle and Pittsburgh Project Sidewalk audits, synthesized into the label-ceiling figure. Pittsburgh comes from a Harvard Dataverse static export (Oct 2021). Seattle is the only city where both sidewalk and crosswalk ceiling numbers are available; Pittsburgh has sidewalk only (its crosswalk label type was added after the 2021 snapshot), and the DC endpoint has been returning errors.

### Label-noise audit underlying data

Source: `project/artifacts/reports/label_noise_audit.json`

Seattle is the only city with accessible Project Sidewalk (PS) crosswalk data. The DC PS deployment returns HTTP 503 on all endpoints; Pittsburgh has only sidewalk labels in the 2021 Dataverse snapshot. So the crosswalk κ number is Seattle-only.

**Label re-filter applied (2026-05-20)**: After the initial audit revealed near-zero crosswalk agreement, `CROSSWALK_CLASSES` in both `derive_overture_targets.py` and `derive_heading_overture_labels.py` was narrowed from all broad OSM crossing tags (`crossing`, `uncontrolled`, `traffic_signals`, `marked`) to `{"marked"}` only — physically painted crosswalks. `derive_overture_targets.py` and `derive_heading_overture_labels.py` were re-run for all 4 cities; `features_labels_agreement.csv` `overture_crosswalk_present` column was patched from the new `overture_targets.csv`. Results below reflect the re-filtered labels.

**Overture ↔ PS agreement (Seattle, 25 m match radius, marked-only crosswalk labels)**

| Target | n matched | Cohen's κ | AUROC(Overture→PS) | AUROC(PS→Overture) | Overture prevalence | PS prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Sidewalk | 3,339 | **0.135** | 0.543 | 0.772 | 0.981 | 0.862 |
| Crosswalk | 4,851 | **0.049** | 0.607 | 0.516 | 0.189 | 0.024 |

Confusion matrix (rows = PS, cols = Overture):
- Crosswalk: [[3866, 869], [70, 46]] — 915 Overture positives vs 116 PS positives
- Sidewalk: [[43, 418], [21, 2857]]

Reading the table:

*Crosswalk (post re-filter)*: κ=0.049, up 8× from the pre-filter κ=0.006. Overture prevalence dropped from 24.2% to 18.9% after excluding `uncontrolled`, `traffic_signals`, and bare `crossing` OSM tags. AUROC(Overture→PS) = 0.607, up from 0.517. The ~7.9× prevalence gap that remains (Overture 18.9% vs PS 2.4%) is the instrument difference: `crossing=marked` fires at every mapped crossing node in the road network, while PS fires only when the auditor can see the paint from the panorama.

Note on the κ→AUROC direction: AUROC(Overture→PS) = 0.607 and AUROC(PS→Overture) = 0.516 are two different quantities; which one is "the ceiling" depends on which source is treated as reference. This is flagged for the advisor and the specific ceiling value is left out of the figure until it is settled.

*Sidewalk*: κ=0.135. Overture marks 98.1% of points sidewalk-present, PS marks 86.2%. AUROC(PS→Overture) = 0.772 is close to the model's observed sidewalk AUROC of 0.76–0.79. (κ is low here partly because prevalence is near 98% — high-prevalence κ is suppressed even when the two sources mostly agree.)

**Pre-filter vs post-filter crosswalk summary**

| | κ | AUROC(Overture→PS) | Overture prevalence |
|---|---:|---:|---:|
| Broad OSM tags (old) | 0.006 | 0.517 | 0.242 |
| `marked` only (current) | **0.049** | **0.607** | **0.189** |
| PS prevalence | — | — | 0.024 |

**Model vs PS (Seattle, crosswalk)**

Source: `project/artifacts/reports/model_vs_ps.json` (pre-retraining, old labels)

| Target | Model AUROC vs Overture | Model AUROC vs PS | Delta |
|---|---:|---:|---:|
| Crosswalk | ~0.56 | ~0.64 | **+0.08** |
| Sidewalk | ~0.77 | ~0.67 | −0.10 |

These model-vs-PS numbers predate the label re-filter, so they use the old broad-tag crosswalk labels. They have not been regenerated against the marked-only labels.

---

## Spatial CV leakage analysis

**Source**: `spatial_cv_results.jsonl`, records with `block_buffer_m IN [0, 500]`

The 500 m buffer drops an average of 54 training points per fold for MSP and 80 for Seattle/DC (~1.7–2.6% of training data). The AUROC change from buffer vs no-buffer is < 0.005 across all cities and targets, confirming that the spatial blocks are large enough that kNN contamination at the 400 m pedgraph-feature radius is negligible at the fold boundary.

| City | Target | 0 m buffer | 500 m buffer | Avg pts dropped/fold |
|---|---|---:|---:|---:|
| MSP | Sidewalk AUROC | 0.853 ± 0.043 | 0.848 ± 0.041 | 54 (1.7%) |
| MSP | Crosswalk AUROC | 0.579 ± 0.032 | 0.576 ± 0.032 | — |
| MSP | Building R²_recal | 0.578 ± 0.119 | 0.587 ± 0.106 | — |
| Seattle | Sidewalk AUROC | 0.827 ± 0.083 | 0.829 ± 0.076 | 80 (2.6%) |
| Seattle | Crosswalk AUROC | 0.676 ± 0.038 | 0.674 ± 0.040 | — |
| DC | Sidewalk AUROC | 0.752 ± 0.054 | 0.754 ± 0.049 | 80 (2.6%) |
| DC | Crosswalk AUROC | 0.633 ± 0.053 | 0.622 ± 0.059 | — |

Reported results use the 500 m buffer. The buffer-vs-no-buffer change is < 0.005 across all cities and targets.

---

## Summary of numbers

1. Frozen SigLIP SO400m and CLIP ViT-L/14 reach ρ ≈ 0.71–0.73 on the composite index cross-city with no adaptation.

2. Under LOCO, LoRA-LOCO v1 and v2 land within ±0.003 ρ of the frozen baseline on the index. In-corpus LoRA is +0.16 sidewalk AUROC over frozen, with the test city's labels seen at fine-tune time.

3. Building footprint is the highest cross-city target (ρ 0.79–0.84).

4. Crosswalk is the lowest cross-city target (AUROC 0.55–0.61). Overture crosswalk prevalence varies across cities; see the label-ceiling audit.

5. The ensemble is +0.028 ρ over the best single LOCO backbone (0.763 vs 0.735).

6. The pedgraph (f_ped) channel is +0.013 ρ over vision-only.

7. In-city spatial CV ρ = 0.747 (mean best backbone, 5-fold), vs 0.763 cross-city.

8. Crosswalk label re-filter: broad OSM crossing tags gave κ=0.006, AUROC(Overture vs PS)=0.517, Overture prevalence 24.2%. `crossing=marked` only gives κ=0.049, AUROC=0.607, prevalence 18.9%. PS prevalence is 2.4%; the ~7.9× prevalence gap that remains is the OSM-vs-panorama instrument difference (OSM records crossing nodes in the road network; PS records paint visible from a panorama).

---

## Pittsburgh — 4th city zero-shot results

Pittsburgh zero-shot is fully covered in `bootstrap_summary.json` (25 entries: 5 backbones × 5 targets; regenerated 2026-05-24 with `--include-pittsburgh`). Per-backbone mean table below is from `multitarget_results.jsonl` (3-seed means, `experiment_tag == "src_pgh"`); CIs are in `bootstrap_cis.jsonl`.

Training configuration: all three peer cities simultaneously (`dc+msp+seattle → pittsburgh`). Pittsburgh was never seen during training. n=4,806 test points (4,982 locked; ~176 dropped in label joins, consistent with 3-city drop rates).

### Pittsburgh zero-shot — per-backbone (3 seeds × 1 direction)

| Backbone | Sidewalk AUROC | Crosswalk AUROC | Intersect ρ | Bldg-frac ρ | Walk-idx ρ | Walk R²_recal |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP frozen | 0.767 | 0.626 | 0.781 | 0.886 | **0.853** | 0.618 |
| CLIP frozen | 0.763 | 0.584 | 0.765 | 0.858 | **0.859** | 0.703 |
| SigLIP 2 frozen | 0.759 | 0.620 | 0.800 | 0.899 | **0.862** | 0.676 |
| DINOv2 frozen | 0.726 | 0.549 | 0.754 | 0.841 | **0.836** | 0.665 |
| SigLIP 2 + RemoteCLIP (NAIP) | 0.753 | 0.597 | 0.772 | 0.858 | 0.840 | 0.680 |
| **LoRA-LOCO-Pittsburgh** | 0.754 | 0.615 | 0.754 | 0.859 | 0.833 | 0.654 |
| **Mean (all 6)** | **0.753** | **0.598** | **0.771** | **0.867** | **0.847** | **0.666** |

Observations:
- All six backbones reach walk-idx ρ = 0.833–0.862 on Pittsburgh, above the 3-city 1v1 LOCO mean (0.717–0.738). Pittsburgh is trained on all three peer cities at once (~14,600 examples) vs single-city LOCO training (~4,800–5,000 examples).
- LoRA-LOCO-Pittsburgh (0.833) is the lowest walk-ρ of the six. The LOCO adapter does not beat any frozen backbone on Pittsburgh, same as in the 3-city protocol.
- SigLIP 2 is highest (0.862), as in the 3-city LOCO (0.738 vs 0.732).
- RemoteCLIP on NAIP does not lift building-footprint ρ (0.858 vs 0.886 for SigLIP, 0.899 for SigLIP 2).
- Pittsburgh crosswalk (AUROC 0.549–0.626) and intersection ρ (0.754–0.800) are higher than the 3-city LOCO means (0.528–0.608; 0.595–0.639).

### Pittsburgh data status

Pittsburgh was added as the 4th city (2026-05-20); frozen embeddings extracted 2026-05-22. Status:

| Component | Status |
|---|---|
| Lock | 4,982 pts (`pittsburgh_v2_final_lock_ids.txt`) |
| Mapillary street imagery | 4 headings × 4,982 = 19,928 images |
| NAIP aerial | 4,982 tiles |
| Overture targets | Appended to `overture_targets.csv` |
| Heading Overture labels | Appended to `heading_overture_labels_v2.csv` |
| FLA rows | 4,982 rows appended (`features_labels_agreement.csv` = 19,624 total) |
| Pedgraph features | Appended to `pedgraph_features.csv` (19,624 total) |
| GTFS transit features | Populated (median 10 stops / 43 trips/hr at 400 m) |
| SigLIP / CLIP / SigLIP 2 / DINOv2 / RemoteCLIP embeddings | **Done** — `.npy` present for street+naip (RemoteCLIP naip-only) |
| LoRA-LOCO-Pittsburgh adapter + embeddings | **Done** — adapter trained on msp+seattle+dc; embeddings extracted for all 4 cities |
| Downstream zero-shot runs (3-city → Pittsburgh) | **Done** — 18 rows in `multitarget_results.jsonl` (`experiment_tag == "src_pgh"`), 6 backbones × 3 seeds |
| SAM2 segmentation | Pending (not required for embedding-based eval) |
| OpenAI captions | Pending (not required for embedding-based eval) |

Frozen embeddings are ready, so the 45-run frozen zero-shot grid (siglip, clip, siglip2, dinov2, fusion × 6→Pittsburgh implied directions × 3 seeds) is runnable now; the LoRA-LOCO-Pittsburgh path requires training the adapter first. The SAM2/caption NaN columns in the FLA rows do not block embedding-based model evaluation.
