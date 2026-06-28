# WalkBench: how far does walkability prediction travel between cities?

## The short version

I built a four-city benchmark for predicting walkability from street-level and aerial images, and I tested it by training on some cities and evaluating on a city the model has never seen.

The main finding is that frozen, off-the-shelf vision features (SigLIP, CLIP, etc.) transfer across US cities better than I expected. 
A composite walkability score reaches Spearman ρ ≈ 0.73 under leave-one-city-out evaluation, and ρ ≈ 0.85 on Pittsburgh as a completely held-out fourth city. The catch is that fine-tuning the encoder on the labels doesn't actually beat the simple frozen baseline once you stop letting the model see the test city. And the hardest target, crosswalks, is limited by how noisy the label is rather than by the model.

So the short version: a simple frozen baseline is hard to beat here. Fine-tuning doesn't add anything once the test city is held out, and the crosswalk target is held back by label noise rather than the model.

## The question

Can modern vision encoders predict walkability from images, and do those predictions still hold in a city the model was never trained on? Most of the work I read trains and tests inside the same city, or pools all the cities together before splitting. Both inflate the score. I wanted the number you actually get when you move to a new city.

## The benchmark: four cities, ~19,600 points

| City | Points | Role |
|---|---:|---|
| Minneapolis–St. Paul | 4,847 | train / leave-one-out |
| Seattle | 4,851 | train / leave-one-out |
| Washington DC | 4,994 | train / leave-one-out |
| Pittsburgh | 4,982 | held-out 4th city (zero-shot only) |

That's 19,624 rows in the master table, `features_labels_agreement.csv`. For each point
I have four Mapillary street images (facing 0/90/180/270°) and one NAIP aerial tile.
Points are intersection-like nodes from the OSMnx walking network, kept only if Mapillary has coverage nearby, and spatially stratified so downtown (higher coverage areas) doesn't dominate.
Pittsburgh ran through the exact same pipeline but I only ever use it for zero-shot testing. Holding a whole city out like this is a stronger test than the leave-one-out
rotation among the first three cities, because nothing about Pittsburgh is ever part of training. Full schemas and checksums are in [`DATA.md`](DATA.md).

## What I predict

Five targets, all derived from Overture Maps:

- sidewalk present (yes/no)
- crosswalk present near the point (yes/no)
- intersection density nearby (a count)
- building-footprint fraction nearby (from the aerial tile)
- a composite **walkability index**: the first principal component of the four above plus GTFS transit-stop density. This is the main number I report.
  PC1 explains 0.365 of the variance in those components and all the loadings are positive. Because it's my own construct, I'm also checking it against the commercial Walk Score as a sanity test (`project/scripts/correlate_walkscore.py`).

[`RESULTS.md`](RESULTS.md) defines each target and how it's scored.

## The method

1. **Frozen image features.** I pull embeddings from encoders I never fine-tune:
   SigLIP SO400m and SigLIP 2 (vision-language), CLIP ViT-L/14, DINOv2 (self-supervised,
   no text at all), and RemoteCLIP (aerial-specific, used on the NAIP tile only). Each
   modality gets PCA-reduced to 128 dims, and the PCA is fit on the training cities only
   so nothing leaks from the test city.
2. **A small head.** An MLP fuses the street features, the aerial features, and a
   pedestrian-graph context feature, then predicts all five targets.
3. **Fine-tuning, two ways.** I also tried LoRA adapters on SigLIP:
   - *in-corpus*: the adapter sees every city during fine-tuning, including the labels of
     the city I later test on. That lets it peek at the answer, so it's a best-case
     ceiling, not a fair cross-city score.
   - *leave-one-city-out (LOCO)*: the adapter is fine-tuned only on the other cities, so
     the test city stays completely unseen. The headline LoRA numbers all use this one.
4. **Evaluation.** Leave-one-city-out across MSP/Seattle/DC (six train→test directions),
   Pittsburgh as a zero-shot fourth city, and an in-city spatial cross-validation with a
   500 m buffer to make sure nearby points aren't leaking across the train/test split.

## Results

Composite walkability index, mean Spearman ρ across the six leave-one-city-out
directions (averaged over 3 seeds):

| Model | Walkability ρ |
|---|---:|
| CLIP, frozen | 0.713 |
| SigLIP, frozen | 0.732 |
| SigLIP + LoRA (LOCO, test city unseen) | 0.735 |
| SigLIP 2, frozen | 0.738 |
| Ensemble (SigLIP + CLIP + LoRA-LOCO) | **0.763** |
| SigLIP + LoRA (in-corpus, sees test city) | 0.752 |

A few things worth pulling out:

1. **Frozen features transfer.** Plain SigLIP and CLIP land around ρ 0.71–0.73 across
   city pairs with no adaptation at all.
2. **The fine-tuning "win" is leakage.** The in-corpus LoRA model scores very high on the
   binary targets (sidewalk AUROC 0.90, crosswalk 0.85), but only because the adapter
   saw the test city's labels during training. Under the LOCO setup, where the test city
   is unseen, those scores drop back down close to the frozen baseline (sidewalk 0.78 vs
   0.75 frozen, crosswalk 0.60 vs 0.55 frozen), so the big in-corpus lead disappears.
3. **Some targets travel, some don't.** Building-footprint fraction is the most
   portable (ρ 0.79–0.84) because aerial building geometry looks the same in every city.
   Crosswalks are the worst (AUROC 0.55–0.61).
4. **Pittsburgh, never seen, holds up.** All the backbones reach ρ 0.833–0.862 zero-shot
   on Pittsburgh, and the LoRA adapter (test city unseen) is the *lowest* of the group
   there, the same pattern as the three-city test, on a city the model truly never touched.
5. **The crosswalk label is noisy.** Overture and Project Sidewalk (an independent
   crowd-sourced source) agree on crosswalks at only κ = 0.05, which mathematically caps
   the achievable AUROC against that label near 0.61. So 0.55–0.61 is about as high as this
   label allows.
6. **The cross-city numbers survive a spatial-leakage check.** In-city spatial CV gives
   ρ ≈ 0.75, and adding a 500 m buffer between train and test points moves AUROC by less
   than 0.005.

## Where everything lives

- [`DATA.md`](DATA.md) — every data file, its schema and checksum, and what to download
  versus what's already in the repo.
- [`PIPELINE.md`](PIPELINE.md) — the stages from raw geography to final results.
- [`RESULTS.md`](RESULTS.md) — every result file and what each number means.
- [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) — the papers this builds on.
- `project/scripts/` — the pipeline and training code. Training runs in WSL
  (`.venv_wsl`) for CUDA; the data scripts run in the Windows `.venv`.
- `project/artifacts/reports/bootstrap_summary.json` — the authoritative seed-averaged
  numbers. (The raw `multitarget_results.jsonl` is an append-only log spanning many runs
  and schema versions; trust `bootstrap_summary.json` for anything reported.)
