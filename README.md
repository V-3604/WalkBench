# WalkBench — a cross-city walkability benchmark

WalkBench tests whether street-level and aerial visual features can predict walkability
and transfer to a city the model was never trained on. It's a four-city benchmark
(Minneapolis–St. Paul, Seattle, Washington DC, Pittsburgh) with frozen vision encoders,
a leave-one-city-out evaluation protocol, and Pittsburgh held out as a zero-shot fourth
city.

[`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) walks through the whole project end
to end — the question, the data, the method, and the results.

- **Cities / scale**: MSP 4,847 · Seattle 4,851 · DC 4,994 · Pittsburgh 4,982 — 19,624
  points in the master table.
- **Inputs per point**: four Mapillary street images (0/90/180/270°) + one NAIP aerial tile.
- **Targets**: sidewalk presence, near-buffer crosswalk, intersection density,
  building-footprint fraction, and a composite walkability index (PC1 of those plus
  transit-stop density).

## Docs

- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — the whole project, end to end.
- [`docs/DATA.md`](docs/DATA.md) — data layout, schemas, checksums, and what to download.
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — the stages from raw geography to results.
- [`docs/RESULTS.md`](docs/RESULTS.md) — every result file and number explained.
- [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) — the papers this builds on.
- [`docs/REPO_MAP.md`](docs/REPO_MAP.md) — where every file and number lives.

## Setup

Two Python environments: the Windows venv for the data pipeline, and a WSL venv with
CUDA for anything that touches the GPU (embedding extraction, LoRA fine-tuning).

```powershell
# Windows (data pipeline)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# WSL (GPU work) — repo mounted at /mnt/c/Users/kvars/Desktop/WalkCLIP
.venv_wsl/bin/python project/scripts/train_multitarget.py --help
```

Run GPU jobs in native WSL, not over `/mnt/c` — reading images through the WSL 9P
filesystem layer caps GPU utilization to near nothing.

## What's in the repo vs. what you download

In git after `git clone`:
- All code under `project/scripts/`.
- Label CSVs, lock files, point tables, and spatial features under `project/data/processed/`.
- Result JSON/JSONL/LaTeX tables under `project/artifacts/reports/`, and prediction
  bundles (`.npz`) under `project/artifacts/predictions/`.

From the Google Drive zip (too big for git — see [`docs/DATA.md`](docs/DATA.md)):
- `project/artifacts/embeddings/` — the `.npy` arrays (~3.3 GB).
- `project/artifacts/models/` — LoRA adapter directories (~1.3 GB).
- `project/data/raw/` — raw imagery (~7 GB; only needed to re-extract embeddings).

After placing the embeddings, check the data lock:

```powershell
.\.venv\Scripts\python.exe project\scripts\audit_v2_lock.py
```

## Running experiments

All the runs behind the reported numbers are already done and logged under
`project/artifacts/reports/` (`bootstrap_summary.json` is the authoritative source). To
reproduce a single cross-city run (embeddings must be in place):

```bash
.venv_wsl/bin/python project/scripts/train_multitarget.py \
  --train-city msp --test-city seattle \
  --target-tier 1 --ablation full --pca-dim 128 \
  --spatial-features pedgraph --backbone siglip --seed 42 --save-predictions
```

## Key invariants

- **Join key is always `(point_id, city)`.** `point_id` is a per-city 0-based integer;
  never join on `point_id` alone.
- **`features_labels_agreement.csv` is the master training table** — don't edit it.
- **The scaler is fit on the training city only**, never on train + test combined.
- **Seed 42** by default; runs are deterministic on the same GPU.
- **Never commit `.env`** — it holds `OPENAI_API_KEY` and `MAPILLARY_ACCESS_TOKEN`.

## Repo layout

```
WalkBench/
├── docs/                  ← documentation (PROJECT_OVERVIEW.md is the entry point)
├── project/
│   ├── data/processed/    ← locks, labels, point tables, spatial features
│   ├── scripts/           ← the pipeline and training code
│   └── artifacts/
│       ├── embeddings/    ← .npy arrays (gitignored; from the Drive zip)
│       ├── models/        ← LoRA adapters (gitignored; from the Drive zip)
│       ├── predictions/   ← .npz prediction bundles (in git)
│       └── reports/       ← result JSON/JSONL + LaTeX tables (in git)
├── sam2/                  ← SAM2 weights
├── requirements.txt
└── README.md
```
