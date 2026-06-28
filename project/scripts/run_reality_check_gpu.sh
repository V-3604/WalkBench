#!/usr/bin/env bash
# Phase 2(b) of the WalkBench reality-check -- the GPU rungs (frozen MLP + LoRA + LoRA-LOCO).
#
# Per cross-city direction x seed, trains the rungs that need the GPU, via
# train_multitarget.py --save-predictions, writing npz bundles consumed unchanged by
# the Phase-3 spatial bootstrap:
#   frozen MLP head   --backbone siglip2                      rung 2 (best-frozen anchor)
#   in-corpus LoRA    --backbone siglip_lora                  rung 3 (LOCO directions only)
#   LoRA-LOCO         --backbone siglip_lora_loco_v2_<test>   rung 4 (honest, test city held out)
# The in-corpus LoRA has no Pittsburgh embedding, so the Pittsburgh zero-shot direction
# runs only the frozen MLP + the pittsburgh-holdout LoRA-LOCO adapter. ~200 runs.
#
# RESUMABLE: a run whose npz already exists is skipped, so re-running continues where
# it stopped (never overwrites completed work).
# PARALLEL: runs JOBS direction-streams at once (default 2) to use spare GPU capacity --
# the small frozen-head model leaves the card ~70% idle, so 2-3 concurrent ~halve wall
# time. Concurrency is *by direction* so the two live runs always write different
# checkpoints (no .pt collision); each direction's own runs stay sequential.
# Caveat: concurrent runs append to the shared multitarget_results.jsonl / summary.json,
# which may interleave -- harmless, Phase 3 reads the npz, not those.
#
# Add the same-encoder frozen MLP baseline:  MLP_BACKBONES="siglip siglip2" ...
# More/less parallelism:  JOBS=3 ...   (each run uses ~2 GB VRAM)
#
# Run (WSL, repo ROOT):
#   wsl -e bash -lc "cd /mnt/c/Users/kvars/Desktop/WalkCLIP && bash project/scripts/run_reality_check_gpu.sh"
set -uo pipefail

PY="${PY:-.venv_wsl/bin/python}"
SEEDS="${SEEDS:-41 42 43 44 45 46 47 48 49 50}"
MLP_BACKBONES="${MLP_BACKBONES:-siglip2}"
JOBS="${JOBS:-2}"
PRED="project/artifacts/predictions"

# train BACKBONE SRC_TOKEN DST <extra train_multitarget args...>
train() {
  local bb="$1" src="$2" dst="$3"
  shift 3
  local npz="${PRED}/${bb}__cross_city__${src}_to_${dst}__tier1__full__seed${SEED}.npz"
  # Skip only if the npz exists AND already carries the crosswalk target. Pre-Phase-0
  # bundles (seeds 41-43) lack osm_marked_crossing_present and MUST be regenerated,
  # not reused -- otherwise the strongest target goes missing for those seeds.
  if [ -f "$npz" ] && $PY -c "import sys,numpy as np; sys.exit(0 if 'logit__osm_marked_crossing_present' in np.load(sys.argv[1]).files else 1)" "$npz" 2>/dev/null; then
    echo "skip (done): $(basename "$npz")"
    return 0
  fi
  echo ">>> [$(date +%H:%M:%S)] ${bb}  ${src}->${dst}  seed${SEED}"
  if ! $PY project/scripts/train_multitarget.py --backbone "$bb" "$@" \
        --test-city "$dst" --target-tier 1 --ablation full --seed "$SEED" --save-predictions; then
    echo "!!! FAILED: ${bb} ${src}->${dst} seed${SEED}" >&2
  fi
}

# One leave-one-city-out direction: all backbones x seeds, sequential within the stream.
run_loco() {
  local tr="$1" te="$2"
  for SEED in $SEEDS; do
    for bb in $MLP_BACKBONES siglip_lora "siglip_lora_loco_v2_${te}"; do
      train "$bb" "$tr" "$te" --train-city "$tr"
    done
  done
}

# Pittsburgh zero-shot: train on MSP+Seattle+DC (in-corpus LoRA N/A -- no PGH embedding).
# --spatial-features pedgraph is required: the default euclidean frame
# (labels/spatial_context_features.csv) is SAM2/caption-derived and has NO Pittsburgh rows,
# so the default would raise "Missing spatial features for city=pittsburgh". pedgraph_features.csv
# covers all 4 cities and matches the prior src_pgh Pittsburgh protocol (LOCO stays euclidean;
# the two blocks are reported separately).
run_pgh() {
  for SEED in $SEEDS; do
    for bb in $MLP_BACKBONES siglip_lora_loco_v2_pittsburgh; do
      train "$bb" "dc+msp+seattle" "pittsburgh" --train-cities msp seattle dc --spatial-features pedgraph
    done
  done
}

# Launch direction-streams, at most JOBS running concurrently.
gate() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; }

for pair in msp:seattle msp:dc seattle:msp seattle:dc dc:msp dc:seattle; do
  gate
  run_loco "${pair%:*}" "${pair#*:}" &
done
gate
run_pgh &
wait

echo "================================================================"
echo "GPU arm complete. npz present: $(ls "$PRED" | grep -cE '^(siglip2|siglip_lora|siglip_lora_loco_v2_)[^_].*__cross_city__.*tier1__full__seed')"
