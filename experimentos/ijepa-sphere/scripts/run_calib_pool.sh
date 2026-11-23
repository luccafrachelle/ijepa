#!/usr/bin/env bash
# Eval-only of the two calibration checkpoints with CLS (sanity: must reproduce
# runs/calib/*/evaluation.yaml) and avg-pool. Output: runs/calib_pool/{token,avg}/<cfg>_seed0/
set -euo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=$P/data/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$P/third_party/SPHEREJEPA"
for POOL in avg token; do
  for CFG in imagenet100_tinyvit_spherejepa imagenet100_tinyvit_lejepa; do
    OUT=$P/runs/calib_pool/$POOL
    if [ -f "$OUT/${CFG}_seed0/evaluation.yaml" ]; then echo "SKIP $POOL $CFG"; continue; fi
    echo "$(date -Is) START $POOL $CFG"
    POOL=$POOL "$P/.venv/bin/python" "$P/scripts/official_eval_pool.py" --config "configs/$CFG.yaml" \
      --eval-only --checkpoint "$P/runs/calib/${CFG}_seed0/checkpoints/last.ckpt" \
      --root "$P/data/imagenet100" --output "$OUT" --num-workers 16
    echo "$(date -Is) DONE $POOL $CFG"
  done
done
