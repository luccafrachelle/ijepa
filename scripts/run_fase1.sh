#!/usr/bin/env bash
# Fase 1 (PLAN §8): calibration against SPHERE-JEPA table 7 with the OFFICIAL code, unmodified
# except paths. Usage: scripts/run_fase1.sh <config-name> [extra run_pretrained args, e.g. --epochs 100]
# Output: runs/calib/<config>_seed0/ (evaluation.yaml, checkpoints/last.ckpt). Skips if evaluation.yaml exists.
set -euo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)
CFG=$1; shift
OUT=$P/runs/calib/${CFG}_seed0
if [ -f "$OUT/evaluation.yaml" ]; then echo "$(date -Is) SKIP $CFG (evaluation.yaml exists)"; exit 0; fi
export HF_HOME=$P/data/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$P/third_party/SPHEREJEPA"
RESUME=()
[ -f "$OUT/checkpoints/last.ckpt" ] && RESUME=(--checkpoint "$OUT/checkpoints/last.ckpt")
echo "$(date -Is) START $CFG $* ${RESUME[*]}"
"$P/.venv/bin/python" "$P/scripts/official_pretrained.py" --config "configs/$CFG.yaml" --root "$P/data/imagenet100" \
  --output "$P/runs/calib" "${RESUME[@]}" "$@"
echo "$(date -Is) DONE $CFG"
