#!/usr/bin/env bash
# Everything after Fase 2, one GPU job at a time:
#   1. wait for the SPHERE-JEPA calibration (tmux ijepa-fase1, 200 epochs) to finish
#   2. LeJEPA calibration at 100 epochs (PLAN §8 Fase 1: >15 h/run -> one at 200, the other at 100)
#   3. Fase 3 grid (seed 0 of all cells first, then seeds 1 and 2), E=100, lr/lambdas from results/sweeps.md
# A failed calibration is logged but does not block the grid.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/data/hf PYTHONPATH=src
while tmux has-session -t ijepa-fase1 2>/dev/null; do sleep 60; done
echo "$(date -Is) ijepa-fase1 finished"
scripts/run_fase1.sh imagenet100_tinyvit_lejepa --num-workers 16 --epochs 100 2>&1 | tee -a logs/fase1_lejepa.log \
  || echo "$(date -Is) LeJEPA calibration FAILED (continuing to grid)"
.venv/bin/python scripts/run_queue.py queues/fase3_grid.txt --epochs 100 \
  --var LR=0.00025 --var LAMBDA_SIGREG=0.05 --var LAMBDA_SUSREG=0.05
.venv/bin/python scripts/aggregate.py grid || true
echo "$(date -Is) run_rest done"
