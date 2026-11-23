#!/usr/bin/env bash
# Fase 2 end to end (PLAN §8): lr sweep on C1 -> pick lr by val_interno kNN ->
# lambda sweep on C4/C6 with that lr -> pick lambdas. Results in results/sweeps.md.
# Launch: tmux new-session -d -s ijepa-fase2 'scripts/run_fase2.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/data/hf PYTHONPATH=src
PY=.venv/bin/python

$PY scripts/run_queue.py queues/fase2a_lr.txt
$PY scripts/aggregate.py sweeps
LR=$(grep -o -- '--var LR=[^ `]*' results/sweeps.md | head -1 | cut -d= -f2)
[ -n "$LR" ] || { echo "no lr pick in results/sweeps.md" >&2; exit 1; }
echo "fase2: chosen lr = $LR"

$PY scripts/run_queue.py queues/fase2b_lambda.txt --var LR="$LR"
$PY scripts/aggregate.py sweeps
echo "fase2 done; grid vars:"
grep -- '--var' results/sweeps.md | tail -1
