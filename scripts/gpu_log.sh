#!/usr/bin/env bash
# GPU telemetry every 60 s, appended to logs/gpu.log (PLAN §7). Run in its own tmux window:
#   tmux new-window -n gpulog 'scripts/gpu_log.sh'
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
exec nvidia-smi \
  --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 60 >> logs/gpu.log
