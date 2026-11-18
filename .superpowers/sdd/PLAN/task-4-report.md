# Task 4 report: tooling

Status: DONE (with deviations/concerns below).

## Files
- scripts/gpu_log.sh: nvidia-smi 60 s loop -> logs/gpu.log.
- scripts/bench_throughput.py: 5 cases, refuses if GPU has compute apps, writes results/throughput.json + results/tabla_tiempos.md, applies the epoch rule. Flags: --from-json, --only, --collapse-s/--knn-s/--final-eval-s (eval overhead is an estimate: 20 s/5 ep, 40 s/10 ep, 600 s final).
- scripts/run_queue.py (NOT queue.py, see deviations): sequential queue, skip if evaluation.json, logs/queue.log, <out>/train.log, continue on failure, disk < 10 GB stop (exit 3), waits (60 s poll) while any GPU process exists, never kills. `{VAR}` placeholders, `VAR=value` header lines, `--epochs`, `--var`, `--dry-run`.
- queues/fase2_sweeps.txt (9 runs), queues/fase3_grid.txt (16 runs: seed0 of 6 cells, then seeds 1,2 of C1,C3-C6; all `--eval-split oficial`, explicit `--out runs/grid_cN_seedK`). Vars EPOCHS, LR, LAMBDA_SIGREG, LAMBDA_SUSREG at the top (set after sweeps; lambda differs per regularizer so a single base.yaml value cannot hold it).
- scripts/aggregate.py (grid | sweeps | all), scripts/figures.py (4 figs PNG+PDF), tests/test_tooling.py (11 tests).
- results/throughput.json, results/tabla_tiempos.md (generated).

## Deviations / findings
1. queue.py renamed run_queue.py: scripts/queue.py shadows stdlib `queue` for every script in scripts/ (train.py crashed inside torch: ImportError from 'queue'). Found by the first bench attempt.
2. 128 px: no `--image-size` needed (`--set model.crop_size=128` already drives masks + pos-embed). BUT the official mask sampler with min_keep=10 (masks.py IJEPA_MASK_CFG) hangs forever on an 8x8 grid: predictor blocks are 9-12 patches and the rejection loop requires > min_keep; with no acceptable_regions the retry never relaxes. Workers spin at 100% CPU. The bench patches min_keep=4 in-process (no src change) and has a 900 s timeout. If 128 px is ever used for the grid, a config key for min_keep (src change) is needed. src/ unmodified.
3. Grid hours conservative: C1 at C1 img/s, the 4 regularized cells at the C6 (EMA off + SUSReg) img/s; C4/C3/C5 not benchmarked separately.

## Throughput (bs 256, SDPA, 200 steps, 20 warmup excluded, GPU alone)
| Case | img/s | peak GB | min/epoch (126689) |
|---|---|---|---|
| 224 bf16 EMA on (C1) | 1442 | 5.60 | 1.46 |
| 224 bf16 EMA off (C2) | 1439 | 5.64 | 1.47 |
| 224 bf16 EMA off + SUSReg (C6) | 1362 | 6.41 | 1.55 |
| 224 fp32 EMA on (100 steps) | 923 | 9.39 | 2.29 |
| 128 px bf16 EMA on (min_keep=4) | 4584 | 1.70 | 0.46 |

Decision rule (15-run grid on full train incl. eval-overhead estimate): E=100 44.2 h (yes), 150 65.0 h (no), 200 85.8 h, 300 127.5 h. Recommended E = 100. Sweeps: 9 runs ~8.3 h. C2 <= 0.7 h. Calibration rows: "pendiente: requiere permiso para ejecutar third_party".

## Tests
Whole suite: 57 passed (11 new). Figures inspected visually on synthetic data.

## Launch (tmux)
    tmux new-session -d -s ijepa -n gpulog 'scripts/gpu_log.sh'
    tmux new-window -t ijepa -n sweeps 'HF_HOME=$PWD/data/hf .venv/bin/python scripts/run_queue.py queues/fase2_sweeps.txt'
    .venv/bin/python scripts/aggregate.py sweeps      # -> results/sweeps.md with the --var line to use
    tmux new-window -t ijepa -n grid 'HF_HOME=$PWD/data/hf .venv/bin/python scripts/run_queue.py queues/fase3_grid.txt --epochs 100 --var LR=<best> --var LAMBDA_SIGREG=<best> --var LAMBDA_SUSREG=<best>'
    .venv/bin/python scripts/aggregate.py grid && .venv/bin/python scripts/figures.py
Dry run: add --dry-run to the run_queue command.

## Concerns
- Eval-overhead constants are estimates; E=150 is marginal (65 h vs 60 h limit).
- Aggregated "distinguible" uses max(sd_A, sd_B) for the 2-sd criterion, or paired t over common seeds (p<0.05).
- Rerunning the queue retries failed runs (they resume from last.pt per trainer).

# Fix round 1
- run_queue.py: before skipping a finished run it now runs the trainer's own identity check (trainer.resolve_config + check_same_run: seed, eval_split, source, epochs, lr, batch size, lambda, ema, reg, smoke) of the resolved queue line against evaluation.json["config"]. Mismatch -> `CONFLICT` in logs/queue.log and in the summary (`res["conflict"]`), the run is neither skipped silently nor rerun/overwritten, the queue continues, exit code 4; `--dry-run` shows CONFLICT. Unresolvable lines also count as conflicts.
- queues/fase2_sweeps.txt replaced by queues/fase2a_lr.txt (3 runs) and queues/fase2b_lambda.txt (6 runs, `--lr {LR}`, header var LR).
- aggregate.py sweeps: works after either stage; excludes eval_split == oficial runs (listed under "Excluidas"); ignores runs without the swept value (listed under "Ignoradas").
- Tests: new test_queue_conflict_on_changed_config, test_sweeps_stage_by_stage_and_exclusions; updated queue-file test.
- Command: `PYTHONPATH=src HF_HOME=$PWD/data/hf CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest tests/test_tooling.py -q` -> `13 passed in 5.30s` (no GPU used).

Updated launch commands (replace step 2/3 above):
    tmux new-window -t ijepa -n sweeps-lr 'HF_HOME=$PWD/data/hf .venv/bin/python scripts/run_queue.py queues/fase2a_lr.txt'
    .venv/bin/python scripts/aggregate.py sweeps          # lr pick -> --var LR=...
    tmux new-window -t ijepa -n sweeps-lam 'HF_HOME=$PWD/data/hf .venv/bin/python scripts/run_queue.py queues/fase2b_lambda.txt --var LR=<best>'
    .venv/bin/python scripts/aggregate.py sweeps          # lambda picks -> --var LAMBDA_SIGREG=... --var LAMBDA_SUSREG=...
    (grid and figures commands unchanged)
