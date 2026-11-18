"""Task 4 tooling: aggregate, figures, run queue, throughput tables (synthetic fixtures, no GPU)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


agg, figs, rq, bench = (_load(n) for n in ("aggregate", "figures", "run_queue", "bench_throughput"))


GRID_VARS = {"LR": "5e-4", "LAMBDA_SIGREG": "0.05", "LAMBDA_SUSREG": "0.05"}   # no defaults in the queue file


def make_run(runs, name, lin, knn, status="ok", erank=50.0, epochs=3, gpu_h=1.0, cfg=None, dup=False, split=None):
    d = runs / name
    d.mkdir(parents=True)
    ev = {"status": status, "eval_split": split or ("interno" if name.startswith("sweep") else "oficial"), "gpu_hours": gpu_h, "epochs_completed": epochs,
          "collapse_final": {"erank_pooled": erank, "uniformity": -2.0, "alignment": 0.3},
          "collapse_epoch": None, "config": cfg or {}}
    if lin is not None:
        ev["metrics"] = {"linear_top1": lin / 100, "knn_top1": knn / 100}
    (d / "evaluation.json").write_text(json.dumps(ev))
    with open(d / "collapse.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "n", "std_norm", "erank_pooled", "erank_tokens", "uniformity", "alignment"])
        for e in range(1, epochs + 1):
            w.writerow([e, e * 10, 100, 0.1, erank / e, 10, -2, 0.3])
        if dup:   # crash duplicate: last row for epoch 1 wins
            w.writerow([1, 10, 100, 0.1, 999.0, 10, -2, 0.3])
    return d


def synth_grid(runs):
    vals = {"c1": (70, 65), "c3": (68, 64), "c4": (60, 58), "c5": (71, 66), "c6": (72, 67)}
    for c, (l, k) in vals.items():
        for s, dv in zip((0, 1, 2), (-0.5, 0.0, 0.5)):
            make_run(runs, f"grid_{c}_seed{s}", l + dv, k + dv)
    make_run(runs, "grid_c2_seed0", None, None, status="collapsed", erank=1.0)


def test_read_csv_dedup_keeps_last(tmp_path):
    d = make_run(tmp_path, "r", 1, 1, dup=True)
    rows = agg.read_csv_dedup(d / "collapse.csv")
    assert [r["epoch"] for r in rows] == [1.0, 2.0, 3.0]
    assert rows[0]["erank_pooled"] == 999.0
    assert agg.read_csv_dedup(tmp_path / "nope.csv") == []


def test_aggregate_grid_files_and_stats(tmp_path):
    runs, res = tmp_path / "runs", tmp_path / "results"
    synth_grid(runs)
    agg.main(["grid", "--runs-dir", str(runs), "--results-dir", str(res)])
    rows = list(csv.DictReader(open(res / "tabla_grilla.csv")))
    by = {r["cell"]: r for r in rows}
    assert set(by) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert by["C1"]["n_seeds"] == "3" and float(by["C1"]["linear_mean"]) == pytest.approx(70.0)
    assert float(by["C1"]["linear_sd"]) == pytest.approx(0.5)      # sample sd of 69.5, 70, 70.5
    assert by["C2"]["status"] == "collapsed 1/1" and by["C2"]["linear_mean"] == ""
    md = (res / "tabla_grilla.md").read_text()
    assert "LeJEPA (publicado)" in md and "70.71 ± 0.38" in md and "| InfoNCE (publicado) | 51.93 | 56.93 |" in md
    assert "máscaras" in md                                   # footnote
    # C6 - C1 = +2 on linear with sd 0.5 -> > 2 sd; P3 C5 - C1 = +1, > 2*0.5? 1 > 1 false but paired t -> p tiny
    assert "| P1 | C6 vs C1 | lineal | +2.00" in md
    assert md.count("distinguible") >= 4


def test_compare_verdicts():
    mk = lambda xs: [{"seed": i, "linear": x} for i, x in enumerate(xs)]
    r = agg.compare(mk([70, 71, 72]), mk([60, 61, 62]), "linear")
    assert r["verdict"] == "distinguible" and r["diff"] == pytest.approx(10)
    # noisy, paired differences also noisy -> not distinguishable
    r = agg.compare(mk([70, 74, 66]), mk([71, 69, 73]), "linear")
    assert r["verdict"] == "no distinguible"
    # paired test rescues a diff within 2 sd (shared seed effect)
    r = agg.compare(mk([60, 70, 80]), mk([59, 69.1, 78.9]), "linear")
    assert abs(r["diff"]) < 2 * r["sd"] and r["p"] < 0.05 and r["verdict"] == "distinguible"
    assert agg.compare(mk([70]), mk([60]), "linear")["verdict"].startswith("no evaluable")
    assert agg.compare([], mk([1, 2]), "linear")["verdict"].startswith("no evaluable")


def test_aggregate_partial_results_do_not_crash(tmp_path):
    runs, res = tmp_path / "runs", tmp_path / "results"
    make_run(runs, "grid_c1_seed0", 70, 65)
    make_run(runs, "grid_c6_seed0", 71, 66)
    make_run(runs, "grid_c6_seed1", 72, 67)
    agg.main(["grid", "--runs-dir", str(runs), "--results-dir", str(res)])
    md = (res / "tabla_grilla.md").read_text()
    assert "C1" in md and "(n=1)" in md and "Celdas sin resultados" in md
    agg.main(["grid", "--runs-dir", str(tmp_path / "empty"), "--results-dir", str(tmp_path / "r2")])
    assert (tmp_path / "r2" / "tabla_grilla.csv").exists()


def test_sweeps_pick_best(tmp_path):
    runs, res = tmp_path / "runs", tmp_path / "results"
    for lr, k in [(2.5e-4, 40), (5e-4, 45), (1e-3, 42)]:
        make_run(runs, f"sweep_lr{lr:g}_seed0", k, k, cfg={"optimization": {"lr": lr}})
    for c, vals in {"c4": [(0.01, 30), (0.05, 35), (0.2, 33)], "c6": [(0.01, 31), (0.05, 30), (0.2, 36)]}.items():
        for lam, k in vals:
            make_run(runs, f"sweep_lambda_{c}_{lam}_seed0", k, k, cfg={"model": {"lambda_reg": lam}})
    make_run(runs, "sweep_lambda_c6_9_seed0", 99, 99, status="diverged", cfg={"model": {"lambda_reg": 9}})
    agg.main(["sweeps", "--runs-dir", str(runs), "--results-dir", str(res)])
    md = (res / "sweeps.md").read_text()
    assert "Elegido: lr = 0.0005" in md
    assert "Elegido: λ = 0.05 (k-NN 35.00%)" in md      # C4
    assert "Elegido: λ = 0.2 (k-NN 36.00%)" in md       # C6; diverged run (99) is ignored
    assert "--var LR=0.0005 --var LAMBDA_SIGREG=0.05 --var LAMBDA_SUSREG=0.2" in md


def test_figures_full_partial_and_empty(tmp_path):
    runs = tmp_path / "runs"
    synth_grid(runs)
    figs.main(["--runs-dir", str(runs), "--results-dir", str(tmp_path / "full")])
    names = {p.name for p in (tmp_path / "full" / "figs").iterdir()}
    for n in ("fig1_grilla_barras", "fig2_rango_efectivo", "fig3_uniformidad_alineacion", "fig4_tabla_combinada"):
        assert f"{n}.png" in names and f"{n}.pdf" in names
    # partial: one cell, one seed; and nothing at all
    make_run(tmp_path / "runs2", "grid_c1_seed0", 70, 65)
    figs.main(["--runs-dir", str(tmp_path / "runs2"), "--results-dir", str(tmp_path / "part")])
    figs.main(["--runs-dir", str(tmp_path / "none"), "--results-dir", str(tmp_path / "empty")])
    assert (tmp_path / "empty" / "figs" / "fig1_grilla_barras.png").exists()


# ------------------------------------------------------------------ queue
def test_parse_queue_vars_and_overrides():
    text = "EPOCHS=100\nLR=5e-4\n# c\nconfigs/c1.yaml --epochs {EPOCHS} --lr {LR} --out runs/a_seed0  # x\n\n"
    v, runs = rq.parse_queue(text)
    assert v == {"EPOCHS": "100", "LR": "5e-4"}
    assert runs == [["configs/c1.yaml", "--epochs", "100", "--lr", "5e-4", "--out", "runs/a_seed0"]]
    _, runs = rq.parse_queue(text, {"EPOCHS": "7"})
    assert runs[0][2] == "7"
    with pytest.raises(ValueError, match="unresolved"):
        rq.parse_queue("configs/c1.yaml --epochs {EPOCHS} --out runs/a")
    with pytest.raises(ValueError, match="missing --out"):
        rq.parse_queue("configs/c1.yaml")


def test_shipped_queue_files_resolve():
    root = SCRIPTS.parent
    _, a = rq.parse_queue((root / "queues" / "fase2a_lr.txt").read_text())
    _, b = rq.parse_queue((root / "queues" / "fase2b_lambda.txt").read_text(), {"LR": "1e-3"})
    assert len(a) == 3 and len(b) == 6
    assert all("--eval-split" not in r for r in a + b)
    assert all(r[r.index("--lr") + 1] == "1e-3" for r in b)
    _, g = rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text(), {"EPOCHS": "50", **GRID_VARS})
    outs = [rq.out_dir_of(r).name for r in g]
    assert len(g) == 16 and outs[:6] == [f"grid_c{i}_seed0" for i in range(1, 7)]
    assert "grid_c2_seed1" not in outs and "grid_c2_seed2" not in outs
    assert all(r[r.index("--eval-split") + 1] == "oficial" for r in g)
    assert all("--epochs" in r and r[r.index("--epochs") + 1] == "50" for r in g)
    assert rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text(), GRID_VARS)[0]["LAMBDA_SIGREG"]
    assert [r for r in g if "c5" in r[0] or "c6" in r[0]][0].count("--lambda") == 1


FAKE = ("import sys, json, pathlib\n"
        "a = sys.argv[1:]; out = pathlib.Path(a[a.index('--out')+1]); out.mkdir(parents=True, exist_ok=True)\n"
        "print('hello', a)\n"
        "if 'FAIL' in str(out): sys.exit(3)\n"
        "(out/'evaluation.json').write_text('{}')\n")


def test_queue_skip_fail_continue_and_logs(tmp_path):
    (tmp_path / "fake.py").write_text(FAKE)
    runs = [["cfgA", "--out", "runs/a_seed0"], ["cfgB", "--out", "runs/FAIL_seed0"],
            ["cfgC", "--out", "runs/c_seed0"]]
    (tmp_path / "runs" / "c_seed0").mkdir(parents=True)
    (tmp_path / "runs" / "c_seed0" / "evaluation.json").write_text("{}")   # already done
    kw = dict(root=tmp_path, train_cmd=[sys.executable, str(tmp_path / "fake.py")],
              gpu_busy_fn=lambda: [], free_gb_fn=lambda p: 500.0, conflict_fn=lambda a, r: None)
    res = rq.run_queue(runs, **kw)
    assert res["done"] == ["runs/a_seed0"] and res["skipped"] == ["runs/c_seed0"]
    assert res["failed"] == ["runs/FAIL_seed0"] and res["stopped"] is None
    assert "hello" in (tmp_path / "runs" / "a_seed0" / "train.log").read_text()
    log = (tmp_path / "logs" / "queue.log").read_text()
    assert "SKIP runs/c_seed0" in log and "FAILED runs/FAIL_seed0 rc=3" in log and "DONE runs/a_seed0" in log
    # rerun: a is skipped now, failed one is retried (no evaluation.json)
    res = rq.run_queue(runs, **kw)
    assert res["done"] == [] and len(res["skipped"]) == 2 and res["failed"] == ["runs/FAIL_seed0"]


def test_queue_waits_for_foreign_gpu_user_and_stops_on_low_disk(tmp_path):
    (tmp_path / "fake.py").write_text(FAKE)
    runs = [["cfgA", "--out", "runs/a_seed0"], ["cfgB", "--out", "runs/b_seed0"]]
    busy, sleeps = [["123, python"], ["123, python"], []], []
    def gpu():
        return busy.pop(0) if busy else []
    res = rq.run_queue(runs, root=tmp_path, train_cmd=[sys.executable, str(tmp_path / "fake.py")],
                       gpu_busy_fn=gpu, free_gb_fn=lambda p: 500.0, sleep=sleeps.append, poll_s=60)
    assert sleeps == [60, 60] and len(res["done"]) == 2
    (tmp_path / "runs" / "a_seed0" / "evaluation.json").unlink()
    res = rq.run_queue(runs, root=tmp_path, train_cmd=[sys.executable, str(tmp_path / "fake.py")],
                       gpu_busy_fn=lambda: [], free_gb_fn=lambda p: 5.0)
    assert res["stopped"] == "disk" and res["done"] == []
    assert "STOP: only 5.0 GB free" in (tmp_path / "logs" / "queue.log").read_text()


# ------------------------------------------------------------------ bench tables
def test_bench_decision_rule_and_tables(tmp_path):
    res = {"c1_bf16_ema": {"label": "a", "img_per_s": 1463.0, "peak_mem_gb": 5.6},
           "noema_bf16": {"label": "b", "img_per_s": 1500.0, "peak_mem_gb": 5.0},
           "noema_susreg_bf16": {"label": "c", "img_per_s": 1357.0, "peak_mem_gb": 6.4},
           "c1_128px_bf16": {"label": "d", "img_per_s": 4000.0, "peak_mem_gb": 2.0},
           "c1_fp32_ema": {"label": "e", "error": "boom"}}
    E, hours = bench.decide(1463.0, 1357.0)
    assert hours[100] < hours[150] < hours[200] and E == max(e for e in hours if hours[e] <= 60)
    E_slow, _ = bench.decide(300.0, 300.0)
    assert E_slow is None
    md, E2, _ = bench.build_tables(res)
    assert E2 == E and f"Recomendación: E = {E}" in md and "pendiente: requiere permiso" in md and "FALLÓ" in md
    # run-time sanity: 126689 imgs at 1463 img/s = 1.44 min/epoch
    assert f"{126689 / 1463 / 60:.2f}" in md


def test_sweeps_stage_by_stage_and_exclusions(tmp_path):
    runs, res = tmp_path / "runs", tmp_path / "results"
    for lr, k in [(5e-4, 45), (1e-3, 42)]:
        make_run(runs, f"sweep_lr{lr:g}_seed0", k, k, cfg={"optimization": {"lr": lr}})
    agg.main(["sweeps", "--runs-dir", str(runs), "--results-dir", str(res)])   # after stage a only
    md = (res / "sweeps.md").read_text()
    assert "Elegido: lr = 0.0005" in md and "--var LR=0.0005`" in md and "--var LAMBDA" not in md
    # official-split run (even a better one) is excluded; run without swept value is ignored
    d = make_run(runs, "sweep_lr2e-3_seed0", 99, 99, cfg={"optimization": {"lr": 2e-3}})
    ev = json.loads((d / "evaluation.json").read_text())
    ev["eval_split"] = "oficial"
    (d / "evaluation.json").write_text(json.dumps(ev))
    make_run(runs, "sweep_lambda_c4_x_seed0", 50, 50, cfg={})
    agg.main(["sweeps", "--runs-dir", str(runs), "--results-dir", str(res)])
    md = (res / "sweeps.md").read_text()
    assert "Elegido: lr = 0.0005" in md and "Excluidas" in md and "sweep_lr2e-3_seed0" in md
    assert "Ignoradas" in md and "sweep_lambda_c4_x_seed0" in md


def test_queue_conflict_on_changed_config(tmp_path):
    """A finished run whose stored config differs from the (re)launched line is CONFLICT, not skipped."""
    root = SCRIPTS.parent
    _, lines = rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text(), {"EPOCHS": "100", **GRID_VARS})
    line = lines[0]                                    # grid_c1_seed0
    trainer = sys.modules.get("ijepa_sphere.trainer") or __import__("ijepa_sphere.trainer", fromlist=["x"])
    cfg = trainer.resolve_config(trainer.parse_args(["--config", str(root / line[0])] + line[1:]))
    out = tmp_path / "runs" / "grid_c1_seed0"
    out.mkdir(parents=True)
    (out / "evaluation.json").write_text(json.dumps({"config": cfg}))
    # same line: identical -> skipped
    kw = dict(root=tmp_path, train_cmd=[sys.executable, "-c", "raise SystemExit(9)"],
              gpu_busy_fn=lambda: [], free_gb_fn=lambda p: 500.0)
    assert rq.check_conflict(line, tmp_path) is None
    res = rq.run_queue([line], **kw)
    assert res["skipped"] == ["runs/grid_c1_seed0"] and not res["conflict"]
    # relaunch with other EPOCHS / LR / lambda-irrelevant seed: conflict, never skipped, never run
    for var in ({**GRID_VARS, "EPOCHS": "150"}, {**GRID_VARS, "EPOCHS": "100", "LR": "1e-3"}):
        _, l2 = rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text(), var)
        assert "optimization.epochs" in rq.check_conflict(l2[0], tmp_path) or "optimization.lr" in rq.check_conflict(l2[0], tmp_path)
        res = rq.run_queue([l2[0]], **kw)
        assert res["conflict"] == ["runs/grid_c1_seed0"] and not res["skipped"] and not res["failed"]
    assert "CONFLICT runs/grid_c1_seed0" in (tmp_path / "logs" / "queue.log").read_text()
    assert (out / "evaluation.json").exists() and not (out / "train.log").exists()


def test_queue_without_var_fails_and_headers_have_no_defaults():
    qdir = Path(__file__).resolve().parents[1] / "queues"
    for name, needed in (("fase2b_lambda.txt", {"LR"}), ("fase3_grid.txt", {"LR", "LAMBDA_SIGREG", "LAMBDA_SUSREG"})):
        text = (qdir / name).read_text()
        v, _ = rq.parse_queue(text, {k: "1" for k in needed})   # resolves with --var
        v0 = {k for k in needed if f"\n{k}=" in "\n" + text}
        assert not v0
        with pytest.raises(ValueError, match="unresolved"):
            rq.parse_queue(text)


def test_grid_runs_validation_and_collapse_label(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "grid_c1_seed0", 70, 65, epochs=100)
    make_run(runs, "grid_c1_seed1", 71, 66, epochs=50)                      # epochs differ
    make_run(runs, "grid_c3_seed0", 60, 55, split="interno")                # excluded
    make_run(runs, "grid_c4_seed0", None, None, status="diverged")          # kept, reported
    make_run(runs, "grid_c2_seed0", 5, 5)                                   # criterion on final metrics
    ev = json.loads((runs / "grid_c2_seed0/evaluation.json").read_text())
    ev["collapse_final"].update(uniformity=-0.1, std_norm=0.01, dim=192)
    (runs / "grid_c2_seed0/evaluation.json").write_text(json.dumps(ev))
    w = []
    cells = agg.grid_runs(agg.load_runs(runs), w)
    assert "c3" not in cells and any("grid_c3_seed0" in x and "interno" in x for x in w)
    assert any("C1" in x and "épocas" in x for x in w)
    assert cells["c2"][0]["collapsed"] and not cells["c1"][0]["collapsed"]
    rows, md, _ = agg.grid_tables(cells, w)
    by = {r["cell"]: r for r in rows}
    assert "colapsa" in by["C2"]["status"] and by["C4"]["status"] == "diverged 1/1"
    assert "Advertencias" in md
