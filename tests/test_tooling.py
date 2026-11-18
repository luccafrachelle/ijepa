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


def make_run(runs, name, lin, knn, status="ok", erank=50.0, epochs=3, gpu_h=1.0, cfg=None, dup=False):
    d = runs / name
    d.mkdir(parents=True)
    ev = {"status": status, "eval_split": "oficial", "gpu_hours": gpu_h, "epochs_completed": epochs,
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
    _, sw = rq.parse_queue((root / "queues" / "fase2_sweeps.txt").read_text())
    assert len(sw) == 9 and all("--eval-split" not in r for r in sw)
    _, g = rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text(), {"EPOCHS": "50"})
    outs = [rq.out_dir_of(r).name for r in g]
    assert len(g) == 16 and outs[:6] == [f"grid_c{i}_seed0" for i in range(1, 7)]
    assert "grid_c2_seed1" not in outs and "grid_c2_seed2" not in outs
    assert all(r[r.index("--eval-split") + 1] == "oficial" for r in g)
    assert all("--epochs" in r and r[r.index("--epochs") + 1] == "50" for r in g)
    for c in ("c3", "c4"):
        assert rq.parse_queue((root / "queues" / "fase3_grid.txt").read_text())[0]["LAMBDA_SIGREG"]
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
              gpu_busy_fn=lambda: [], free_gb_fn=lambda p: 500.0)
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
