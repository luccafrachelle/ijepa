import csv
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from ijepa_sphere import schedulers as S
from ijepa_sphere.evaluation import linear_probe, weighted_knn
from ijepa_sphere.metrics import alignment, effective_rank, mean_std_normalized, uniformity
from ijepa_sphere import trainer as T
from ijepa_sphere.trainer import load_config, main, parse_args, resolve_config

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- config
def test_config_merge(tmp_path):
    (tmp_path / "b.yaml").write_text("a: 1\nm: {x: 1, y: 2}\nopt: {lr: 1.0e-3}\n")
    (tmp_path / "c.yaml").write_text("base: b.yaml\nm: {y: 3}\n")
    (tmp_path / "d.yaml").write_text("base: c.yaml\nopt: {lr: 2.0e-3}\n")
    cfg = load_config(tmp_path / "d.yaml")
    assert cfg == {"a": 1, "m": {"x": 1, "y": 3}, "opt": {"lr": 2e-3}}


@pytest.mark.parametrize("name,ema,reg,early", [
    ("c1", True, "none", False), ("c2", False, "none", True), ("c3", True, "sigreg", False),
    ("c4", False, "sigreg", False), ("c5", True, "susreg", False), ("c6", False, "susreg", False)])
def test_cell_configs(name, ema, reg, early):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert (cfg["model"]["ema"], cfg["model"]["reg"]) == (ema, reg)
    assert cfg["logging"]["early_stop_collapse"] is early
    o = cfg["optimization"]
    assert o["epochs"] == 100 and o["batch_size"] == 256 and o["lr"] == 5e-4
    assert cfg["model"]["lambda_reg"] == 0.05 and cfg["data"]["num_workers"] == 12


def test_sweep_configs():
    for lr in ("2.5e-4", "5e-4", "1e-3"):
        cfg = load_config(ROOT / "configs" / f"sweep_lr_{lr}.yaml")
        assert cfg["optimization"]["lr"] == float(lr) and isinstance(cfg["optimization"]["lr"], float)
        assert cfg["optimization"]["epochs"] == 30 and cfg["data"]["eval_split"] == "interno"
        assert cfg["model"]["reg"] == "none" and cfg["model"]["ema"] is True
    for c, reg in (("c4", "sigreg"), ("c6", "susreg")):
        for lam in ("0.01", "0.05", "0.2"):
            cfg = load_config(ROOT / "configs" / f"sweep_lambda_{c}_{lam}.yaml")
            assert cfg["model"]["lambda_reg"] == float(lam) and cfg["model"]["reg"] == reg
            assert cfg["model"]["ema"] is False and cfg["optimization"]["epochs"] == 30


# ---------------------------------------------------------------- schedules
def test_schedule_endpoints():
    p = nn.Parameter(torch.zeros(2, 2))
    b = nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([{"params": [p]}, {"params": [b], "WD_exclude": True, "weight_decay": 0}])
    ipe, epochs, lr = 10, 20, 1e-3
    cfg = dict(epochs=epochs, lr=lr, start_lr_ratio=0.2, final_lr=1e-6, warmup_frac=0.1,
               weight_decay=0.04, final_weight_decay=0.4)
    sched, wd, total = S.build_schedules(opt, ipe, cfg)
    assert total == 200 and sched.warmup_steps == 20
    lrs, wds = [], []
    for _ in range(total):
        lrs.append(sched.step())
        wds.append(wd.step())
    assert lrs[0] == pytest.approx(0.2 * lr + (lr - 0.2 * lr) / 20)
    assert max(lrs) == pytest.approx(lr) and lrs[19] == pytest.approx(lr)
    assert lrs[-1] == pytest.approx(1e-6)
    assert all(a <= b + 1e-15 for a, b in zip(lrs[:19], lrs[1:20]))
    assert wds[0] == pytest.approx(0.04, abs=1e-4) and wds[-1] == pytest.approx(0.4)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.4)
    assert opt.param_groups[1]["weight_decay"] == 0
    assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"] == pytest.approx(1e-6)


def test_momentum_schedule_official():
    ema, ipe, epochs = (0.996, 1.0), 7, 5
    vals = list(S.momentum_schedule(ema, ipe, epochs, 1.0))
    assert len(vals) == ipe * epochs + 1
    assert vals[0] == 0.996 and vals[-1] == pytest.approx(1.0)
    assert all(S.momentum_at(i, ema, ipe, epochs) == pytest.approx(v) for i, v in enumerate(vals))
    # last value actually used in training (step ipe*epochs-1) is just below 1.0
    assert S.momentum_at(ipe * epochs - 1, ema, ipe, epochs) == pytest.approx(1.0 - 0.004 / 35)


# ---------------------------------------------------------------- metrics / eval
def test_collapse_metrics():
    torch.manual_seed(0)
    v = torch.randn(1, 64)
    collapsed = v.repeat(2000, 1) + 1e-6 * torch.randn(2000, 64)
    iso = torch.randn(2000, 64)
    assert effective_rank(collapsed) == pytest.approx(1.0, abs=0.01)
    assert mean_std_normalized(collapsed) < 1e-5
    assert effective_rank(iso) > 55
    assert mean_std_normalized(iso) == pytest.approx(1 / math.sqrt(64), rel=0.05)
    assert uniformity(collapsed) == pytest.approx(0.0, abs=1e-6)
    assert uniformity(iso) < -3.5
    assert alignment(iso, iso) == pytest.approx(0.0, abs=1e-8)
    assert alignment(iso, -iso) == pytest.approx(4.0)


def test_knn_and_probe_separable():
    torch.manual_seed(0)
    centers = torch.randn(5, 16) * 5
    y = torch.arange(500) % 5
    x = centers[y] + 0.1 * torch.randn(500, 16)
    assert weighted_knn(x[:400], y[:400], x[400:], y[400:], 5) == 1.0
    assert linear_probe(x[:400], y[:400], x[400:], y[400:], 5, "cpu", epochs=20, batch_size=64) == 1.0


# ---------------------------------------------------------------- training smoke on fake data
FAKE = ["--set", "data.source=fake", "--set", "data.fake_n=30", "--set", "data.num_classes=10",
        "--set", "data.num_workers=0", "--set", "optimization.batch_size=8",
        "--set", "eval.batch_size=8", "--set", "eval.linear_probe_epochs=2",
        "--set", "logging.collapse_every=1", "--set", "logging.knn_every=1",
        "--set", "device=cpu"]


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("ema", ["on", "off"])
@pytest.mark.parametrize("reg", ["none", "susreg"])
def test_train_smoke_and_resume(tmp_path, ema, reg):
    common = ["--config", str(ROOT / "configs/c1.yaml"), "--ema", ema, "--reg", reg,
              "--epochs", "2", *FAKE]
    a, b = tmp_path / "a", tmp_path / "b"
    # uninterrupted 2-epoch run (24 train imgs / batch 8 -> 3 steps per epoch)
    res = main(common + ["--out", str(a)])
    assert (a / "evaluation.json").exists() and (a / "last.pt").exists() and (a / "best.pt").exists()
    rows = _rows(a / "metrics.csv")
    assert [r["epoch"] for r in rows] == ["1", "2"] and rows[-1]["step"] == "6"
    assert all(math.isfinite(float(r["loss"])) for r in rows)
    assert (float(rows[0]["loss_reg"]) > 0) == (reg != "none")
    assert float(rows[0]["momentum"]) == pytest.approx(0.996 + 2 * 0.004 / 6)
    assert len(_rows(a / "collapse.csv")) == 2 and len(_rows(a / "knn.csv")) == 2
    ev = json.loads((a / "evaluation.json").read_text())
    assert ev["config"]["model"]["ema"] == (ema == "on") and ev["config"]["model"]["reg"] == reg
    assert 0 <= ev["metrics"]["knn_top1"] <= 1 and 0 <= ev["metrics"]["linear_top1"] <= 1
    assert ev["metrics"]["n_train"] == 24 and ev["metrics"]["n_eval"] == 6
    assert set(ev["collapse_final"]) >= {"std_norm", "erank_pooled", "erank_tokens",
                                         "uniformity", "alignment"}
    assert ev["epochs_completed"] == 2 and ev["gpu_hours"] > 0 and ev["status"] == "ok"
    assert res["metrics"] == ev["metrics"]

    # interrupted after epoch 1, then resumed automatically -> identical result
    assert main(common + ["--out", str(b), "--stop-after-epoch", "1"]) is None
    assert not (b / "evaluation.json").exists()
    ck = torch.load(b / "last.pt", weights_only=False)
    assert ck["global_step"] == 3 and ck["state"]["epoch"] == 1
    assert (ck["reg_generator"] is not None) == (reg != "none")
    main(common + ["--out", str(b)])
    rb = _rows(b / "metrics.csv")
    assert [r["epoch"] for r in rb] == ["1", "2"]
    for ra_, rb_ in zip(rows, rb):
        for k in ("loss", "loss_pred", "loss_reg", "lr", "wd", "momentum"):
            assert float(ra_[k]) == pytest.approx(float(rb_[k]), rel=1e-5, abs=1e-7), k
    sa = torch.load(a / "last.pt", weights_only=False)["model"]
    sb = torch.load(b / "last.pt", weights_only=False)["model"]
    for k in sa:
        assert torch.allclose(sa[k].float(), sb[k].float(), atol=1e-6), k
    # a finished run is skipped
    assert main(common + ["--out", str(b)]) == 0


def test_max_steps_bench(capsys):
    res = main(["--config", str(ROOT / "configs/c6.yaml"), "--max-steps", "4", *FAKE])
    assert res["steps"] == 4 and res["img_per_s"] > 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["steps"] == 4


# ---------------------------------------------------------------- run identity
def _cfg(*extra):
    return resolve_config(parse_args(["--config", str(ROOT / "configs/c1.yaml"), *extra]))


def test_default_out_depends_on_split_and_overrides():
    base = _cfg("--seed", "0")["out"]
    assert base.startswith("runs/c1_interno_seed0_")
    outs = {base, _cfg("--seed", "0", "--eval-split", "oficial")["out"],
            _cfg("--seed", "0", "--lr", "1e-3")["out"], _cfg("--seed", "0", "--epochs", "30")["out"],
            _cfg("--seed", "0", "--lambda", "0.2")["out"], _cfg("--seed", "0", "--ema", "off")["out"],
            _cfg("--seed", "0", "--reg", "susreg")["out"], _cfg("--seed", "1")["out"]}
    assert len(outs) == 8
    assert "_oficial_seed0_" in _cfg("--seed", "0", "--eval-split", "oficial")["out"]
    # stable, and not affected by workers / device
    assert _cfg("--seed", "0", "--workers", "4", "--set", "device=cpu")["out"] == base


def test_mismatch_on_skip_and_resume(tmp_path):
    common = ["--config", str(ROOT / "configs/c1.yaml"), "--epochs", "2", *FAKE]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])
    for bad in (["--eval-split", "oficial"], ["--epochs", "3"], ["--lr", "1e-3"],
                ["--lambda", "0.2"], ["--ema", "off"], ["--reg", "susreg"], ["--seed", "1"]):
        with pytest.raises(RuntimeError, match="config mismatch"):
            main(common + bad + ["--out", str(a)])
    assert main(common + ["--out", str(a)]) == 0  # identical config: skipped
    main(common + ["--out", str(b), "--stop-after-epoch", "1"])
    for bad in (["--epochs", "3"], ["--eval-split", "oficial"], ["--lr", "1e-3"]):
        with pytest.raises(RuntimeError, match="config mismatch"):
            main(common + bad + ["--out", str(b)])
    assert not (b / "evaluation.json").exists()


# ---------------------------------------------------------------- terminal statuses
def test_diverged_writes_terminal_evaluation(tmp_path):
    out = tmp_path / "d"
    args = ["--config", str(ROOT / "configs/c6.yaml"), "--epochs", "2", *FAKE,
            "--set", "model.lambda_reg=.nan", "--out", str(out)]
    res = main(args)
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["status"] == "diverged" == res["status"] and ev["diverged_epoch"] == 1
    assert ev["last_metrics"]["epoch"] == 1 and "metrics" not in ev
    assert main(args) == 0  # queue / rerun skips it


def test_c2_collapse_stop_status_and_no_duplicate_passes(tmp_path, monkeypatch):
    calls = {"collapse": 0, "eval": 0}
    real_c, real_e = T.collapse_report, T.evaluate

    def cnt_c(*a, **k):
        calls["collapse"] += 1
        return real_c(*a, **k)

    def cnt_e(*a, **k):
        calls["eval"] += 1
        return real_e(*a, **k)
    monkeypatch.setattr(T, "collapse_report", cnt_c)
    monkeypatch.setattr(T, "evaluate", cnt_e)
    out = tmp_path / "c2"
    main(["--config", str(ROOT / "configs/c2.yaml"), "--epochs", "3", *FAKE,
          "--set", "logging.collapse_uniformity_threshold=-1000", "--set", "logging.collapse_std_factor=1000",
          "--set", "logging.collapse_patience=1",
          "--out", str(out)])
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["status"] == "collapsed" and ev["stopped_early"] and ev["collapse_epoch"] == 1
    assert ev["epochs_completed"] == 1 and "linear_top1" in ev["metrics"]
    # one collapse pass (reused as final) and one evaluate (final; interno quick kNN skipped)
    assert calls == {"collapse": 1, "eval": 1}
    assert len(_rows(out / "collapse.csv")) == 1 and len(_rows(out / "knn.csv")) == 1
    assert ev["collapse_final"]["epoch"] == 1
    assert ev["eval_hours"] <= ev["gpu_hours"]


# ---------------------------------------------------------------- GPU: regularizer generator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reg_generator_survives_resume_on_gpu():
    from ijepa_sphere.jepa import IJEPA
    from ijepa_sphere.masks import make_mask_collator
    from ijepa_sphere.trainer import capture_state, restore_state
    torch.manual_seed(0)
    model = IJEPA(ema=False, reg="susreg").cuda()
    opt = torch.optim.AdamW(model.param_groups())
    coll = make_mask_collator(224)
    z = torch.randn(32, 256, device="cuda")
    model.regularizer(z)  # generator moves to cuda (official forward)
    ck = capture_state(model, opt, coll, {"global_step": 5})
    assert ck["reg_generator"]["device"] == "cuda:0"
    expected = model.regularizer(z)
    model2 = IJEPA(ema=False, reg="susreg").cuda()
    opt2 = torch.optim.AdamW(model2.param_groups())
    restore_state(ck, model2, opt2, make_mask_collator(224))
    g = model2.regularizer.stat._generator
    assert g.device == z.device  # so _official_regs.py:98 does not reseed it
    assert torch.equal(model2.regularizer(z), expected)
    assert model2.regularizer.stat._generator is g


def test_is_collapsed_criterion_and_check_run_data():
    from ijepa_sphere.metrics import is_collapsed
    d = 192
    # real check_c2_5ep final point: std .0105, uniformity -.11 -> collapsed
    assert is_collapsed(-0.11, 0.0105, d)
    # first eval of that run (std .0186 > .25/sqrt(192)=.01804) and random init: not collapsed
    assert not is_collapsed(-0.33, 0.0186, d) and not is_collapsed(-0.79, 0.035, d)
    assert not is_collapsed(-0.11, 0.05, d) and not is_collapsed(-0.5, 0.001, d)
    assert not is_collapsed(None, 0.001, d)
    cfg = load_config(ROOT / "configs" / "base.yaml")["logging"]
    assert (cfg["collapse_uniformity_threshold"], cfg["collapse_std_factor"], cfg["collapse_patience"]) == (-0.2, 0.25, 3)
    assert "collapse_rank_threshold" not in cfg


def test_trainer_passes_seed_to_regularizer(tmp_path, monkeypatch):
    import ijepa_sphere.trainer as T
    seen = []
    real = T.IJEPA

    def spy(*a, **k):
        seen.append(k.get("reg_kwargs"))
        return real(*a, **k)
    monkeypatch.setattr(T, "IJEPA", spy)
    for sd in (0, 1):
        main(["--config", str(ROOT / "configs/c6.yaml"), "--epochs", "1", "--seed", str(sd), *FAKE,
              "--out", str(tmp_path / f"s{sd}")])
    assert [k["seed"] for k in seen] == [0, 1]


def test_regularizer_seed_changes_projections():
    from ijepa_sphere.regularizers import build_regularizer
    z = torch.randn(64, 256, generator=torch.Generator().manual_seed(3))
    for kind in ("susreg", "sigreg"):
        l0, l0b, l1 = (build_regularizer(kind, dim=256, seed=s)(z) for s in (0, 0, 1))
        assert torch.allclose(l0, l0b) and not torch.allclose(l0, l1)
