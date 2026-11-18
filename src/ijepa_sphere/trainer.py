"""I-JEPA (+ optional SUSReg/SIGReg) pretraining with per-epoch logging, collapse
metrics, quick kNN, resumable checkpoints and final evaluation (PLAN §5–§8).

Entry point: scripts/train.py (calls `main`). Run from the repo root.

Data splits (`data.eval_split`, PLAN §6):
  interno : pretrain on train minus val_interno; quick kNN and final eval
            (kNN + linear probe) with bank = train minus val_interno, queries = val_interno.
            Used for hyperparameter selection (lr / lambda sweeps).
  oficial : pretrain on the FULL train split; final eval with bank = full train,
            queries = official val (5000). Used for the final grid numbers.
            The periodic quick kNN still uses (train minus val_interno -> val_interno);
            in this mode val_interno images are seen in pretraining, so that curve is a
            monitoring signal only (it also selects best.pt).
Collapse metrics: fixed 10k subset of the full train split (`collapse_subset_ids`,
seed 0), eval transform, same images in both modes and in all cells.

Run folder: --out, else runs/[smoke_]<config>_<eval_split>_seed<k>_<hash8>, hash of the
resolved config (excluding out/name/device/workers). On skip (evaluation.json exists) or
resume (last.pt exists) the stored config must match the requested one on IDENTITY_KEYS
(seed, eval_split, source, epochs, lr, batch_size, lambda, ema, reg, smoke), else RuntimeError.

evaluation.json `status`: ok | collapsed (C2 stopping rule fired; full eval still run) |
diverged (non-finite epoch-mean loss/grad norm; no eval, terminal so the queue skips it).

Official ijepa train.py details reproduced: schedulers stepped at the start of every
iteration; AdamW with the official wd/no-wd groups; EMA momentum generator;
no gradient clipping (the official loop has none; we only log the total grad norm).
Deviation: the official loop wraps bf16 in a GradScaler; bf16 does not need loss
scaling, so no scaler is used here.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from ijepa_sphere.data import (TransformDataset, collapse_subset_ids, eval_transform,
                               ijepa_train_transform, load_imagenet100, split_train)
from ijepa_sphere.evaluation import evaluate
from ijepa_sphere.jepa import IJEPA
from ijepa_sphere.masks import make_mask_collator
from ijepa_sphere.metrics import TwoCropDataset, collapse_report
from ijepa_sphere.schedulers import build_schedules, momentum_at

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_FIELDS = ["epoch", "step", "loss", "loss_pred", "loss_reg", "lr", "wd", "momentum",
                 "grad_norm", "img_per_s", "epoch_time", "peak_mem_gb"]
COLLAPSE_FIELDS = ["epoch", "step", "n", "std_norm", "erank_pooled", "erank_tokens",
                   "uniformity", "alignment"]
KNN_FIELDS = ["epoch", "step", "knn_top1", "n_train", "n_eval"]


# ------------------------------------------------------------------ config
def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    """YAML with optional `base: <file>` (relative to this file), merged recursively."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def _parse_value(s):
    return yaml.safe_load(s)


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lambda", dest="lambda_reg", type=float)
    p.add_argument("--ema", choices=["on", "off"])
    p.add_argument("--reg", choices=["none", "sigreg", "susreg"])
    p.add_argument("--out", help="run dir (default runs/<config>_seed<k>)")
    p.add_argument("--eval-split", choices=["interno", "oficial"])
    p.add_argument("--workers", type=int)
    p.add_argument("--max-steps", type=int, help="benchmark: N steps, print JSON, exit")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch of <=smoke_steps steps, small collapse subset, official smoke eval")
    p.add_argument("--stop-after-epoch", type=int,
                   help="stop (as if interrupted) after this many epochs; resume later")
    p.add_argument("--set", action="append", default=[], metavar="KEY.PATH=VALUE",
                   help="generic config override (YAML value)")
    return p.parse_args(argv)


def resolve_config(args):
    cfg = load_config(args.config)
    for flag, path in [("seed", "seed"), ("epochs", "optimization.epochs"),
                       ("lr", "optimization.lr"), ("lambda_reg", "model.lambda_reg"),
                       ("reg", "model.reg"), ("eval_split", "data.eval_split"),
                       ("workers", "data.num_workers")]:
        v = getattr(args, flag)
        if v is not None:
            set_by_path(cfg, path, v)
    if args.ema is not None:
        set_by_path(cfg, "model.ema", args.ema == "on")
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_by_path(cfg, k, _parse_value(v))
    name = cfg.get("name") or Path(args.config).stem
    cfg["name"] = name
    if args.smoke:
        cfg["smoke"] = True
        if args.epochs is None:
            cfg["optimization"]["epochs"] = 1
    if args.out:
        cfg["out"] = args.out
    elif not cfg.get("out"):
        cfg["out"] = default_out(cfg)
    return cfg


# Keys that do not change what is trained/evaluated: excluded from the run hash.
_HASH_EXCLUDE = {("out",), ("name",), ("device",), ("data", "num_workers"),
                 ("data", "prefetch_factor"), ("data", "val_interno_sha256")}
# Keys that must match between the resolved config and a stored run on skip/resume.
IDENTITY_KEYS = [("seed",), ("data", "eval_split"), ("data", "source"),
                 ("optimization", "epochs"), ("optimization", "lr"),
                 ("optimization", "batch_size"), ("model", "lambda_reg"),
                 ("model", "ema"), ("model", "reg"), ("smoke",)]


def _strip(cfg, prefix=()):
    out = {}
    for k, v in cfg.items():
        key = prefix + (k,)
        if key in _HASH_EXCLUDE:
            continue
        out[k] = _strip(v, key) if isinstance(v, dict) else v
    return out


def config_hash(cfg, n=8):
    blob = json.dumps(_strip(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def default_out(cfg):
    """runs/[smoke_]<name>_<eval_split>_seed<k>_<hash8 of the resolved config>."""
    split = cfg.get("data", {}).get("eval_split", "interno")
    return (f"runs/{'smoke_' if cfg.get('smoke') else ''}{cfg['name']}_{split}"
            f"_seed{cfg['seed']}_{config_hash(cfg)}")


def _get(cfg, key):
    for k in key:
        if not isinstance(cfg, dict) or k not in cfg:
            return None
        cfg = cfg[k]
    return cfg


def check_same_run(resolved, stored, where):
    """Fail loudly if a stored run (evaluation.json / last.pt) was made with a
    different identity config than the one requested now."""
    diffs = []
    for key in IDENTITY_KEYS:
        a, b = _get(resolved, key), _get(stored, key)
        if key == ("smoke",):
            a, b = bool(a), bool(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            same = (math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
                    or (math.isnan(float(a)) and math.isnan(float(b))))
        else:
            same = a == b
        if not same:
            diffs.append(f"{'.'.join(key)}: stored={b!r} requested={a!r}")
    if diffs:
        raise RuntimeError(f"config mismatch with existing run in {where}:\n  " + "\n  ".join(diffs)
                           + "\nUse a different --out (or the default run folder).")


# ------------------------------------------------------------------ utils
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def atomic_torch_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv(path, fields, row):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init_fn(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


class EpochSampler(Sampler):
    """Random permutation seeded by (seed, epoch): data order is reproducible and resumable."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g).tolist())


class FakeImages(Dataset):
    """Deterministic random RGB images for tests (source: fake)."""

    def __init__(self, n, num_classes=10, size=64, offset=0):
        self.n, self.nc, self.size, self.offset = n, num_classes, size, offset

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from PIL import Image
        rng = np.random.default_rng(self.offset + int(i))
        arr = rng.integers(0, 256, (self.size, self.size, 3), dtype=np.uint8)
        return Image.fromarray(arr), int(i) % self.nc


# ------------------------------------------------------------------ data
def build_datasets(cfg):
    """-> dict of raw (PIL, label) datasets: train_full, train_part, val_interno, val_oficial."""
    d = cfg["data"]
    if d.get("source") == "fake":
        n = d.get("fake_n", 30)
        full = FakeImages(n, d.get("num_classes", 10))
        val_ids = list(range(0, n, 5))
        part_ids = [i for i in range(n) if i % 5]
        return {"train_full": full, "train_part": Subset(full, part_ids),
                "val_interno": Subset(full, val_ids),
                "val_oficial": FakeImages(d.get("fake_val_n", 10), d.get("num_classes", 10),
                                          offset=10_000)}
    raw = load_imagenet100(d.get("root", "data/imagenet100"), d.get("source", "hf"))
    part, vint = split_train(raw["train"], d.get("val_ids", "data/val_interno.txt"))
    return {"train_full": raw["train"], "train_part": part, "val_interno": vint,
            "val_oficial": raw["val"]}


def make_eval_loader(ds, cfg, shuffle=False):
    nw = cfg["data"].get("num_workers", 12)
    return DataLoader(ds, batch_size=cfg["eval"].get("batch_size", 256), shuffle=shuffle,
                      num_workers=nw, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init_fn)


# ------------------------------------------------------------------ checkpoint helpers
def reg_generator(model):
    reg = getattr(model, "regularizer", None)
    return None if reg is None else reg.stat


def capture_state(model, opt, coll, extra):
    st = reg_generator(model)
    return {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "random": random.getstate()},
        "reg_generator": None if st is None else
        {"device": str(st._generator.device), "state": st._generator.get_state()},
        "mask_counter": coll._itr_counter.value,
        **extra,
    }


def restore_state(ck, model, opt, coll):
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["rng"]["torch"])
    if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    np.random.set_state(ck["rng"]["numpy"])
    random.setstate(ck["rng"]["random"])
    rg = ck.get("reg_generator")
    st = reg_generator(model)
    if rg is not None and st is not None:
        g = torch.Generator(device=rg["device"])
        g.set_state(rg["state"])
        st._generator = g
    # Official resume steps the mask collator start_epoch*ipe times from -1.
    coll._itr_counter.value = ck["global_step"] - 1


# ------------------------------------------------------------------ main
def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    smoke = bool(cfg.get("smoke"))
    out = Path(cfg["out"])
    bench = args.max_steps is not None
    if not bench:
        if (out / "evaluation.json").exists():
            with open(out / "evaluation.json") as f:
                check_same_run(cfg, json.load(f).get("config", {}), out / "evaluation.json")
            print(f"[train] {out}/evaluation.json exists, skipping", flush=True)
            return 0
        out.mkdir(parents=True, exist_ok=True)

    d, mc, oc, lc, ec = cfg["data"], cfg["model"], cfg["optimization"], cfg["logging"], cfg["eval"]
    if d.get("source") != "fake":
        cfg["data"]["val_interno_sha256"] = sha256_file(d.get("val_ids", "data/val_interno.txt"))

    seed = int(cfg["seed"])
    seed_everything(seed)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = bool(oc.get("bf16", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- data
    dsets = build_datasets(cfg)
    oficial = d.get("eval_split", "interno") == "oficial"
    pre_ds = dsets["train_full"] if oficial else dsets["train_part"]
    crop = mc.get("crop_size", 224)
    train_ds = TransformDataset(pre_ds, ijepa_train_transform(crop))
    coll = make_mask_collator(crop)
    sampler = EpochSampler(len(train_ds), seed)
    nw = d.get("num_workers", 12)
    loader = DataLoader(train_ds, batch_size=oc.get("batch_size", 256), sampler=sampler,
                        collate_fn=coll, num_workers=nw, pin_memory=device.type == "cuda",
                        persistent_workers=nw > 0, drop_last=True, worker_init_fn=worker_init_fn,
                        prefetch_factor=d.get("prefetch_factor", 4) if nw > 0 else None,
                        generator=torch.Generator().manual_seed(seed))
    ipe = len(loader)
    cap = oc.get("steps_per_epoch") or (lc.get("smoke_steps", 20) if smoke else None)
    if cap:
        ipe = min(ipe, int(cap))
    epochs = int(oc["epochs"])

    et = eval_transform(crop)
    sub_n = lc.get("collapse_subset", 10000) if not smoke else lc.get("smoke_collapse_subset", 512)
    sub_ids = collapse_subset_ids(len(dsets["train_full"]), n=sub_n, seed=0).tolist()
    collapse_ds = TransformDataset(Subset(dsets["train_full"], sub_ids), et)
    pair_ds = TwoCropDataset(Subset(dsets["train_full"], sub_ids[:lc.get("alignment_n", 2000)]),
                             ijepa_train_transform(crop), seed=0)
    knn_bank = TransformDataset(dsets["train_part"], et)
    knn_query = TransformDataset(dsets["val_interno"], et)
    eval_cfg = {**ec, "num_workers": nw, "seed": seed,
                "num_classes": d.get("num_classes", 100)}

    # ---- model / optim
    model = IJEPA(ema=mc["ema"], reg=mc["reg"], lambda_reg=mc.get("lambda_reg", 0.05),
                  patch_size=mc.get("patch_size", 16), crop_size=crop,
                  pred_depth=mc.get("pred_depth", 6), pred_emb_dim=mc.get("pred_emb_dim", 192),
                  proj_hidden_dim=mc.get("proj_hidden_dim", 2048), proj_dim=mc.get("proj_dim", 256),
                  reg_kwargs=mc.get("reg_kwargs")).to(device)
    opt = torch.optim.AdamW(model.param_groups())
    sched, wd_sched, total_steps = build_schedules(opt, ipe, oc)
    ema_rng = tuple(oc.get("ema", [0.996, 1.0]))
    ipe_scale = oc.get("ipe_scale", 1.0)
    params = [p for g in opt.param_groups for p in g["params"]]

    state = {"epoch": 0, "global_step": 0, "best_knn": -1.0, "best_epoch": None,
             "rank_low": 0, "rank_low_first": None, "collapse_epoch": None,
             "stopped_early": False, "train_seconds": 0.0, "eval_seconds": 0.0,
             "img_per_s": [], "peak_mem_gb": 0.0, "last_collapse": None}
    last_path = out / "last.pt"
    if not bench and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        check_same_run(cfg, ck.get("config", {}), last_path)
        restore_state(ck, model, opt, coll)
        state.update(ck["state"])
        for _ in range(state["global_step"]):  # official resume: replay schedules
            sched.step()
            wd_sched.step()
        print(f"[train] resumed from {last_path} at epoch {state['epoch']} "
              f"step {state['global_step']}", flush=True)

    writer = None
    if not bench:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb"))
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def run_collapse(epoch):
        t0 = time.time()
        r = collapse_report(model, make_eval_loader(collapse_ds, cfg),
                            make_eval_loader(pair_ds, cfg), device=device,
                            max_tokens_per_img=lc.get("tokens_per_img", 4), seed=0)
        model.train()
        state["eval_seconds"] += time.time() - t0
        return {"epoch": epoch, "step": state["global_step"], **r}

    def build_result(status, final_eval_s=0.0, **extra):
        commit, dirty = git_info()
        return {
            "status": status,  # ok | collapsed (C2 stopping rule) | diverged (non-finite loss)
            "run": cfg["name"], "out": str(out), "smoke": smoke,
            "config": cfg, "git_commit": commit, "git_dirty": dirty, "seed": seed,
            "eval_split": "oficial" if oficial else "interno",
            **extra,
            "collapse_epoch": state["collapse_epoch"], "stopped_early": state["stopped_early"],
            "epochs_completed": state["epoch"], "global_step": state["global_step"],
            "best_quick_knn": {"epoch": state["best_epoch"], "knn_top1": state["best_knn"]},
            "train_hours": state["train_seconds"] / 3600,
            "eval_hours": (state["eval_seconds"] + final_eval_s) / 3600,
            "gpu_hours": (state["train_seconds"] + state["eval_seconds"] + final_eval_s) / 3600,
            "throughput_img_per_s": float(np.mean(state["img_per_s"])) if state["img_per_s"] else None,
            "peak_mem_gb": state["peak_mem_gb"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ---- training
    model.train()
    stop = state["stopped_early"]
    bench_t0, bench_n = time.time(), 0
    step_in_run = 0
    while not stop and state["epoch"] < epochs:
        epoch = state["epoch"]
        sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        acc = torch.zeros(4, device=device)  # loss, loss_pred, loss_reg, grad_norm
        n_steps = 0
        t_ep = time.time()
        lr = wd = m = float("nan")
        for itr, ((imgs, _), masks_enc, masks_pred) in enumerate(loader):
            if itr >= ipe:
                break
            lr = sched.step()
            wd = wd_sched.step()
            imgs = imgs.to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outd = model(imgs, masks_enc, masks_pred)
            opt.zero_grad(set_to_none=True)
            outd["loss"].backward()
            grads = [p.grad for p in params if p.grad is not None]
            gn = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            opt.step()
            m = momentum_at(state["global_step"], ema_rng, ipe, epochs, ipe_scale)
            model.ema_update(m)
            state["global_step"] += 1
            n_steps += 1
            acc += torch.stack([outd["loss"].detach().float(), outd["loss_pred"].detach().float(),
                                outd["loss_reg"].detach().float(), gn.float()])
            if bench:
                step_in_run += 1
                warm = min(20, max(args.max_steps - 1, 0))
                if step_in_run == warm:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    bench_t0, bench_n = time.time(), 0
                elif step_in_run > warm:
                    bench_n += imgs.shape[0]
                if step_in_run >= args.max_steps:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.time() - bench_t0
                    a = (acc / n_steps).tolist()
                    res = {"config": cfg["name"], "ema": mc["ema"], "reg": mc["reg"],
                           "batch_size": imgs.shape[0], "steps": step_in_run,
                           "warmup_steps_excluded": warm,
                           "img_per_s": bench_n / dt if dt > 0 else None,
                           "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else None,
                           "loss": a[0], "loss_pred": a[1], "loss_reg": a[2], "grad_norm": a[3]}
                    print(json.dumps(res), flush=True)
                    return res
        if bench:  # epoch boundary during a benchmark: keep going without logging
            state["epoch"] += 1
            if state["epoch"] >= epochs:
                state["epoch"] = 0
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t_ep
        state["train_seconds"] += ep_time
        a = (acc / max(n_steps, 1)).tolist()
        ips = n_steps * oc.get("batch_size", 256) / ep_time
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        state["img_per_s"].append(ips)
        state["peak_mem_gb"] = max(state["peak_mem_gb"], peak)
        state["epoch"] = epoch + 1
        ep1 = epoch + 1
        row = {"epoch": ep1, "step": state["global_step"], "loss": a[0], "loss_pred": a[1],
               "loss_reg": a[2], "lr": lr, "wd": wd, "momentum": m, "grad_norm": a[3],
               "img_per_s": ips, "epoch_time": ep_time, "peak_mem_gb": peak}
        append_csv(out / "metrics.csv", METRIC_FIELDS, row)
        for k, v in row.items():
            if k not in ("epoch", "step"):
                writer.add_scalar(f"train/{k}", v, ep1)
        print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in row.items()), flush=True)
        if not all(math.isfinite(x) for x in a[:4]):
            # Terminal state: the queue must not retry a diverged run forever.
            result = build_result("diverged", diverged_epoch=ep1, last_metrics=row)
            writer.close()
            atomic_write_json(result, out / "evaluation.json")
            print(f"[train] non-finite loss/grad at epoch {ep1}: {row}; wrote status=diverged",
                  flush=True)
            return result

        if ep1 % lc.get("collapse_every", 5) == 0 or ep1 == epochs or smoke:
            cr = run_collapse(ep1)
            state["last_collapse"] = cr
            append_csv(out / "collapse.csv", COLLAPSE_FIELDS, cr)
            for k in COLLAPSE_FIELDS[3:]:
                if k in cr:
                    writer.add_scalar(f"collapse/{k}", cr[k], ep1)
            print(f"[collapse] {cr}", flush=True)
            if cr["erank_pooled"] <= lc.get("collapse_rank_threshold", 1.5):
                state["rank_low"] += 1
                if state["rank_low"] == 1:
                    state["rank_low_first"] = ep1
            else:
                state["rank_low"], state["rank_low_first"] = 0, None
            if state["rank_low"] >= lc.get("collapse_patience", 3) and state["collapse_epoch"] is None:
                state["collapse_epoch"] = state["rank_low_first"]
                if lc.get("early_stop_collapse", False):
                    state["stopped_early"] = True
                    stop = True
                    print(f"[train] C2 stopping rule: collapsed since epoch "
                          f"{state['collapse_epoch']}, stopping at epoch {ep1}", flush=True)

        final_epoch = ep1 == epochs or stop
        if ((ep1 % lc.get("knn_every", 10) == 0 or smoke or final_epoch)
                and not (final_epoch and not oficial)):  # interno: final eval gives this kNN
            t0 = time.time()
            kr = evaluate(model, knn_bank, knn_query, eval_cfg, device=device,
                          smoke=smoke, probe=False)
            model.train()
            state["eval_seconds"] += time.time() - t0
            krow = {"epoch": ep1, "step": state["global_step"], **kr}
            append_csv(out / "knn.csv", KNN_FIELDS, krow)
            writer.add_scalar("knn/top1_val_interno", kr["knn_top1"], ep1)
            print(f"[knn] {krow}", flush=True)
            if kr["knn_top1"] > state["best_knn"]:
                state["best_knn"], state["best_epoch"] = kr["knn_top1"], ep1
                atomic_torch_save({"model": model.state_dict(), "epoch": ep1,
                                   "knn_top1": kr["knn_top1"], "config": cfg}, out / "best.pt")

        atomic_torch_save(capture_state(model, opt, coll, {"state": state, "config": cfg,
                                                           "global_step": state["global_step"]}),
                          last_path)
        writer.flush()
        if args.stop_after_epoch is not None and ep1 >= args.stop_after_epoch and ep1 < epochs:
            print(f"[train] stop-after-epoch {ep1}: exiting before final evaluation", flush=True)
            writer.close()
            return None

    # ---- final evaluation (weights at the end of training)
    cr = state["last_collapse"]
    if cr is None or cr["epoch"] != state["epoch"]:  # reuse the pass of the last epoch
        cr = run_collapse(state["epoch"])            # (run_collapse counts its own time)
    if oficial:
        bank = TransformDataset(dsets["train_full"], et)
        query = TransformDataset(dsets["val_oficial"], et)
    else:
        bank, query = knn_bank, knn_query
    t0 = time.time()
    ev = evaluate(model, bank, query, eval_cfg, device=device, smoke=smoke, probe=True)
    final_eval_s = time.time() - t0
    if not oficial:
        # The final interno kNN IS the last-epoch quick kNN (same bank/query/weights).
        append_csv(out / "knn.csv", KNN_FIELDS, {"epoch": state["epoch"], "step": state["global_step"],
                                                 "knn_top1": ev["knn_top1"], "n_train": ev["n_train"],
                                                 "n_eval": ev["n_eval"]})
        if ev["knn_top1"] > state["best_knn"]:
            state["best_knn"], state["best_epoch"] = ev["knn_top1"], state["epoch"]
            atomic_torch_save({"model": model.state_dict(), "epoch": state["epoch"],
                               "knn_top1": ev["knn_top1"], "config": cfg}, out / "best.pt")
    result = build_result(
        "collapsed" if state["stopped_early"] else "ok", final_eval_s=final_eval_s,
        eval_sets={"bank": "train_full" if oficial else "train_minus_val_interno",
                   "query": "val_oficial" if oficial else "val_interno"},
        metrics={**ev, "protocol": {"features": "avg-pool target encoder tokens",
                                    "knn_k": ec.get("knn_k", 20),
                                    "knn_temperature": ec.get("knn_temperature", 0.07),
                                    "linear_probe_epochs": 1 if smoke else ec.get("linear_probe_epochs", 90),
                                    "linear_probe_lr_effective": ec.get("linear_probe_lr", 0.1)
                                    * ec.get("batch_size", 256) / 256}},
        collapse_final=cr)
    if writer is not None:
        writer.add_scalar("final/knn_top1", ev["knn_top1"], state["epoch"])
        writer.add_scalar("final/linear_top1", ev["linear_top1"], state["epoch"])
        writer.close()
    atomic_write_json(result, out / "evaluation.json")  # last, atomic
    print(f"[eval] status={result['status']} {json.dumps(result['metrics'])} "
          f"gpu_hours={result['gpu_hours']:.3f}", flush=True)
    return result

if __name__ == "__main__":
    main()
