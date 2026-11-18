#!/usr/bin/env python
"""Phase-0 throughput benchmark for I-JEPA (PLAN §8 fase 0 step 5) + epoch decision rule.

    .venv/bin/python scripts/bench_throughput.py            # run the benchmark
    .venv/bin/python scripts/bench_throughput.py --from-json results/throughput.json   # redo tables only

Runs scripts/train.py --max-steps N as subprocesses (one at a time), refuses to start if any
process is using the GPU, and writes results/throughput.json + results/tabla_tiempos.md.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_INTERNO = 114020      # train minus val_interno (sweeps)
N_FULL = 126689         # full train (final grid)
EPOCH_CANDIDATES = [100, 150, 200, 300]
BUDGET_H = 60.0
# Eval-overhead estimates (not measured; override with CLI flags). Per run: collapse pass every
# 5 epochs, quick kNN every 10 epochs (~40 s, Task 3 report), final evaluation (kNN + 90-ep probe).
COLLAPSE_S, KNN_S, FINAL_EVAL_S = 20.0, 40.0, 600.0

# key -> (label, train.py args, steps)
CASES = {
    "c1_bf16_ema": ("I-JEPA 224 bf16, EMA on (C1)", ["--config", "configs/c1.yaml"], 200),
    "noema_bf16": ("I-JEPA 224 bf16, EMA off (C2)", ["--config", "configs/c1.yaml", "--ema", "off"], 200),
    "noema_susreg_bf16": ("I-JEPA 224 bf16, EMA off + SUSReg (C6)",
                          ["--config", "configs/c1.yaml", "--ema", "off", "--reg", "susreg"], 200),
    "c1_fp32_ema": ("I-JEPA 224 fp32, EMA on", ["--config", "configs/c1.yaml",
                                                 "--set", "optimization.bf16=false"], 100),
    # With the 8x8 grid (128 px) the official mask sampler hangs at min_keep=10: predictor blocks have
    # 9-12 patches and the rejection loop (> min_keep) never ends for blocks of <= 10. The bench
    # therefore patches min_keep=4 in-process (no change to src/); see WRAPPED.
    "c1_128px_bf16": ("I-JEPA 128 px bf16, EMA on (min_keep=4; official min_keep=10 hangs)",
                      ["--config", "configs/c1.yaml", "--set", "model.crop_size=128"], 200),
}
WRAPPED = {"c1_128px_bf16": {"min_keep": 4}}
TIMEOUT_S = 900
_WRAP = ("import os,sys; os.environ.setdefault('HF_HOME', %r); os.environ.setdefault('HF_DATASETS_OFFLINE','1'); "
         "sys.path.insert(0, %r); from ijepa_sphere import masks; masks.IJEPA_MASK_CFG.update(%r); "
         "from ijepa_sphere.trainer import main; main(sys.argv[1:])")
PENDING = "pendiente: requiere permiso para ejecutar third_party"


def gpu_apps():
    """PIDs of processes using the GPU ([] if none)."""
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {r.stderr.strip()}")
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def gpu_state():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


def run_case(key, extra_steps=None):
    label, args, steps = CASES[key]
    if key in WRAPPED:
        code = _WRAP % (str(ROOT / "data" / "hf"), str(ROOT / "src"), WRAPPED[key])
        cmd = [sys.executable, "-c", code, *args, "--max-steps", str(steps)]
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), *args, "--max-steps", str(steps)]
    t0 = time.time()
    before = gpu_state()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"key": key, "label": label, "error": f"timeout after {TIMEOUT_S}s", "returncode": None}
    out = {"key": key, "label": label, "cmd": ("[wrapped min_keep] " + " ".join(args) if key in WRAPPED else " ".join(cmd[1:])), "wall_s": time.time() - t0,
           "gpu_before": before, "gpu_after": gpu_state(), "returncode": p.returncode}
    js = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode == 0 and js:
        out.update(json.loads(js[-1]))
    else:
        out["error"] = (p.stderr or p.stdout)[-1500:]
    return out


def eval_overhead_h(epochs, cs=COLLAPSE_S, ks=KNN_S, fs=FINAL_EVAL_S):
    return (cs * (epochs // 5) + ks * (epochs // 10) + fs) / 3600


def run_hours(img_s, n_imgs, epochs, **kw):
    return n_imgs / img_s * epochs / 3600 + eval_overhead_h(epochs, **kw)


def decide(img_s_ema, img_s_reg, **kw):
    """Largest E in EPOCH_CANDIDATES whose 15-run grid (C1 at img_s_ema, 12 regularized runs at
    img_s_reg, full train set) fits in BUDGET_H. Returns (E or None, {E: hours})."""
    hours = {E: 3 * run_hours(img_s_ema, N_FULL, E, **kw) + 12 * run_hours(img_s_reg, N_FULL, E, **kw)
             for E in EPOCH_CANDIDATES}
    ok = [E for E in EPOCH_CANDIDATES if hours[E] <= BUDGET_H]
    return (max(ok) if ok else None), hours


def build_tables(res, **kw):
    ips = {k: v.get("img_per_s") for k, v in res.items() if v.get("img_per_s")}
    need = ["c1_bf16_ema", "noema_susreg_bf16", "noema_bf16"]
    miss = [k for k in need if k not in ips]
    if miss:
        raise RuntimeError(f"missing throughput for {miss}")
    a, c, b = ips["c1_bf16_ema"], ips["noema_susreg_bf16"], ips["noema_bf16"]
    mpe = lambda s, n: n / s / 60
    L = ["# Tabla de tiempos (fase 0, medida)", "",
         "Horas/corrida = entrenamiento (img/s medido) + estimación de overhead de evaluación "
         f"(colapso cada 5 ép. x {kw.get('cs', COLLAPSE_S):.0f} s, k-NN rápido cada 10 ép. x "
         f"{kw.get('ks', KNN_S):.0f} s, evaluación final {kw.get('fs', FINAL_EVAL_S):.0f} s; "
         "estimado, no medido). Throughput: bs 256, 200 pasos (20 de warmup excluidos), "
         "SDPA, GPU sola.", "",
         "| Corrida | img/s medido | min/época | épocas | horas/corrida | corridas | horas totales |",
         "|---|---|---|---|---|---|---|",
         f"| Calibración LeJEPA (6 vistas, 224) | {PENDING} | | 200 | | 1 | |",
         f"| Calibración SPHERE-JEPA (6 vistas, 224) | {PENDING} | | 200 | | 1 | |"]
    h = run_hours(a, N_INTERNO, 30, **kw)
    L.append(f"| Barrido lr (C1, split interno 114020) | {a:.0f} | {mpe(a, N_INTERNO):.2f} | 30 | {h:.2f} | 3 | {3*h:.1f} |")
    h2 = run_hours(c, N_INTERNO, 30, **kw)
    L.append(f"| Barrido λ (C4/C6, sin EMA + reg; medido con SUSReg) | {c:.0f} | {mpe(c, N_INTERNO):.2f} | 30 | {h2:.2f} | 6 | {6*h2:.1f} |")
    L.append(f"| **Barridos I-JEPA (lr, λ) total** | | | 30 | | 9 | **{3*h + 6*h2:.1f}** |")
    E, hours = decide(a, c, **kw)
    for E_ in EPOCH_CANDIDATES:
        hc = run_hours(c, N_FULL, E_, **kw)
        L.append(f"| Grilla I-JEPA (C1 con img/s={a:.0f}; C3–C6 con img/s={c:.0f}, conservador) | {c:.0f} | {mpe(c, N_FULL):.2f} | {E_} | {hc:.2f} (C1: {run_hours(a, N_FULL, E_, **kw):.2f}) | 15 | {hours[E_]:.1f} |")
    h3 = run_hours(b, N_FULL, 20, **kw)
    L.append(f"| Control C2 (sin EMA, sin reg; early-stop) | {b:.0f} | {mpe(b, N_FULL):.2f} | ≤ 20 | ≤ {h3:.2f} | 1 | ≤ {h3:.1f} |")
    L += ["", f"Minutos/época: split interno N={N_INTERNO}, entrenamiento completo N={N_FULL}.",
          "Las filas de grilla repiten una fila por cada E candidato. Filas de calibración: "
          f"{PENDING}.", "", "## Otras mediciones", "",
          "| Caso | img/s | memoria pico (GB) | min/época (N completo) |", "|---|---|---|---|"]
    for k, v in res.items():
        if v.get("img_per_s"):
            L.append(f"| {v['label']} | {v['img_per_s']:.0f} | {v.get('peak_mem_gb', float('nan')):.2f} | {mpe(v['img_per_s'], N_FULL):.2f} |")
        else:
            L.append(f"| {v.get('label', k)} | FALLÓ | | |")
    if "c1_128px_bf16" in ips:
        s = ips["c1_128px_bf16"]
        L.append("")
        L.append(f"128 px (C1 como proxy): grilla de 15 corridas a E=100 = {15 * run_hours(s, N_FULL, 100, **kw):.1f} h.")
    L += ["", "## Regla de decisión de épocas", "",
          f"Mayor E en {EPOCH_CANDIDATES} con la grilla de 15 corridas <= {BUDGET_H:.0f} h de GPU "
          "(incluye overhead de evaluación estimado):", "",
          "| E | horas grilla (15 corridas) | entra |", "|---|---|---|"]
    for E_ in EPOCH_CANDIDATES:
        L.append(f"| {E_} | {hours[E_]:.1f} | {'sí' if hours[E_] <= BUDGET_H else 'no'} |")
    if E:
        rec = f"**Recomendación: E = {E}** ({hours[E]:.1f} h <= {BUDGET_H:.0f} h)."
    else:
        rec = ("**Ni E = 100 entra en 60 h**: bajar a 128 px para toda la grilla (PLAN §8) y repetir "
               "la calibración a 128 px.")
    L += ["", rec, ""]
    return "\n".join(L), E, hours


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", help="skip the benchmark, rebuild tables from this json")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--collapse-s", type=float, default=COLLAPSE_S)
    ap.add_argument("--knn-s", type=float, default=KNN_S)
    ap.add_argument("--final-eval-s", type=float, default=FINAL_EVAL_S)
    ap.add_argument("--only", nargs="*", choices=list(CASES), help="subset of cases")
    args = ap.parse_args(argv)
    kw = dict(cs=args.collapse_s, ks=args.knn_s, fs=args.final_eval_s)
    rd = Path(args.results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        res = json.load(open(args.from_json))
    else:
        busy = gpu_apps()
        if busy:
            print("REFUSING to start: GPU is in use by:\n  " + "\n  ".join(busy), file=sys.stderr)
            return 2
        res = {}
        for key in (args.only or CASES):
            print(f"[bench] {CASES[key][0]} ...", flush=True)
            res[key] = run_case(key)
            r = res[key]
            print(f"[bench]   -> {r.get('img_per_s')} img/s, peak {r.get('peak_mem_gb')} GB"
                  + (f"  ERROR: {r['error'][-300:]}" if "error" in r else ""), flush=True)
            json.dump(res, open(rd / "throughput.json", "w"), indent=2)
    json.dump(res, open(rd / "throughput.json", "w"), indent=2)
    md, E, hours = build_tables(res, **kw)
    (rd / "tabla_tiempos.md").write_text(md)
    print(md)
    print(f"RECOMMENDED_E={E}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
