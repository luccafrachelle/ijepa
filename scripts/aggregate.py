#!/usr/bin/env python
"""Aggregate runs/*/evaluation.json (PLAN §7 Salida, §6 statistics).

    .venv/bin/python scripts/aggregate.py grid      # -> results/tabla_grilla.{csv,md}
    .venv/bin/python scripts/aggregate.py sweeps    # -> results/sweeps.md
    .venv/bin/python scripts/aggregate.py all

Grid runs are runs/grid_c<n>_seed<k>; sweep runs are runs/sweep_*. Accuracies are reported in %.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["c1", "c2", "c3", "c4", "c5", "c6"]
CELL_DESC = {"c1": "EMA, sin reg (I-JEPA)", "c2": "sin EMA, sin reg (control)",
             "c3": "EMA + SIGReg", "c4": "sin EMA + SIGReg",
             "c5": "EMA + SUSReg", "c6": "sin EMA + SUSReg"}
COMPARISONS = [("P1", "c6", "c1"), ("P2", "c6", "c4"), ("P2", "c5", "c3"), ("P3", "c5", "c1")]
# Published SPHERE-JEPA Table 7 (ViT-Tiny, ImageNet-100, 200 ep): (linear mean, sd, kNN mean, sd)
TABLE7 = [("LeJEPA (publicado)", 70.71, 0.38, 67.39, 0.61),
          ("SPHERE-JEPA (publicado)", 71.91, 0.90, 69.03, 0.43),
          ("InfoNCE (publicado)", 51.93, None, 56.93, None),
          ("VICReg (publicado)", 57.74, None, 56.01, None)]
FOOTNOTE = ("Nota: las filas publicadas (tabla 7 de SPHERE-JEPA) usan 200 épocas y 6 vistas por imagen "
            "(token CLS); nuestras celdas I-JEPA usan máscaras (no vistas), avg-pool y las épocas "
            "indicadas en el protocolo de la grilla. No son estrictamente comparables en costo.")


# ---------------------------------------------------------------- loading
def read_csv_dedup(path, key="epoch"):
    """Rows of a CSV as dicts of floats, deduplicated by epoch (keep last), sorted by epoch."""
    path = Path(path)
    if not path.exists():
        return []
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rr = {k: float(v) for k, v in r.items() if v not in (None, "")}
            except ValueError:
                continue
            if key in rr:
                rows[rr[key]] = rr
    return [rows[k] for k in sorted(rows)]


def load_runs(runs_dir):
    """All runs with an evaluation.json -> list of dicts (name, ev, dir)."""
    out = []
    for p in sorted(Path(runs_dir).glob("*/evaluation.json")):
        try:
            ev = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"name": p.parent.name, "dir": p.parent, "ev": ev})
    return out


def grid_runs(runs):
    """-> {cell: [run with .seed, .status, .linear, .knn, ...]} for runs named grid_c<n>_seed<k>."""
    cells = {}
    for r in runs:
        m = re.fullmatch(r"grid_(c[1-6])_seed(\d+)", r["name"])
        if not m:
            continue
        ev = r["ev"]
        met = ev.get("metrics") or {}
        cf = ev.get("collapse_final") or {}
        d = {"cell": m.group(1), "seed": int(m.group(2)), "name": r["name"], "dir": r["dir"],
             "status": ev.get("status", "ok"),
             "linear": met["linear_top1"] * 100 if "linear_top1" in met else None,
             "knn": met["knn_top1"] * 100 if "knn_top1" in met else None,
             "erank": cf.get("erank_pooled"), "uniformity": cf.get("uniformity"),
             "alignment": cf.get("alignment"), "gpu_h": ev.get("gpu_hours"),
             "epochs": ev.get("epochs_completed"), "collapse_epoch": ev.get("collapse_epoch"),
             "eval_split": ev.get("eval_split")}
        cells.setdefault(m.group(1), []).append(d)
    for v in cells.values():
        v.sort(key=lambda d: d["seed"])
    return cells


# ---------------------------------------------------------------- statistics
def mean_sd(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None, 0
    m = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
    return m, sd, len(xs)


def fmt(m, sd, digits=2):
    if m is None:
        return "—"
    return f"{m:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{m:.{digits}f} (n=1)"


def fmt_cell(ms, note, digits=2):
    """Combined-table cell: published rows without sd are plain numbers, ours show (n=1)."""
    m, sd = ms
    return f"{m:.{digits}f}" if (note == "publicado" and sd is None and m is not None) else fmt(m, sd, digits)


def compare(a_runs, b_runs, metric):
    """A vs B on `metric`. -> dict(diff, sd, p, verdict).

    "distinguible" if |diff of means| > 2 * max(sd_A, sd_B)  OR the paired t-test over common seeds
    gives p < 0.05; "no distinguible" otherwise; "no evaluable" if a side has no data or no sd
    can be computed (n < 2 on both sides, no usable paired test)."""
    a = {r["seed"]: r[metric] for r in a_runs if r.get(metric) is not None}
    b = {r["seed"]: r[metric] for r in b_runs if r.get(metric) is not None}
    if not a or not b:
        return {"diff": None, "sd": None, "p": None, "verdict": "no evaluable (faltan datos)"}
    ma, sda, _ = mean_sd(a.values())
    mb, sdb, _ = mean_sd(b.values())
    diff = ma - mb
    sds = [s for s in (sda, sdb) if s is not None]
    sd = max(sds) if sds else None
    p = None
    common = sorted(set(a) & set(b))
    if len(common) >= 2:
        from scipy import stats
        x, y = np.array([a[s] for s in common]), np.array([b[s] for s in common])
        if np.allclose(x - y, (x - y)[0]):   # zero variance of the differences
            p = 0.0 if abs((x - y)[0]) > 0 else 1.0
        else:
            p = float(stats.ttest_rel(x, y).pvalue)
    crit_sd = sd is not None and abs(diff) > 2 * sd
    crit_p = p is not None and p < 0.05
    if sd is None and p is None:
        verdict = "no evaluable (n<2)"
    else:
        verdict = "distinguible" if (crit_sd or crit_p) else "no distinguible"
    return {"diff": diff, "sd": sd, "p": p, "verdict": verdict}


def cell_summary(rs):
    ok = [r for r in rs if r["linear"] is not None]
    lin = mean_sd([r["linear"] for r in ok])
    knn = mean_sd([r["knn"] for r in ok])
    status = {}
    for r in rs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    return {"n": len(rs), "linear": lin, "knn": knn,
            "erank": mean_sd([r["erank"] for r in rs])[0],
            "uniformity": mean_sd([r["uniformity"] for r in rs])[0],
            "alignment": mean_sd([r["alignment"] for r in rs])[0],
            "gpu_h_mean": mean_sd([r["gpu_h"] for r in rs])[0],
            "gpu_h_total": float(sum(r["gpu_h"] for r in rs if r["gpu_h"] is not None)),
            "status": status, "epochs": sorted({r["epochs"] for r in rs if r["epochs"] is not None})}


def status_str(s):
    bad = {k: v for k, v in s["status"].items() if k != "ok"}
    return "ok" if not bad else ", ".join(f"{k} {v}/{s['n']}" for k, v in sorted(bad.items()))


# ---------------------------------------------------------------- tables
def grid_tables(cells):
    """-> (csv_rows, md_text, combined_rows) from grid_runs() output."""
    rows, summ = [], {}
    for c in CELLS:
        if c not in cells:
            continue
        s = summ[c] = cell_summary(cells[c])
        rows.append({
            "cell": c.upper(), "description": CELL_DESC[c], "n_seeds": s["n"], "status": status_str(s),
            "epochs": "/".join(str(e) for e in s["epochs"]),
            "linear_mean": s["linear"][0], "linear_sd": s["linear"][1],
            "knn_mean": s["knn"][0], "knn_sd": s["knn"][1], "erank_pooled": s["erank"],
            "uniformity": s["uniformity"], "alignment": s["alignment"],
            "gpu_h_per_run": s["gpu_h_mean"], "gpu_h_total": s["gpu_h_total"]})
    L = ["# Grilla I-JEPA (ImageNet-100, ViT-Tiny)", "",
         "Accuracy top-1 en % (media ± desvío estándar muestral entre semillas; `(n=1)` = una sola semilla, "
         "sin desvío). Rango efectivo, uniformidad y alineación: finales, embeddings pooled, media entre semillas.", "",
         "| Celda | Descripción | n semillas | Estado | Lineal | k-NN | Rango ef. | Uniformidad | Alineación | GPU-h/corrida | GPU-h total |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda x, d=2: "—" if x is None else f"{x:.{d}f}"
    for r in rows:
        s = summ[r["cell"].lower()]
        L.append(f"| {r['cell']} | {r['description']} | {r['n_seeds']} | {r['status']} | "
                 f"{fmt(*s['linear'][:2])} | {fmt(*s['knn'][:2])} | {f2(r['erank_pooled'], 1)} | "
                 f"{f2(r['uniformity'], 3)} | {f2(r['alignment'], 3)} | {f2(r['gpu_h_per_run'])} | {f2(r['gpu_h_total'])} |")
    missing = [c.upper() for c in CELLS if c not in cells]
    if missing:
        L += ["", f"Celdas sin resultados todavía: {', '.join(missing)}."]
    L += ["", "## Comparaciones (diferencia de medias A − B)", "",
          "\"distinguible\" si |diferencia| > 2 desvíos (el mayor de los dos) o si el test t pareado por "
          "semilla da p < 0.05; si no, \"no distinguible\" (PLAN §6).", "",
          "| Pregunta | A vs B | Métrica | Dif. (pts) | p (t pareado) | Veredicto |", "|---|---|---|---|---|---|"]
    comps = []
    for q, a, b in COMPARISONS:
        for metric, lab in (("linear", "lineal"), ("knn", "k-NN")):
            res = compare(cells.get(a, []), cells.get(b, []), metric)
            comps.append({"question": q, "a": a.upper(), "b": b.upper(), "metric": lab, **res})
            L.append(f"| {q} | {a.upper()} vs {b.upper()} | {lab} | "
                     f"{'—' if res['diff'] is None else format(res['diff'], '+.2f')} | "
                     f"{'—' if res['p'] is None else format(res['p'], '.3f')} | {res['verdict']} |")
    combined = [(n, (lm, ls), (km, ks), "publicado") for n, lm, ls, km, ks in TABLE7]
    for r in rows:
        s = summ[r["cell"].lower()]
        combined.append((f"{r['cell']}: {r['description']}", s["linear"][:2], s["knn"][:2],
                         f"n={s['n']}, {status_str(s)}"))
    L += ["", "## Tabla combinada con la tabla 7 publicada", "",
          "| Método | Lineal | k-NN | Notas |", "|---|---|---|---|"]
    for name, lin, knn, note in combined:
        L.append(f"| {name} | {fmt_cell(lin, note)} | {fmt_cell(knn, note)} | {note} |")
    L += ["", FOOTNOTE, ""]
    return rows, "\n".join(L), combined


def sweep_tables(runs):
    """-> markdown for fase 2 sweeps: best lr (C1) and best lambda (C4, C6) by val_interno kNN."""
    excluded, skipped_none = [], []
    groups = {}   # (title, param) -> [(value, knn, status, name)]
    for r in runs:
        n, ev = r["name"], r["ev"]
        met, cfg = ev.get("metrics") or {}, ev.get("config") or {}
        if "knn_top1" not in met:
            continue
        if ev.get("eval_split") == "oficial" or cfg.get("data", {}).get("eval_split") == "oficial":
            excluded.append(n)   # sweeps select on val_interno only; never use the official val
            continue
        if n.startswith("sweep_lr"):
            key, val = ("lr sweep (C1)", "lr"), cfg.get("optimization", {}).get("lr")
        else:
            m = re.match(r"sweep_lambda_(c[46])_", n)
            if not m:
                continue
            key, val = (f"λ sweep ({m.group(1).upper()})", "λ"), cfg.get("model", {}).get("lambda_reg")
        if val is None:
            skipped_none.append(n)
            continue
        groups.setdefault(key, []).append((val, met["knn_top1"] * 100, ev.get("status", "ok"), n,
                                           ev.get("eval_split")))
    L = ["# Barridos fase 2 (30 épocas, semilla 0, k-NN en val_interno)", ""]
    picks = {}
    for (title, param), items in sorted(groups.items()):
        items.sort(key=lambda t: (t[0] is None, t[0]))
        good = [t for t in items if t[2] == "ok"]
        best = max(good, key=lambda t: t[1]) if good else None
        L += [f"## {title}", "", f"| {param} | k-NN val_interno (%) | estado | corrida |", "|---|---|---|---|"]
        for v, k, st, n, es in items:
            star = " **(mejor)**" if best and n == best[3] else ""
            L.append(f"| {v:g} | {k:.2f}{star} | {st} | {n} |")
        L.append("")
        if best:
            picks[title] = best[0]
            L += [f"Elegido: {param} = {best[0]:g} (k-NN {best[1]:.2f}%).", ""]
    if excluded:
        L += [f"Excluidas (eval_split=oficial; los barridos usan val_interno): {', '.join(excluded)}", ""]
    if skipped_none:
        L += [f"Ignoradas (sin valor del parámetro barrido en la config): {', '.join(skipped_none)}", ""]
    if not groups:
        L.append("Sin resultados de barridos todavía.")
    else:
        parts = []
        if "lr sweep (C1)" in picks:
            parts.append(f"--var LR={picks['lr sweep (C1)']:g}")
        if "λ sweep (C4)" in picks:
            parts.append(f"--var LAMBDA_SIGREG={picks['λ sweep (C4)']:g}")
        if "λ sweep (C6)" in picks:
            parts.append(f"--var LAMBDA_SUSREG={picks['λ sweep (C6)']:g}")
        L += ["## Para la cola de la grilla", "", "`" + " ".join(parts) + "`"
              " (o editar LR / LAMBDA_* en queues/fase3_grid.txt)", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def write_grid(runs_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = grid_runs(load_runs(runs_dir))
    rows, md, _ = grid_tables(cells)
    cols = ["cell", "description", "n_seeds", "status", "epochs", "linear_mean", "linear_sd", "knn_mean",
            "knn_sd", "erank_pooled", "uniformity", "alignment", "gpu_h_per_run", "gpu_h_total"]
    with open(out_dir / "tabla_grilla.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "tabla_grilla.md").write_text(md)
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "sweeps", "all"], nargs="?", default="all")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    if a.mode in ("grid", "all"):
        cells = write_grid(a.runs_dir, a.results_dir)
        print(f"grid: {sum(len(v) for v in cells.values())} runs in {len(cells)} cells -> {a.results_dir}/tabla_grilla.*")
    if a.mode in ("sweeps", "all"):
        md = sweep_tables(load_runs(a.runs_dir))
        (Path(a.results_dir) / "sweeps.md").write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
