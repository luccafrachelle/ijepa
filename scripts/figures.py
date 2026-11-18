#!/usr/bin/env python
"""Figures for the grid (PLAN §7 Salida): PNG + PDF in results/figs/. Tolerates partial results.

    .venv/bin/python scripts/figures.py [--runs-dir runs] [--results-dir results]
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_aggregate():
    spec = importlib.util.spec_from_file_location("ijepa_aggregate", Path(__file__).with_name("aggregate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


agg = _load_aggregate()

# Okabe-Ito (colour-blind safe), one colour per cell.
COLORS = {"c1": "#000000", "c2": "#D55E00", "c3": "#56B4E9", "c4": "#0072B2", "c5": "#E69F00", "c6": "#009E73"}
# layout: rows = EMA yes/no, columns = regularizer none/SIGReg/SUSReg
LAYOUT = [["c1", "c3", "c5"], ["c2", "c4", "c6"]]


def _save(fig, figs_dir, name):
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _empty(ax, msg="sin datos"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xticks([])


def fig_grid_bars(cells, figs_dir):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharey=True)
    for i, row in enumerate(LAYOUT):
        for j, c in enumerate(row):
            ax = axes[i][j]
            ax.set_title(f"{c.upper()}: {agg.CELL_DESC[c]}", fontsize=9)
            rs = cells.get(c, [])
            s = agg.cell_summary(rs) if rs else None
            if not rs or s["linear"][0] is None:
                st = {r["status"] for r in rs}
                _empty(ax, "colapsa (sin métricas finales)" if "collapsed" in st else
                       "diverge" if "diverged" in st else "sin datos")
                continue
            for k, (name, key) in enumerate((("lineal", "linear"), ("k-NN", "knn"))):
                m, sd, n = s[key]
                ax.bar(k, m, yerr=sd if sd is not None else 0, capsize=4, color=COLORS[c], alpha=0.85,
                       edgecolor="black", linewidth=2.5 if c == "c1" else 0.8)
                pts = [r[key] for r in rs if r[key] is not None]
                ax.scatter([k] * len(pts), pts, color="white", edgecolor="black", s=14, zorder=3)
            ax.set_xticks([0, 1], ["lineal", "k-NN"])
            collapsed = any(r["status"] == "collapsed" for r in rs)
            if collapsed:
                ax.text(0.5, 0.97, "colapsa", transform=ax.transAxes, ha="center", va="top", fontsize=11,
                        color="#D55E00", weight="bold")
            elif any(r["status"] == "diverged" for r in rs):
                ax.text(0.5, 0.92, "diverge", transform=ax.transAxes, ha="center", fontsize=11, color="red")
            if c == "c1":
                ax.set_title(f"C1 (referencia): {agg.CELL_DESC[c]}", fontsize=9, weight="bold")
            ax.text(0.98, 0.02, f"n={s['n']}", transform=ax.transAxes, ha="right", fontsize=8)
    for i in range(2):
        axes[i][0].set_ylabel("top-1 (%)")
    fig.suptitle("Grilla I-JEPA: lineal y k-NN (media ± sd; puntos = semillas)")
    fig.tight_layout()
    _save(fig, figs_dir, "fig1_grilla_barras")


def fig_erank(cells, figs_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_data = False
    for c in agg.CELLS:
        per_seed = [agg.read_csv_dedup(Path(r["dir"]) / "collapse.csv") for r in cells.get(c, [])]
        per_seed = [rows for rows in per_seed if rows]
        if not per_seed:
            continue
        any_data = True
        for rows in per_seed:
            ax.plot([r["epoch"] for r in rows], [r["erank_pooled"] for r in rows], color=COLORS[c],
                    alpha=0.25, lw=0.8)
        ep = sorted(set.intersection(*[{r["epoch"] for r in rows} for rows in per_seed]))
        if ep:
            mean = [np.mean([next(r["erank_pooled"] for r in rows if r["epoch"] == e) for rows in per_seed])
                    for e in ep]
            ax.plot(ep, mean, color=COLORS[c], lw=1.8, marker="o", ms=3, label=c.upper())
    if any_data:
        ax.set_yscale("log")
        ax.legend(title="celda", ncol=2)
    else:
        _empty(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("época")
    ax.set_ylabel("rango efectivo (pooled)")
    ax.set_title("Rango efectivo vs época (línea gruesa = media de semillas)")
    ax.grid(alpha=0.3)
    _save(fig, figs_dir, "fig2_rango_efectivo")


def fig_unif_align(cells, figs_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    any_data = False
    for c in agg.CELLS:
        rs = [r for r in cells.get(c, []) if r["uniformity"] is not None and r["alignment"] is not None]
        if not rs:
            continue
        any_data = True
        ax.scatter([r["alignment"] for r in rs], [r["uniformity"] for r in rs], color=COLORS[c],
                   s=45, label=c.upper(), edgecolor="black", linewidth=0.6)
    if any_data:
        ax.legend(title="celda")
    else:
        _empty(ax)
    ax.set_xlabel("alineación (menor = mejor)")
    ax.set_ylabel("uniformidad (menor = mejor)")
    ax.set_title("Uniformidad vs alineación al final (una marca por semilla)")
    ax.grid(alpha=0.3)
    _save(fig, figs_dir, "fig3_uniformidad_alineacion")


def fig_table(cells, figs_dir):
    _, _, combined = agg.grid_tables(cells)
    data = [[n, agg.fmt_cell(lin, note), agg.fmt_cell(knn, note), note] for n, lin, knn, note in combined]
    fig, ax = plt.subplots(figsize=(11, 0.45 * (len(data) + 3)))
    ax.axis("off")
    t = ax.table(cellText=data, colLabels=["Método", "Lineal (%)", "k-NN (%)", "Notas"],
                 loc="center", cellLoc="left", colWidths=[0.40, 0.14, 0.14, 0.32])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    for (r, _c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
        elif data[r - 1][3] == "publicado":
            cell.set_facecolor("#f2f2f2")
    ax.set_title("Tabla combinada: tabla 7 publicada + celdas I-JEPA", fontsize=11)
    fig.text(0.01, 0.01, "I-JEPA usa máscaras, no vistas (publicados: 200 ép., 6 vistas, CLS).", fontsize=8)
    _save(fig, figs_dir, "fig4_tabla_combinada")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    figs = Path(a.results_dir) / "figs"
    cells = agg.grid_runs(agg.load_runs(a.runs_dir))
    for f in (fig_grid_bars, fig_erank, fig_unif_align, fig_table):
        f(cells, figs)
    print(f"figures written to {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
