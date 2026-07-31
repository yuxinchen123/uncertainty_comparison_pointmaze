#!/usr/bin/env python
"""Convergence run 1 result tables (.tex), generated from analysis/data/fits.json:
- slope_table_perseed.tex  (R1: per-seed fits, slope mean +- SE over the 30 seeds)
- slope_table_seedavg.tex  (R2: seed-average-curve fits, same row order as R1)
- opt_effect_table.tex     (optimizer hyperparameter effect, per-seed slopes averaged over the
                            4 initializations within each seed, then mean +- SE over seeds)
Rows ranked by closeness of R1's aggregate slope to -0.5; per column the closest cell is bold,
the second underlined; cells whose every run diverged show "div."; n < 30 is annotated.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python make_results.py
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
PLOTS = os.path.join(ANALYSIS, "plots")
FITS = json.load(open(os.path.join(ANALYSIS, "data", "fits.json")))
TARGET = FITS["target_slope"]
ENV_COLS = FITS["env_cols"]
ENV_HEAD = {"center_square": "env 1.1", "top_right_cell": "env 1.2",
            "cell_midpoints": "env 2", "aggregate": "aggregate"}

INIT_TEX = {"I1-zero": "zero-bias", "I2-ptbias": "pytorch-bias",
            "I3-ptfull": "pytorch-full", "I4-normal0.5": "normal-0.5"}

BY = {(r["init"], r["opt"], r["env"]): r for r in FITS["results"]}


def exp_tex(v: float) -> str:
    """Format a power-of-ten value as 10^{k} LaTeX (values here are exact decades)."""
    return "10^{%d}" % round(math.log10(v))


def opt_tex(opt: str) -> str:
    """Pretty LaTeX for an optimizer-config label like 'adam-lr0.001' / 'sgd1t-eta0.01-t1000'."""
    if opt.startswith("adam-lr"):
        return "Adam $\\mathrm{lr}\\,%s$" % exp_tex(float(opt[len("adam-lr"):]))
    body = opt[len("sgd1t-eta"):]
    eta, t0 = body.split("-t")
    return "SGD-$1/t$ $\\eta_0\\,%s$ $t_0\\,%s$" % (exp_tex(float(eta)), exp_tex(float(t0)))


def cell_tex(row: dict, kind: str) -> str:
    """One table cell: slope +- SE (per-seed) or the single slope (seed-average); 'div.' if the
    cell has no surviving seeds; small (n=..) annotation when n < 30."""
    if row is None or "slope_mean" not in row:
        return "div."
    if kind == "perseed":
        se = row["slope_se"]
        fmt = "%.3f" if se < 0.005 else "%.2f"
        txt = "$" + (fmt % row["slope_mean"]) + " \\pm " + (fmt % se) + "$"
    else:
        txt = "$%.2f$" % row["avg_slope"]
    if row["n_seeds"] < 30:
        txt += " {\\scriptsize($n{=}%d$)}" % row["n_seeds"]
    return txt


def mark_best(cells, values):
    """Bold the value closest to TARGET, underline the second; cells/values aligned lists."""
    order = sorted((i for i, v in enumerate(values) if v is not None),
                   key=lambda i: abs(values[i] - TARGET))
    if order:
        cells[order[0]] = "\\textbf{" + cells[order[0]] + "}"
    if len(order) > 1:
        cells[order[1]] = "\\underline{" + cells[order[1]] + "}"
    return cells


def config_rows():
    """All 48 (init, opt) configs ranked by closeness of the per-seed aggregate slope to TARGET
    (configs with no aggregate fit sink to the bottom in grid order)."""
    rows = [(i, o) for i in FITS["init_order"] for o in FITS["opt_order"]]

    def key(io):
        r = BY.get((io[0], io[1], "aggregate"))
        return abs(r["slope_mean"] - TARGET) if r and "slope_mean" in r else float("inf")
    return sorted(rows, key=key)


def emit_slope_table(kind: str, label: str, caption: str, path: str) -> None:
    """Write one ranked 48-row slope table (per-seed or seed-average granularity)."""
    ranked = config_rows()
    # column-wise best/second marking needs every column's values first
    col_values = {e: [] for e in ENV_COLS}
    col_cells = {e: [] for e in ENV_COLS}
    for init, opt in ranked:
        for e in ENV_COLS:
            r = BY.get((init, opt, e))
            has = r is not None and "slope_mean" in r
            v = (r["slope_mean"] if kind == "perseed" else r["avg_slope"]) if has else None
            col_values[e].append(v)
            col_cells[e].append(cell_tex(r, kind))
    for e in ENV_COLS:
        mark_best(col_cells[e], col_values[e])
    lines = [
        "\\begin{table}[H]", "\\centering", "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.0}",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.2cm} cccc@{}}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{env 1.1} & \\textbf{env 1.2} & \\textbf{env 2} "
        "& \\textbf{aggregate} \\\\", "\\midrule",
    ]
    for i, (init, opt) in enumerate(ranked):
        cells = " & ".join(col_cells[e][i] for e in ENV_COLS)
        lines.append("%s --- %s & %s \\\\" % (INIT_TEX[init], opt_tex(opt), cells))
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{%s}" % caption, "\\label{%s}" % label, "\\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def emit_effect_table() -> None:
    """Optimizer hyperparameter-effect table: per-seed slopes averaged over the 4 inits WITHIN
    each seed, then mean +- SE over seeds; one row per optimizer configuration."""
    lines = [
        "\\begin{table}[H]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}", "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.6cm} cccc@{}}",
        "\\toprule",
        "\\textbf{Optimizer configuration} & \\textbf{env 1.1} & \\textbf{env 1.2} & "
        "\\textbf{env 2} & \\textbf{aggregate} \\\\", "\\midrule",
    ]
    col_values = {e: [] for e in ENV_COLS}
    col_cells = {e: [] for e in ENV_COLS}
    for opt in FITS["opt_order"]:
        for e in ENV_COLS:
            # within-seed average over inits: seed -> [slopes of the inits that survived]
            per_seed = defaultdict(list)
            n_inits = 0
            for init in FITS["init_order"]:
                r = BY.get((init, opt, e))
                if r and "slope_mean" in r:
                    n_inits += 1
                    for s, sl in zip(r["seeds"], r["per_seed_slopes"]):
                        per_seed[s].append(sl)
            if not per_seed:
                col_values[e].append(None)
                col_cells[e].append("div.")
                continue
            vals = np.array([np.mean(v) for v in per_seed.values()])
            mean = float(vals.mean())
            se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            fmt = "%.3f" if se < 0.005 else "%.2f"
            txt = "$" + (fmt % mean) + " \\pm " + (fmt % se) + "$"
            if n_inits < 4:
                txt += " {\\scriptsize(%d/4)}" % n_inits
            col_values[e].append(mean)
            col_cells[e].append(txt)
    for e in ENV_COLS:
        mark_best(col_cells[e], col_values[e])
    for i, opt in enumerate(FITS["opt_order"]):
        lines.append("%s & %s \\\\" % (opt_tex(opt), " & ".join(col_cells[e][i] for e in ENV_COLS)))
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Optimizer hyperparameter effect on the fitted slope (per-seed fits, slopes "
        "averaged over the four initializations within each seed, then mean $\\pm$ standard "
        "error over the 30 seeds). Best per column (closest to $-1/2$) bold, second underlined. "
        "``div.''\\ marks cells whose runs all diverged; ``($k$/4)'' marks cells where only "
        "$k$ of the 4 initializations survived (SGD-$1/t$ at $\\eta_0\\,10^{-1}$ on env~2 "
        "diverged for pytorch-full/normal-0.5 at every $t_0$).}",
        "\\label{tab:convergence-run1-opt-effect}", "\\end{table}"]
    path = os.path.join(PLOTS, "opt_effect_table.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


def main() -> None:
    """Emit the three result tables."""
    os.makedirs(PLOTS, exist_ok=True)
    emit_slope_table(
        "perseed", "tab:convergence-run1-slopes-perseed",
        "Convergence run~1, PER-SEED fits: the fitted slope $-\\hat\\alpha$ (mean $\\pm$ "
        "standard error over the 30 seeds) per configuration and point set; ``aggregate'' pools "
        "all 308 points of the three sets. Rows ranked by closeness of the aggregate slope to "
        "the reference $-1/2$ (the ranking metric; best per column bold, second underlined --- "
        "``best'' means CLOSEST to $-1/2$, not largest). ``div.''\\ marks cells whose 30 runs "
        "all diverged (excluded from every fit); $(n{=}k)$ marks partial survival.",
        os.path.join(PLOTS, "slope_table_perseed.tex"))
    emit_slope_table(
        "seedavg", "tab:convergence-run1-slopes-seedavg",
        "Convergence run~1, SEED-AVERAGE fits: the slope of the single mean curve (all seeds "
        "and points averaged before fitting), same row order as "
        "Table~\\ref{tab:convergence-run1-slopes-perseed} for comparison. Agreement with the "
        "per-seed table indicates the estimate is stable; a gap flags seed heterogeneity.",
        os.path.join(PLOTS, "slope_table_seedavg.tex"))
    emit_effect_table()


if __name__ == "__main__":
    main()
