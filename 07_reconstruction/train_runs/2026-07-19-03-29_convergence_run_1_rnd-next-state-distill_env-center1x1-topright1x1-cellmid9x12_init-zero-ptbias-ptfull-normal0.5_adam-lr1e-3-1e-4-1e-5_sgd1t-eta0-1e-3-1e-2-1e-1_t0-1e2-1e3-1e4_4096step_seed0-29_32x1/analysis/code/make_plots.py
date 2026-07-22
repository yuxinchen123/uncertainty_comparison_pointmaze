#!/usr/bin/env python
"""Convergence run 1 figures (PDF, into analysis/plots/), from analysis/data/fits.json:
- curves_env11 / curves_env12 / curves_env2: log-log normalized mean curves with seed-SE bands,
  the grey SOLID slope -1/2 reference, 8 lines = 4 initializations x {best Adam, best SGD-1/t}
  (chosen per env + init by per-seed slope closeness to -1/2), plus a local-exponent lower panel.
- opt_effect: fitted slope vs learning rate (eta0 for SGD, one line per t0; lr for Adam), one
  panel per env column, slopes averaged over the four inits within each seed, SE bands over seeds.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python make_plots.py
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
PLOTS = os.path.join(ANALYSIS, "plots")
FITS = json.load(open(os.path.join(ANALYSIS, "data", "fits.json")))
TARGET = FITS["target_slope"]
BY = {(r["init"], r["opt"], r["env"]): r for r in FITS["results"]}

# faint per-seed sublines (both panels). Alpha chosen by visual iteration 2026-07-19 over
# {0.10, 0.18, 0.25}: 0.10 left single seeds untraceable, 0.25 blurred the mean lines into
# their halos in dense regions; 0.18 keeps the means crisp with every seed followable.
SUBLINE_ALPHA = float(os.environ.get("SUBLINE_ALPHA", "0.18"))
SUBLINE_LW = 0.5
BURN_IN = FITS["burn_in"]

ENV_FILE = {"center_square": "env11", "top_right_cell": "env12", "cell_midpoints": "env2"}
ENV_TITLE = {"center_square": "env 1.1 (center square)",
             "top_right_cell": "env 1.2 (top-right cell)",
             "cell_midpoints": "env 2 (cell midpoints)", "aggregate": "aggregate"}
INIT_NICE = {"I1-zero": "I1 zero", "I2-ptbias": "I2 pt-bias",
             "I3-ptfull": "I3 pt-full", "I4-normal0.5": "I4 normal-0.5"}
COLORS = dict(zip(FITS["init_order"], ["tab:blue", "tab:orange", "tab:green", "tab:red"]))


def short_opt(opt: str) -> str:
    """Compact legend text for an optimizer-config label."""
    if opt.startswith("adam-lr"):
        return "Adam lr 1e%d" % round(math.log10(float(opt[7:])))
    eta, t0 = opt[len("sgd1t-eta"):].split("-t")
    return "SGD e0 1e%d t0 1e%d" % (round(math.log10(float(eta))), round(math.log10(float(t0))))


def local_exp(steps: np.ndarray, y: np.ndarray, c: float):
    """alpha_eff(n) = -dlog(y-c)/dlog(n) between successive in-window checkpoints (n >= burn-in,
    masked where y <= c) — the same formula fit_convergence.py uses for the mean curve."""
    m = steps >= BURN_IN
    n, yy = steps[m].astype(float), y[m] - c
    mids, vals = [], []
    for i in range(len(n) - 1):
        if yy[i] > 0 and yy[i + 1] > 0:
            mids.append(float(np.sqrt(n[i] * n[i + 1])))
            vals.append(float(-(np.log(yy[i + 1]) - np.log(yy[i]))
                              / (np.log(n[i + 1]) - np.log(n[i]))))
    return mids, vals


def best_config(init: str, env: str, family: str):
    """The family's config (adam / sgd1t prefix) whose per-seed mean slope in this env is
    closest to TARGET; None if every candidate diverged out."""
    cands = [(o, BY.get((init, o, env))) for o in FITS["opt_order"] if o.startswith(family)]
    cands = [(o, r) for o, r in cands if r and "slope_mean" in r]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c[1]["slope_mean"] - TARGET))


def main_figure(env: str) -> None:
    """One per-point-set figure: mean curves + SE bands + reference (top), local exponent
    (bottom)."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 7.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1.0]})
    shown = []
    line_idx = 0
    for init in FITS["init_order"]:
        for family, ls in (("adam", "--"), ("sgd1t", "-")):
            pick = best_config(init, env, family)
            if pick is None:
                continue
            opt, r = pick
            steps = np.array(r["steps"][1:])              # drop step 0 (log axis)
            mean = np.array(r["curve_mean"][1:])
            se = np.array(r["curve_se"][1:])
            # the 30 faint per-seed sublines, both panels (drawn first, lowest zorder); the
            # alpha panel uses each seed's own fitted floor for its local exponent
            all_steps = np.array(r["steps"])
            for y_seed, c_seed in zip(r["per_seed_curves"], r["per_seed_cs"]):
                y_seed = np.array(y_seed)
                ax.plot(steps, y_seed[1:], ls, color=COLORS[init],
                        alpha=SUBLINE_ALPHA, lw=SUBLINE_LW, zorder=1)
                mids, vals = local_exp(all_steps, y_seed, c_seed)
                ax2.plot(mids, vals, ls, color=COLORS[init],
                         alpha=SUBLINE_ALPHA, lw=SUBLINE_LW, zorder=1)
            ax.plot(steps, mean, ls, color=COLORS[init], lw=1.4, zorder=3,
                    label=f"{INIT_NICE[init]} — {short_opt(opt)} (slope {r['slope_mean']:.2f})")
            ax.fill_between(steps, mean - se, mean + se, color=COLORS[init], alpha=0.25,
                            lw=0, zorder=2)
            # the seed SE is often thinner than the line on a log axis, so also draw it as
            # capped bars at every 6th checkpoint, staggered per line so caps don't overlap
            sel = np.arange(line_idx % 6, len(steps), 6)
            ax.errorbar(steps[sel], mean[sel], yerr=se[sel], fmt="none", zorder=4,
                        ecolor=COLORS[init], elinewidth=0.9, capsize=2.4, capthick=0.9)
            ax2.plot(r["alpha_eff_steps"], r["alpha_eff"], ls, color=COLORS[init], lw=1.2,
                     zorder=3)
            shown.append(mean)
            line_idx += 1
    # grey SOLID slope -1/2 reference, anchored just above the shown curves at n = 2
    top = max(m[0] for m in shown)
    n_ref = np.array([2.0, 4096.0])
    ax.plot(n_ref, (1.1 * top) * np.sqrt(2.0) / np.sqrt(n_ref), "-", color="0.45", lw=1.8,
            label="reference slope $-1/2$", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("normalized l2 bonus $\\bar y(n)$")
    ax.set_title(f"Convergence run 1 — {ENV_TITLE[env]}")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax2.axhline(0.5, color="0.45", lw=1.4)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.15, 2.0)
    ax2.set_xlabel("optimizer step $n$")
    ax2.set_ylabel("local exponent $\\alpha_{\\mathrm{eff}}(n)$")
    fig.tight_layout()
    out = os.path.join(PLOTS, f"curves_{ENV_FILE[env]}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def effect_figure() -> None:
    """Slope vs learning rate per env column: one line per SGD t0 plus an Adam line, slopes
    averaged over inits within each seed, SE over seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharey=True)
    for ax, env in zip(axes.ravel(), FITS["env_cols"]):
        series = defaultdict(list)  # (family, t0-or-'') -> [(rate, mean, se)]
        for opt in FITS["opt_order"]:
            per_seed = defaultdict(list)
            for init in FITS["init_order"]:
                r = BY.get((init, opt, env))
                if r and "slope_mean" in r:
                    for s, sl in zip(r["seeds"], r["per_seed_slopes"]):
                        per_seed[s].append(sl)
            if not per_seed:
                continue
            vals = np.array([np.mean(v) for v in per_seed.values()])
            mean, se = float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))
            if opt.startswith("adam-lr"):
                series[("Adam", "")].append((float(opt[7:]), mean, se))
            else:
                eta, t0 = opt[len("sgd1t-eta"):].split("-t")
                series[("SGD", t0)].append((float(eta), mean, se))
        for (fam, t0), pts in sorted(series.items()):
            pts.sort()
            x, m, s = zip(*pts)
            lbl = "Adam (lr)" if fam == "Adam" else "SGD-1/t t0 1e%d" % round(math.log10(float(t0)))
            ls = ":" if fam == "Adam" else "-"
            ax.errorbar(x, m, yerr=s, fmt="o" + ls, ms=3.5, lw=1.2, capsize=2, label=lbl)
        ax.axhline(TARGET, color="0.45", lw=1.4)
        ax.set_xscale("log")
        ax.set_title(ENV_TITLE[env], fontsize=10)
        ax.set_xlabel("learning rate ($\\eta_0$ for SGD, lr for Adam)", fontsize=9)
        ax.set_ylabel("fitted slope $-\\hat\\alpha$", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Optimizer hyperparameter effect on the fitted slope "
                 "(within-seed init average; bars = seed SE)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOTS, "opt_effect.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    """Emit the three main figures and the effect figure."""
    os.makedirs(PLOTS, exist_ok=True)
    for env in ("center_square", "top_right_cell", "cell_midpoints"):
        main_figure(env)
    effect_figure()


if __name__ == "__main__":
    main()
