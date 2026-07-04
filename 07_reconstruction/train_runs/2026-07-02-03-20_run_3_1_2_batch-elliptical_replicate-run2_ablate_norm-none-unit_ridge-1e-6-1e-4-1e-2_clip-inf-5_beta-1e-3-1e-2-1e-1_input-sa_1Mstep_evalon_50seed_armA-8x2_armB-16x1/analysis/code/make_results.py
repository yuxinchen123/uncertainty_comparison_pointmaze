#!/usr/bin/env python3
"""Train run 3.1.2 results generator: the 12-cell table (each cell at its best beta) and the eval-reward
learning-curve figure, from both arms' completed runs pooled.

Definitions (also stated in the writeup):
- Per run, R_i = final eval extrinsic reward (the 100-episode deterministic eval at 1e6 steps, the same
  protocol as run-2's 44.49).
- Per (normalization, lambda, clip) CELL: pick the beta with the highest mean R over seeds (all three
  betas shown to the analysis; the table reports the winner). Rbar = mean, SE = std(ddof=1)/sqrt(n).
- Success rate = fraction of seeds with R_i > 5 (the pooled per-seed distribution is bimodal: a mass at
  ~0 and a mass well above 20; 5 sits in the gap). Wilson intervals are not shown in the table (SE and n
  are), success is a robustness companion to the bimodality-sensitive mean.
- rho = the measured effective ridge ratio from each run's intrinsic_diagnostics (mean over the cell's
  runs at the winning beta); clip-hit = the measured fraction of computed bonuses that exceeded the cap.

Outputs: <plots>/reward_table.tex, <plots>/line_reward_curve.{pdf,png}; console summary to stderr.
Usage: make_results.py [plots_dir]  (data dirs are fixed to this run's two arm sweeps)
"""
from __future__ import annotations

import datetime
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
RUN_DIR = CODE_DIR.parent.parent
SWEEPS = ["2026-07-02-03-52_armA-8tasks-2cpu_replicate-ablate",
          "2026-07-02-03-52_armB-16tasks-1cpu_replicate-ablate"]
RUN2_DATA = RUN_DIR.parent / "2026-06-24-21-24_run_2_after_reorganization" / "data" / "local"
SUCCESS_THRESHOLD = 5.0
EVAL_KEY = "eval/mean_extrinsic_reward"
TRAIN_KEY = "train/mean_extrinsic_reward"

# The arm-A/arm-B sweep was STOPPED on 2026-07-03 (all jobs cancelled), so the analysis is now final:
# load every completed run (the finished runs). While the sweep was still running this was pinned to the
# validated writeup snapshot (2026-07-03 14:46, replica cell n=33) so regenerating did not move the
# numbers; that pin is retired now that the run is done and we want all finished runs.
FREEZE_BEFORE = None  # local-time string to pin to a snapshot; None = load every completed run (final)


def _freeze_cutoff() -> float | None:
    """POSIX timestamp of the FREEZE_BEFORE snapshot instant (None when the freeze is disabled)."""
    return None if FREEZE_BEFORE is None else datetime.datetime.strptime(FREEZE_BEFORE, "%Y-%m-%d %H:%M").timestamp()


def _finite(x) -> bool:
    """True for a real finite number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_runs() -> list[dict]:
    """Load both arms' COMPLETED runs into tidy dicts.

    Before: per-run JSON with eval_history rows like {"step": 1000000, "eval/mean_extrinsic_reward": 52.1}
            and train_history rows like {"step": 1000000, "train/mean_extrinsic_reward": 8.4,
            "train/n_episodes_averaged": 100} (train mean over the past 100 training episodes at each eval).
    After:  {"cell": ("none", 1e-06, inf), "beta": 0.01, "seed": 7, "final_eval": 52.1,
             "final_train": 49.3, "eval_curve": [(50000, 1.2), ...],
             "train_curve": [(50000, 0.0), ...], "rho": 3.7e-06, "clip_hit": 0.0}.
    """
    runs = []
    cutoff = _freeze_cutoff()  # None = load all; a timestamp = only runs completed at/before the snapshot
    for sw in SWEEPS:
        for jf in sorted((RUN_DIR / "data" / sw / "local").glob("*.json")):
            # pin to the frozen writeup snapshot: skip runs whose last write is after the cutoff
            if cutoff is not None and jf.stat().st_mtime > cutoff:
                continue
            try:
                r = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if r.get("completed", True) is False:
                continue
            ev = [(int(e["step"]), float(e[EVAL_KEY])) for e in r.get("eval_history", [])
                  if EVAL_KEY in e and _finite(e.get(EVAL_KEY))]
            # train_history: mean extrinsic reward over the past 100 training episodes, same cadence as eval
            tr = [(int(e["step"]), float(e[TRAIN_KEY])) for e in r.get("train_history", [])
                  if TRAIN_KEY in e and _finite(e.get(TRAIN_KEY))]
            if not ev:
                continue
            ev.sort()
            tr.sort()
            d = r.get("intrinsic_diagnostics") or {}
            runs.append({
                "cell": (r["elliptical_feature_normalization"], float(r["elliptical_regularization"]),
                         float(r["elliptical_bonus_clip"])),
                "beta": float(r["beta"]),
                "seed": int(r["a_seed"]),
                "final_eval": ev[-1][1],
                "final_train": max(tr)[1] if tr else None,
                "eval_curve": ev,
                "train_curve": tr,
                "rho": d.get("effective_ridge_ratio"),
                "clip_hit": d.get("clip_hit_fraction"),
            })
    return runs


def cell_summaries(runs: list[dict]) -> list[dict]:
    """Per CELL: stats of every beta, then the winner. One output row per cell (12 rows).

    Before: run dicts; After: {"cell": ..., "beta": 0.01, "n": 33, "mean": 52.76, "se": 6.18,
    "success": 0.82, "train_mean": 49.9, "rho": 3.7e-6, "clip_hit": 0.0}, sorted by mean descending."""
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r["cell"], r["beta"])].append(r)
    rows = []
    for cell in sorted({r["cell"] for r in runs}):
        best = None
        for beta in (0.001, 0.01, 0.1):
            grp = by_cfg.get((cell, beta), [])
            if len(grp) < 2:
                continue
            R = [g["final_eval"] for g in grp]
            cand = {
                "cell": cell, "beta": beta, "n": len(R),
                "mean": statistics.mean(R),
                "se": statistics.stdev(R) / math.sqrt(len(R)),
                "success": sum(1 for x in R if x > SUCCESS_THRESHOLD) / len(R),
                "train_mean": statistics.mean([g["final_train"] for g in grp if g["final_train"] is not None]),
                "rho": statistics.mean([g["rho"] for g in grp if _finite(g.get("rho"))]),
                "clip_hit": statistics.mean([g["clip_hit"] for g in grp if _finite(g.get("clip_hit"))]),
            }
            if best is None or cand["mean"] > best["mean"]:
                best = cand
        if best:
            rows.append(best)
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows


def _pow10(x: float) -> str:
    """LaTeX power-of-ten for the grid values (1e-6 -> $10^{-6}$)."""
    k = round(math.log10(x))
    return f"$10^{{{k}}}$"


def _sci(x: float) -> str:
    """Compact scientific LaTeX for a measured value (3.7e-06 -> $3.7\\times10^{-6}$)."""
    if x == 0 or not _finite(x):
        return "---"
    k = math.floor(math.log10(abs(x)))
    m = x / 10 ** k
    return f"${m:.1f}\\times10^{{{k}}}$"


def _mark(values: list[float], formatted: list[str]) -> list[str]:
    """Bold the best and underline the second-best entry of one metric column (higher is better)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    out = list(formatted)
    if order:
        out[order[0]] = f"\\textbf{{{out[order[0]]}}}"
    if len(order) > 1:
        out[order[1]] = f"\\underline{{{out[order[1]]}}}"
    return out


def reward_table(rows: list[dict], plots: Path) -> None:
    """Write the 12-row LaTeX tabular: cells sorted by mean eval reward descending; per the analysis
    convention, EACH metric column (Rbar, success, train Rbar) gets its best bolded / second underlined;
    measured rho and clip-hit columns carry the mechanism evidence."""
    mean_s = _mark([r["mean"] for r in rows], [f"{r['mean']:.2f}" for r in rows])
    succ_s = _mark([r["success"] for r in rows], [f"{r['success']:.2f}" for r in rows])
    train_s = _mark([r["train_mean"] for r in rows], [f"{r['train_mean']:.2f}" for r in rows])
    lines = []
    for i, r in enumerate(rows):
        norm, lam, clip = r["cell"]
        clip_s = "$\\infty$" if math.isinf(clip) else f"{clip:g}"
        hit_s = "---" if math.isinf(clip) else f"{100 * r['clip_hit']:.0f}\\%"
        lines.append(
            f"\\texttt{{{norm}}} & {_pow10(lam)} & {clip_s} & {_sci(r['rho'])} & {hit_s} & "
            f"{_pow10(r['beta'])} & {mean_s[i]} & {r['se']:.2f} & {succ_s[i]} & {int(r['n'])} & "
            f"{train_s[i]} \\\\"
        )
    body = "\n".join(lines)
    (plots / "reward_table.tex").write_text(
        "% Train run 3.1.2 cell table (12 cells at each cell's best beta; pooled over both slurm arms).\n"
        "% Higher eval reward is better; rows sorted by Rbar descending; best bold, second underlined.\n"
        "\\begin{tabular}{@{}l c c c c c r r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Features} & \\textbf{$\\lambda$} & \\textbf{clip} & \\textbf{$\\rho$ (meas.)} & "
        "\\textbf{clip hits} & \\textbf{$\\beta^\\star$} & \\textbf{$\\bar{R}$ ($\\downarrow$)} & "
        "\\textbf{SE} & \\textbf{succ.} & \\textbf{$n$} & \\textbf{$\\bar{R}_{\\text{train}}$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# The curve series: the replica, the single-factor probes on the rho ladder, the 3.1.1-like cell.
CURVE_CELLS = [
    (("none", 1e-06, float("inf")), "raw, $\\lambda$=1e-6, no clip (run-2 replica)"),
    (("none", 0.0001, float("inf")), "raw, $\\lambda$=1e-4, no clip"),
    (("none", 0.01, float("inf")), "raw, $\\lambda$=1e-2, no clip"),
    (("unit", 1e-06, float("inf")), "unit, $\\lambda$=1e-6, no clip"),
    (("unit", 0.0001, float("inf")), "unit, $\\lambda$=1e-4, no clip"),
    (("unit", 0.01, 5.0), "unit, $\\lambda$=1e-2, clip 5 (run-3.1.1 config)"),
    (("none", 1e-06, 5.0), "raw, $\\lambda$=1e-6, clip 5"),
]
MIN_SEEDS_CURVE = 10  # a curve point needs at least this many seeds
RUN2_MIN_SEEDS = 30   # the run-2 reference curve needs at least this many seeds per step


def _pooled_curve(curves: list[list[tuple[int, float]]], min_seeds: int):
    """Pool a set of per-run (step, value) curves into per-step mean/SE over seeds, keeping only steps
    reached by at least min_seeds runs.

    Before: curves = [[(50000, 1.2), (1000000, 20.0)], [(50000, 0.9), ...], ...]  (one list per run).
    After:  (x, m, se) = ([0.05, ..., 1.0], [1.05, ..., 22.3], [0.31, ..., 4.1]) with x in millions of steps.
    """
    # bucket every run's value by step, then keep steps with enough seeds
    by_step = defaultdict(list)
    for c in curves:
        for step, val in c:
            by_step[step].append(val)
    steps = sorted(s for s, v in by_step.items() if len(v) >= min_seeds)
    m = [statistics.mean(by_step[s]) for s in steps]
    se = [statistics.stdev(by_step[s]) / math.sqrt(len(by_step[s])) if len(by_step[s]) > 1 else 0.0
          for s in steps]
    x = [s / 1e6 for s in steps]
    return x, m, se


def _run2_eval_curves() -> list[list[tuple[int, float]]]:
    """Load the run-2 rnd_elliptical (beta=0.01) per-seed eval-reward curves (the reference series).

    Before: run-2 per-run JSONs. After: [[(50000, 1.1), ..., (1000000, 44.0)], ...] one list per seed.
    """
    curves = []
    for jf in RUN2_DATA.glob("*.json"):
        try:
            r = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("algorithm") == "rnd_elliptical" and abs(float(r.get("beta", 0)) - 0.01) < 1e-9:
            c = [(int(e["step"]), float(e[EVAL_KEY])) for e in r.get("eval_history", [])
                 if EVAL_KEY in e and _finite(e.get(EVAL_KEY))]
            if c:
                curves.append(sorted(c))
    return curves


def curve_figure(runs: list[dict], rows: list[dict], plots: Path) -> None:
    """Reward-over-training figure. For each of the CURVE_CELLS at its best beta, two same-colored lines:
    SOLID = eval extrinsic reward (100-episode deterministic eval), DASHED = training extrinsic reward
    (mean over the past 100 training episodes). Plus the run-2 rnd_elliptical eval reference (gray dotted).
    Solid lines carry a shaded +-SE band; the dashed training lines are drawn without a band to keep the
    (7 cells x 2 lines) figure readable."""
    from matplotlib.lines import Line2D  # proxy handles for the solid/dashed style legend

    best_beta = {r["cell"]: r["beta"] for r in rows}
    cmap = matplotlib.colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    for k, (cell, label) in enumerate(CURVE_CELLS):
        beta = best_beta.get(cell)
        if beta is None:
            continue
        grp = [r for r in runs if r["cell"] == cell and r["beta"] == beta]
        # solid eval line + shaded SE band (per-step pooled over seeds, >= MIN_SEEDS_CURVE)
        xe, me, see = _pooled_curve([g["eval_curve"] for g in grp], MIN_SEEDS_CURVE)
        if not xe:
            continue
        ax.plot(xe, me, color=cmap(k), lw=1.8, label=f"{label} ($\\beta$={beta:g})")
        ax.fill_between(xe, [a - b for a, b in zip(me, see)], [a + b for a, b in zip(me, see)],
                        color=cmap(k), alpha=0.15, linewidth=0)
        # dashed same-color training line (past-100-episode mean); no band, no separate legend entry
        xt, mt, _ = _pooled_curve([g["train_curve"] for g in grp], MIN_SEEDS_CURVE)
        if xt:
            ax.plot(xt, mt, color=cmap(k), lw=1.3, ls="--")
    # run-2 eval reference: gray DOTTED (distinct from the dashed = training convention above)
    xr, mr, ser = _pooled_curve(_run2_eval_curves(), RUN2_MIN_SEEDS)
    if xr:
        ax.plot(xr, mr, color="0.25", lw=2.0, ls=":", label="run-2 rnd_elliptical eval ($\\beta$=0.01)")
        ax.fill_between(xr, [a - b for a, b in zip(mr, ser)], [a + b for a, b in zip(mr, ser)],
                        color="0.25", alpha=0.12, linewidth=0)
    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("Extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title("Train run 3.1.2: eval (solid) and training (dashed) reward, key cells at their best $\\beta$")
    ax.grid(linestyle=":", alpha=0.5)
    # keep the colored per-cell + reference legend, then append two style-only proxies (solid vs dashed)
    handles, labels = ax.get_legend_handles_labels()
    style = [Line2D([0], [0], color="0.35", lw=1.8, ls="-"),
             Line2D([0], [0], color="0.35", lw=1.4, ls="--")]
    style_labels = ["eval reward (solid)", "training reward, past 100 episodes (dashed)"]
    ax.legend(handles + style, labels + style_labels, fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots / "line_reward_curve.pdf", bbox_inches="tight")
    fig.savefig(plots / "line_reward_curve.png", bbox_inches="tight")
    plt.close(fig)


def config_stats(runs: list[dict]) -> dict:
    """All-36-config statistics: {(cell, beta): {"n":..,"mean":..,"se":..,"rho":..,"clip_hit":..}}.

    Before: run dicts; After: e.g. {(("none",1e-06,inf), 0.01): {"n":33,"mean":52.76,"se":6.18,...}}."""
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r["cell"], r["beta"])].append(r)
    out = {}
    for key, grp in by_cfg.items():
        R = [g["final_eval"] for g in grp]
        out[key] = {
            "n": len(R), "mean": statistics.mean(R),
            "se": statistics.stdev(R) / math.sqrt(len(R)) if len(R) > 1 else 0.0,
            "rho": statistics.mean([g["rho"] for g in grp if _finite(g.get("rho"))]),
            "clip_hit": statistics.mean([g["clip_hit"] for g in grp if _finite(g.get("clip_hit"))]),
        }
    return out


def full_grid_table(stats: dict, plots: Path) -> None:
    """The complete data: 12 rows (features x lambda x clip) x 3 beta columns, each cell Rbar +- SE (n).
    Row block order: raw before unit, small lambda first, no-clip before clip. The per-row best beta is
    bolded so the beta dimension is readable without collapsing it (no single beta dominates)."""
    lines = []
    for norm in ("none", "unit"):
        for lam in (1e-06, 0.0001, 0.01):
            for clip in (float("inf"), 5.0):
                cells = []
                means = []
                for beta in (0.001, 0.01, 0.1):
                    s = stats.get((( norm, lam, clip), beta))
                    if s is None:
                        cells.append("---")
                        means.append(-1e9)
                    else:
                        cells.append(f"{s['mean']:.1f} $\\pm$ {s['se']:.1f} ({s['n']})")
                        means.append(s["mean"])
                # bold the row's best-beta entry (the beta chosen by the best-beta table)
                b = means.index(max(means))
                cells[b] = f"\\textbf{{{cells[b]}}}"
                clip_s = "$\\infty$" if math.isinf(clip) else "5"
                lines.append(f"\\texttt{{{norm}}} & {_pow10(lam)} & {clip_s} & " + " & ".join(cells) + " \\\\")
            lines.append("\\addlinespace[2pt]")
    body = "\n".join(lines)
    (plots / "full_grid_table.tex").write_text(
        "% Train run 3.1.2 FULL grid: all 36 configurations, eval reward mean +- SE (n); per-row best beta bold.\n"
        "\\begin{tabular}{@{}l c c r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Features} & \\textbf{$\\lambda$} & \\textbf{clip} & "
        "\\textbf{$\\beta{=}10^{-3}$} & \\textbf{$\\beta{=}10^{-2}$} & \\textbf{$\\beta{=}10^{-1}$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def heatmap_facets(stats: dict, plots: Path) -> None:
    """Small-multiples heatmaps (the standard factorial-grid view): 2x2 panels (rows: clip off/on;
    columns: raw/unit features); each panel a lambda x beta grid colored by mean eval reward, with the
    number printed in each cell. One shared color scale across panels."""
    lams, betas = [1e-06, 0.0001, 0.01], [0.001, 0.01, 0.1]
    vmax = max(s["mean"] for s in stats.values())
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), dpi=150, sharex=True, sharey=True)
    for i, clip in enumerate((float("inf"), 5.0)):
        for j, norm in enumerate(("none", "unit")):
            ax = axes[i][j]
            grid = [[stats.get(((norm, lam, clip), b), {"mean": float("nan")})["mean"] for b in betas]
                    for lam in lams]
            # pcolormesh (not imshow) so the cells stay crisp in the vector PDF
            mesh = ax.pcolormesh([0, 1, 2, 3], [0, 1, 2, 3], grid, cmap="viridis", vmin=0, vmax=vmax)
            for a, lam in enumerate(lams):
                for b, beta in enumerate(betas):
                    s = stats.get(((norm, lam, clip), beta))
                    if s:
                        ax.text(b + 0.5, a + 0.5, f"{s['mean']:.1f}", ha="center", va="center", fontsize=9,
                                color="white" if s["mean"] < 0.6 * vmax else "black")
            ax.set_xticks([0.5, 1.5, 2.5], ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
            ax.set_yticks([0.5, 1.5, 2.5], ["$10^{-6}$", "$10^{-4}$", "$10^{-2}$"])
            clip_lbl = "no clip" if math.isinf(clip) else "clip 5"
            feat_lbl = "raw features" if norm == "none" else "unit-norm features"
            ax.set_title(f"{feat_lbl}, {clip_lbl}", fontsize=10)
            if i == 1:
                ax.set_xlabel("coefficient $\\beta$")
            if j == 0:
                ax.set_ylabel("ridge $\\lambda$")
    fig.colorbar(mesh, ax=axes, shrink=0.85, label="mean eval extrinsic reward")
    fig.suptitle("Train run 3.1.2: mean eval reward over the full grid", fontsize=11)
    fig.savefig(plots / "heatmap_facets.pdf", bbox_inches="tight")
    fig.savefig(plots / "heatmap_facets.png", bbox_inches="tight")
    plt.close(fig)


def rho_scatter(stats: dict, plots: Path) -> None:
    """The mechanism plot (interaction view): mean eval reward vs the MEASURED effective ridge ratio rho
    (log x), one point per configuration (all betas), split by clip (color) and features (marker). The
    ridge-ratio account predicts the no-clip points decline as rho grows; the clip-5 points sit collapsed
    below wherever the measured saturation is high."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=150)
    for clip, color, lbl in ((float("inf"), "#1f77b4", "no clip"), (5.0, "#d62728", "clip 5")):
        for norm, marker in (("none", "o"), ("unit", "s")):
            xs, ys, es = [], [], []
            for ((n_, lam, c_), beta), s in stats.items():
                if n_ == norm and c_ == clip:
                    xs.append(s["rho"]); ys.append(s["mean"]); es.append(s["se"])
            feat = "raw" if norm == "none" else "unit"
            ax.errorbar(xs, ys, yerr=es, fmt=marker, color=color, alpha=0.85, ms=6, lw=0, elinewidth=1,
                        capsize=2, label=f"{lbl}, {feat} features")
    ax.set_xscale("log")
    ax.set_xlabel("measured effective ridge ratio $\\rho = \\lambda d\\,/\\,\\overline{\\|\\varphi\\|^2}$")
    ax.set_ylabel("mean eval extrinsic reward ($\\pm$ SE)")
    ax.set_title("Reward vs the measured ridge ratio (one point per configuration; all $\\beta$)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "reward_vs_rho.pdf", bbox_inches="tight")
    fig.savefig(plots / "reward_vs_rho.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build all tables + figures and print the summary."""
    plots = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_DIR / "analysis" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    rows = cell_summaries(runs)
    stats = config_stats(runs)
    reward_table(rows, plots)
    full_grid_table(stats, plots)
    heatmap_facets(stats, plots)
    rho_scatter(stats, plots)
    curve_figure(runs, rows, plots)
    print(f"runs loaded (completed): {len(runs)}", file=sys.stderr)
    for r in rows:
        print(f"  {r['cell']} beta*={r['beta']:g} n={r['n']} mean={r['mean']:.2f} se={r['se']:.2f} "
              f"succ={r['success']:.2f} rho={r['rho']:.3g} clip_hit={r['clip_hit']:.3g}", file=sys.stderr)


if __name__ == "__main__":
    main()
