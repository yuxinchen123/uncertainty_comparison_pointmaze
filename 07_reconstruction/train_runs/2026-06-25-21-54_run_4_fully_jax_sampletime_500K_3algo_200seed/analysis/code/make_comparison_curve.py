#!/usr/bin/env python3
"""
Train run 4 vs Train run 2 reward learning-curve comparison plot.

One color per algorithm (shared with run-2's plots via common.ALGO_COLOR). Three lines per algorithm:
  - run-2 eval extrinsic reward  -> DASHED, base color        (the torch reference)
  - run-4 eval extrinsic reward  -> SOLID,  base color        (the fully-JAX sample-time reproduction)
  - run-4 training-evaluation    -> SOLID,  DARKER base color (mean over past n_eval_episodes TRAINING
                                                               episodes, from train_history)
At each step s and algorithm a (over the n_a(s) seeds that reached step s):
    mean_a(s) = (1/n) sum_i R_i(s),  SE_a(s) = std_i R_i(s)/sqrt(n) (ddof=1),
each line truncated at the last step reached by >= MIN_SEEDS seeds (no tail averaged over few runs).

Robust to run-4 having no finished runs yet: then only run-2's dashed lines are drawn (a valid placeholder).
Output: <plots_dir>/comparison_reward_curve.{pdf,png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RUN2_DATA = Path("/p/rlprojects/RND/07_reconstruction/train_runs/2026-06-24-21-24_run_2_after_reorganization/data")
MIN_SEEDS = 30  # draw each line only up to the last step reached by >= this many seeds


def curve_stats(records, attr):
    """Per (algorithm, step): pooled mean / SE / n of the chosen reward curve across seeds (empty if none).

    attr = "eval_curve" (eval reward) or "train_curve" (training-eval reward over past n_eval_episodes).
    Before: long frame [algorithm, mode, seed, step, reward]. After: [algorithm, step, mean, se, n]."""
    long = C.curve_frame(records, attr)
    if long.empty:
        return pd.DataFrame(columns=["algorithm", "step", "mean", "se", "n"])
    s = (long.groupby(["algorithm", "step"])
         .agg(mean=("reward", "mean"), se=("reward", "sem"), n=("reward", "count")).reset_index())
    s["se"] = s["se"].fillna(0.0)
    return s


def _trunc(stats, algo):
    """One algorithm's per-step curve, kept only up to the last step with >= MIN_SEEDS seeds."""
    return stats[(stats["algorithm"] == algo) & (stats["n"] >= MIN_SEEDS)].sort_values("step")


def make_plot(run2_eval, run4_eval, run4_train, out_base):
    """Draw the three-line-per-algorithm comparison and save pdf+png."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
    # order algorithms by run-4 eval at its last >=MIN_SEEDS step (fallback run-2 eval) so the legend is ranked
    def last_mean(stats, algo):
        d = _trunc(stats, algo)
        return float(d.iloc[-1]["mean"]) if not d.empty else float("-inf")
    algos = sorted(C.ALGORITHMS, key=lambda a: max(last_mean(run4_eval, a), last_mean(run2_eval, a)), reverse=True)
    for algo in algos:
        color = C.ALGO_COLOR.get(algo, "0.5")
        beta = C.ALGO_BETA.get(algo, float("nan"))
        # run-2 eval: dashed, base color
        d2 = _trunc(run2_eval, algo)
        if not d2.empty:
            x, m, se = d2["step"].to_numpy() / 1e3, d2["mean"].to_numpy(), d2["se"].to_numpy()
            ax.plot(x, m, color=color, lw=1.7, ls="--")
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.10, linewidth=0)
        # run-4 eval: solid, base color (legend entry = the algorithm + its current run-4 seed count)
        d4 = _trunc(run4_eval, algo)
        if not d4.empty:
            n4 = int(d4["n"].max())  # run-4 seeds available for this algorithm (honest sample size in the legend)
            x, m, se = d4["step"].to_numpy() / 1e3, d4["mean"].to_numpy(), d4["se"].to_numpy()
            ax.plot(x, m, color=color, lw=2.0, ls="-", label=f"{algo} ($\\beta$={beta:g}, run-4 $n$={n4})")
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.16, linewidth=0)
        # run-4 training-eval: DASH-DOT, SAME base color (distinguished from the eval lines by line style, not
        # shade, which is hard to tell apart): run-2 eval dashed, run-4 eval solid, run-4 train-eval dash-dot
        dt = _trunc(run4_train, algo)
        if not dt.empty:
            x, m, se = dt["step"].to_numpy() / 1e3, dt["mean"].to_numpy(), dt["se"].to_numpy()
            ax.plot(x, m, color=color, lw=1.6, ls="-.")
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("Extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title(f"Train run 4 (fully-JAX, sample-time) vs Train run 2 (torch): reward over training\n"
                 f"(each line up to the last step with $\\geq${MIN_SEEDS} seeds; run-4 $n$ per algorithm in legend)")
    ax.grid(linestyle=":", alpha=0.5)
    # legend: algorithm colors + a line-style key (run-2 eval dashed / run-4 eval solid / run-4 train darker)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="0.3", ls="--", lw=1.7),
                Line2D([0], [0], color="0.3", ls="-", lw=2.0),
                Line2D([0], [0], color="0.3", ls="-.", lw=1.6)]
    labels += ["run-2 eval (torch)", "run-4 eval (JAX)", "run-4 train-eval (past 100 ep.)"]
    ax.legend(handles, labels, fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def build(run4_data, plots_dir):
    """Load run-2 + run-4 records, compute curve stats, draw the comparison (run-2 alone if run-4 empty)."""
    r2 = C.load_records(RUN2_DATA)
    r4 = C.load_records(run4_data)
    run2_eval = curve_stats(r2, "eval_curve")
    run4_eval = curve_stats(r4, "eval_curve")
    run4_train = curve_stats(r4, "train_curve")
    n4 = sum(1 for _ in r4)
    print(f"[make_comparison_curve] run-2 records={len(r2)} run-4 records={n4}", file=sys.stderr)
    make_plot(run2_eval, run4_eval, run4_train, plots_dir / "comparison_reward_curve")


def latest_sweep_dir(run_dir):
    """The most recent data/<sweep_id> dir holding run JSONs (sweep_id is timestamp-led, so reverse name-sort).
    Returns data/<sweep_id> (which load_records globs as <dir>/local/*.json). Falls back to data/ if none."""
    data = run_dir / "data"
    sweeps = sorted([d for d in data.glob("*") if (d / "local").is_dir()], reverse=True)
    return sweeps[0] if sweeps else data


def main():
    p = argparse.ArgumentParser(description="Train run 4 vs run 2 reward-curve comparison.")
    # default run-4 data = the latest sweep's data/<sweep_id> dir; pass a specific data/<sweep_id> to pin a sweep,
    # or several (pool reruns) by editing build(). The sweep-id layout keeps legacy sweeps out of the plot.
    run_dir = Path(__file__).resolve().parent.parent.parent
    p.add_argument("run4_data", nargs="?", type=Path, default=latest_sweep_dir(run_dir))
    p.add_argument("plots_dir", nargs="?", type=Path,
                   default=Path(__file__).resolve().parent.parent / "plots")
    # min seeds to draw a line up to a step. 30 (the convention) for the final plot; lower for an intermediate
    # view while run-4 is still filling in (the legend reports the actual run-4 n per algorithm either way).
    p.add_argument("--min_seeds", type=int, default=30)
    args = p.parse_args()
    global MIN_SEEDS
    MIN_SEEDS = args.min_seeds
    print(f"[make_comparison_curve] run-4 sweep dir = {args.run4_data} | MIN_SEEDS={MIN_SEEDS}", file=sys.stderr)
    build(args.run4_data, args.plots_dir)


if __name__ == "__main__":
    main()
