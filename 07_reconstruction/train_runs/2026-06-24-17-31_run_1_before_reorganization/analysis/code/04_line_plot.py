#!/usr/bin/env python3
"""
Learning-curve line plot for "train run 1": one line per algorithm (its best-beta
group), eval extrinsic reward vs training step, shaded band = standard error over
seeds.

Input:  data/y24vyh06/history_best_beta.csv  (long: algorithm, beta, a_seed, run_id, step, eval_reward)
Output: plots/line_best_beta_eval_curve.{pdf,png}

At each logged step s and algorithm a (best-beta group of n_a seeds):
  mean_a(s) = (1/n) sum_i R_i(s) ,  SE_a(s) = std_i R_i(s) / sqrt(n)  (ddof=1).
Legend order = final-step mean reward, descending (matches the results table).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def curve_stats(hist: pd.DataFrame) -> pd.DataFrame:
    """Per (algorithm, step): mean, SEM, n of eval reward across the best-beta group's seeds."""
    s = (
        hist.groupby(["algorithm", "step"])
        .agg(mean=("eval_reward", "mean"), se=("eval_reward", "sem"), n=("eval_reward", "count"))
        .reset_index()
    )
    s["se"] = s["se"].fillna(0.0)
    return s


def final_order(stats: pd.DataFrame) -> list[str]:
    """Algorithms ordered by their mean reward at the largest step, descending (legend order)."""
    last_step = stats["step"].max()
    fin = stats[stats["step"] == last_step].sort_values("mean", ascending=False)
    return fin["algorithm"].tolist()


def make_line(stats: pd.DataFrame, best_beta: dict[str, float], out_base: Path) -> None:
    """One mean curve per algorithm with a shaded +-SE band, colored per common.ALGO_COLOR."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    # draw in legend order so the legend reads best-to-worst at the final step
    for algo in final_order(stats):
        d = stats[stats["algorithm"] == algo].sort_values("step")
        x = d["step"].to_numpy() / 1e6  # before: env steps 0..2e6  -> after: millions of steps
        m = d["mean"].to_numpy()
        se = d["se"].to_numpy()
        color = C.ALGO_COLOR[algo]
        label = f"{algo} ($\\beta$={best_beta[algo]:g})"
        ax.plot(x, m, color=color, lw=1.8, label=label)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("Eval extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title("Best intrinsic coefficient per algorithm: eval reward over training (sweep y24vyh06)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=1)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def main() -> None:
    hist = pd.read_csv(C.SWEEP_DIR / "history_best_beta.csv", low_memory=False)
    # map algorithm -> its best beta (constant within the file) for legend labels
    best_beta = hist.groupby("algorithm")["beta"].first().to_dict()
    stats = curve_stats(hist)
    stats.to_csv(C.SWEEP_DIR / "curve_stats.csv", index=False)
    make_line(stats, best_beta, C.PLOTS_DIR / "line_best_beta_eval_curve")
    # report the seed count range per algorithm at the final step
    last = stats[stats["step"] == stats["step"].max()]
    print("seeds per algorithm at final step:", file=sys.stderr)
    print(last[["algorithm", "n", "mean", "se"]].to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
