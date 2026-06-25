#!/usr/bin/env python3
"""
Train run 2 reward learning-curve plot (pooled across both logging modes).

One line per algorithm: mean eval/mean_extrinsic_reward vs training step, pooled
over both W&B logging modes and all seeds, with a light standard-error band. At
each logged step s and algorithm a (over the n_a(s) runs that reached step s):

    mean_a(s) = (1/n) sum_i R_i(s) ,  SE_a(s) = std_i R_i(s) / sqrt(n)  (ddof=1).

Legend order = mean reward at the final step, descending (matches the reward table).

Output: <plots_dir>/line_reward_curve.{pdf,png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def curve_stats(records: list[C.RunRecord], attr: str = "eval_curve") -> pd.DataFrame:
    """Per (algorithm, step): pooled mean / SE / n of the chosen reward curve across all seeds.

    attr = "eval_curve" (eval reward, solid line) or "train_curve" (training reward over the past 100
    episodes, dashed line). Returns empty if no run carries that curve (e.g. train on pre-fix runs).
    Before: long curve frame [algorithm, mode, seed, step, reward].
    After:  [algorithm, step, mean, se, n], one row per (algorithm, step).
    """
    long = C.curve_frame(records, attr)
    if long.empty:
        return pd.DataFrame(columns=["algorithm", "step", "mean", "se", "n"])
    # collapse modes + seeds: at each step, mean / standard error / count of reward per algorithm
    s = (
        long.groupby(["algorithm", "step"])
        .agg(mean=("reward", "mean"), se=("reward", "sem"), n=("reward", "count"))
        .reset_index()
    )
    s["se"] = s["se"].fillna(0.0)
    return s


# Plot each algorithm's line only up to the largest step reached by at least this many seeds, so the
# high-step tail (where only a few runs have data) is not shown as an unreliable mean.
MIN_SEEDS = 30


def _trunc(stats: pd.DataFrame, algo: str) -> pd.DataFrame:
    """One algorithm's per-step curve, kept only up to the last step with >= MIN_SEEDS seeds.

    Before: rows for `algo` at steps [50k (n=140), ..., 700k (n=35), 750k (n=12)].
    After:  same rows truncated at the last step with n>=30 -> the 750k (n=12) tail is dropped.
    """
    return stats[(stats["algorithm"] == algo) & (stats["n"] >= MIN_SEEDS)].sort_values("step")


def final_order(stats: pd.DataFrame) -> list[str]:
    """Algorithms ordered by mean reward at their own last >=MIN_SEEDS step, descending (legend order)."""
    # rank each algorithm by its reward at the last step where it still has MIN_SEEDS seeds
    ranked = []
    for algo in stats["algorithm"].unique():
        d = _trunc(stats, algo)
        if not d.empty:
            ranked.append((algo, float(d.iloc[-1]["mean"])))
    ranked.sort(key=lambda am: am[1], reverse=True)
    return [a for a, _ in ranked]


def make_line(eval_stats: pd.DataFrame, train_stats: pd.DataFrame, out_base: Path) -> None:
    """Per algorithm: eval reward as a SOLID line and training reward (past 100 episodes) as a DASHED line
    in the SAME color, each truncated at >=MIN_SEEDS seeds, with shaded +-SE bands. The dashed lines are
    only drawn for algorithms whose runs carry train_curve data (older runs have none)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    has_train = not train_stats.empty
    # draw in legend order (best-to-worst eval at the final >=MIN_SEEDS step)
    for algo in final_order(eval_stats):
        color = C.ALGO_COLOR.get(algo, "0.5")
        # eval: solid line + band, labeled by algorithm
        de = _trunc(eval_stats, algo)
        if not de.empty:
            x, m, se = de["step"].to_numpy() / 1e6, de["mean"].to_numpy(), de["se"].to_numpy()
            ax.plot(x, m, color=color, lw=1.8, ls="-", label=f"{algo} ($\\beta$={C.ALGO_BETA.get(algo, float('nan')):g})")
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.16, linewidth=0)
        # train (past 100 episodes): dashed line + lighter band, same color, no separate legend entry
        dt = _trunc(train_stats, algo)
        if not dt.empty:
            xt, mt, st = dt["step"].to_numpy() / 1e6, dt["mean"].to_numpy(), dt["se"].to_numpy()
            ax.plot(xt, mt, color=color, lw=1.5, ls="--")
            ax.fill_between(xt, mt - st, mt + st, color=color, alpha=0.09, linewidth=0)

    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("Extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title("Train run 2: reward over training" + (", solid = eval / dashed = train (past 100 episodes)" if has_train else f" (eval; each line up to $n\\geq{MIN_SEEDS}$ seeds)"))
    ax.grid(linestyle=":", alpha=0.5)
    # legend: algorithm colors, plus a solid/dashed style key when the training lines are present
    handles, labels = ax.get_legend_handles_labels()
    if has_train:
        handles += [Line2D([0], [0], color="0.3", ls="-", lw=1.8), Line2D([0], [0], color="0.3", ls="--", lw=1.5)]
        labels += ["eval reward", "train reward (past 100 ep.)"]
    ax.legend(handles, labels, fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def build(data_dir: Path, plots_dir: Path) -> None:
    """Load records, compute pooled per-step curve stats, and save the line plot (no-op if no curves)."""
    records = C.load_records(data_dir)
    eval_stats = curve_stats(records, "eval_curve")
    # skip until at least one (algorithm, step) has been reached by >= MIN_SEEDS seeds (else nothing to draw)
    if eval_stats.empty or eval_stats["n"].max() < MIN_SEEDS:
        print(f"[make_reward_curve] no step reached by >= {MIN_SEEDS} seeds yet; skipping line plot", file=sys.stderr)
        return
    # training curve (dashed); empty until runs save train_history, in which case only eval lines are drawn
    train_stats = curve_stats(records, "train_curve")
    make_line(eval_stats, train_stats, plots_dir / "line_reward_curve")


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 2: pooled reward learning-curve plot.")
    p.add_argument("data_dir", nargs="?", type=Path, default=C.DEFAULT_DATA_DIR)
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR)
    args = p.parse_args()
    build(args.data_dir, args.plots_dir)


if __name__ == "__main__":
    main()
