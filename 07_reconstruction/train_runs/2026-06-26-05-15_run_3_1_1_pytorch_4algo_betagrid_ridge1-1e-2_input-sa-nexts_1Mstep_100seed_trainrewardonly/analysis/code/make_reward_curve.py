#!/usr/bin/env python3
"""Train run 3.1.1 reward learning-curve plot: per-method TRAINING-episode reward over training, each method
at its single best configuration (the (beta, ridge, input) chosen in make_reward_table), with a light
standard-error band. The run-2 gt_position_velocity oracle curve (EVAL reward; run 2 logged eval only) is
overlaid as a dashed reference. At each step s and series, pooled over the n(s) seeds that reached step s:

    mean(s) = (1/n) sum_i R_i(s) ,  SE(s) = std_i R_i(s) / sqrt(n)  (ddof=1).

Each series is drawn only up to the last step with >= MIN_SEEDS seeds. Output:
<plots_dir>/line_reward_curve.{pdf,png}.
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
from make_reward_table import final_reward_frame, best_config_per_method  # noqa: E402


def _step_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per-step pooled mean / SE / n over seeds for one series' long frame [seed, step, reward]."""
    if long.empty:
        return pd.DataFrame(columns=["step", "mean", "se", "n"])
    s = long.groupby("step").agg(mean=("reward", "mean"), se=("reward", "sem"), n=("reward", "count")).reset_index()
    s["se"] = s["se"].fillna(0.0)
    return s.sort_values("step")


def _trunc(stats: pd.DataFrame) -> pd.DataFrame:
    """Keep only steps reached by >= MIN_SEEDS seeds (drops the unreliable high-step tail)."""
    return stats[stats["n"] >= C.MIN_SEEDS].sort_values("step")


def _row_config_id(row) -> str:
    """Rebuild the canonical config_id string from a best-config row (its ridge/input are already strings),
    matching common.config_id_str so it equals the config_id stored on each training-curve row."""
    return f"{row['method']}|beta={row['beta']:g}|ridge={row['ridge']}|input={row['input']}"


def method_curves(records) -> dict:
    """For each method, the per-step training-reward stats of its BEST config (truncated to >=MIN_SEEDS).
    Returns {method_id: (best_meta_row, truncated_stats_df)} for methods that have a qualifying config."""
    best = best_config_per_method(final_reward_frame(records))
    long = C.train_curve_frame(records)
    out = {}
    for _, row in best.iterrows():
        # filter the long training-curve frame to this method's best config (by canonical config_id), then per-step stats
        sub = long[long["config_id"] == _row_config_id(row)]
        stats = _trunc(_step_stats(sub[["seed", "step", "reward"]]))
        if not stats.empty:
            out[row["method"]] = (row, stats)
    return out


def make_line(curves: dict, run2_stats: dict, out_base: Path) -> None:
    """Solid line per method (best config, training reward) + dashed run-2 references (EVAL reward):
    the gt oracle, rnd_state, and the run-2 batch elliptical -- 4 + 3 = 7 lines."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    # legend/draw order: methods by final training reward descending
    ordered = sorted(curves.items(), key=lambda kv: kv[1][1].iloc[-1]["mean"], reverse=True)
    for mid, (meta, stats) in ordered:
        color = C.ALGO_COLOR.get(mid, "0.5")
        x, m, se = stats["step"].to_numpy() / 1e6, stats["mean"].to_numpy(), stats["se"].to_numpy()
        # label: method + its chosen beta (and ridge for elliptical)
        lbl = C.METHOD_LABEL[mid] + f" ($\\beta$={meta['beta']:g}"
        lbl += ")" if mid == "A4_rnd_next_state" else f", $\\lambda$={float(meta['ridge']):g})"
        ax.plot(x, m, color=color, lw=1.8, ls="-", label=lbl)
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.16, linewidth=0)
    # run-2 reference curves (EVAL reward) as dashed lines in their stable colors
    for (algo, beta) in C.RUN2_OVERLAYS:
        stats = run2_stats.get(algo)
        if stats is None or stats.empty:
            continue
        color = C.ALGO_COLOR.get(algo, "0.5")
        x, m, se = stats["step"].to_numpy() / 1e6, stats["mean"].to_numpy(), stats["se"].to_numpy()
        ax.plot(x, m, color=color, lw=1.8, ls="--", label=f"{algo} ($\\beta$={beta:g}, run-2 eval)")
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("Extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title(f"Train run 3.1.1: training-episode reward per method at its best config "
                 f"(each line up to $n\\geq{C.MIN_SEEDS}$ seeds)")
    ax.grid(linestyle=":", alpha=0.5)
    # legend: per-series colors, plus a solid/dashed key (solid=run-3.1.1 train, dashed=run-2 eval refs)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="0.3", ls="-", lw=1.8), Line2D([0], [0], color="0.3", ls="--", lw=1.8)]
    labels += ["run-3.1.1 train reward", "run-2 eval reward"]
    ax.legend(handles, labels, fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def build(data_dir: Path, plots_dir: Path) -> None:
    """Load run-3.1.1 records + run-2 gt records, compute curves, save the line plot (no-op if nothing ready)."""
    records = C.load_records(data_dir)
    curves = method_curves(records)
    if not curves:
        print(f"[make_reward_curve] no method has a config with >= {C.MIN_SEEDS} seeds yet; skipping", file=sys.stderr)
        return
    # run-2 eval-reward references (gt oracle + rnd_state + batch elliptical), skipped if data absent
    run2_stats = {}
    if (C.RUN2_DATA_DIR / "local").is_dir():
        run2_records = C.load_records(C.RUN2_DATA_DIR)
        for algo, beta in C.RUN2_OVERLAYS:
            run2_stats[algo] = _trunc(_step_stats(C.eval_curve_frame(run2_records, algo, beta)))
    make_line(curves, run2_stats, plots_dir / "line_reward_curve")


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 3.1.1: per-method best-config training-reward curve.")
    p.add_argument("data_dir", type=Path, help="run-3.1.1 sweep data dir (holds the local/ subdir), e.g. data/<sweep_id>")
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR)
    args = p.parse_args()
    build(args.data_dir, args.plots_dir)


if __name__ == "__main__":
    main()
