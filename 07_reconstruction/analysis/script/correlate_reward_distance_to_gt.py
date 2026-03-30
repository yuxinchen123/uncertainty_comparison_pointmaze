#!/usr/bin/env python3
"""
Correlate best-β mean eval extrinsic reward with summary distance_to_gt/* metrics.

- **Algorithm-level:** one point per algorithm = cell mean at (algorithm, β*) where β*
  maximizes mean reward; *n* = 7 here because three algorithms never log distances
  (action-conditioned RND variants).
- **Pooled runs:** all finished runs at (algorithm, β*); larger *n* but runs nest within
  algorithms (interpret as exploratory marginal association).

All analysis in ../analysis.md assumes finished W&B runs only (`state == finished`).

Outputs: ../plot/corr_reward_vs_distance_algorithm_means.png,
         ../plot/corr_reward_vs_distance_pooled_runs.png

Dependencies: pip install pandas matplotlib scipy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = ANALYSIS_ROOT / "data" / "combined_runs_index.csv"
PLOT_DIR = ANALYSIS_ROOT / "plot"

COL_ALGO = "config.algorithm"
COL_BETA = "config.beta"
COL_STATE = "state"
COL_REWARD = "summary.eval/mean_extrinsic_reward"

DIST_SPECS: list[tuple[str, str]] = [
    ("summary.distance_to_gt/min_c_l1_diff", "min_c_l1_diff"),
    ("summary.distance_to_gt/min_c_l2_diff", "min_c_l2_diff"),
    ("summary.distance_to_gt/min_c_l1_inv", "min_c_l1_inv"),
    ("summary.distance_to_gt/min_c_l2_inv", "min_c_l2_inv"),
    ("summary.distance_to_gt/normalized_l2", "normalized_l2"),
    ("summary.distance_to_gt/normalized_angle_rad", "normalized_angle_rad"),
]

SHORT_ALGO = {
    "gt_position_velocity": "gt_pv",
    "gt_position": "gt_p",
    "rnd_next_state": "rnd_ns",
    "rnd_state_action_next_state": "rnd_sans",
    "rnd_state": "rnd_s",
    "rnd_state_action": "rnd_sa",
    "rnd_elliptical": "rnd_e",
    "rnd_next_state_position_only": "rnd_nspo",
    "rnd_linear_next_state": "rnd_lns",
    "no_exploration": "no_exp",
}


def _abbr_legend_text() -> str:
    pairs = sorted(SHORT_ALGO.items(), key=lambda kv: kv[1])
    lines = ["abbr -> algorithm"] + [f"{abbr}: {algo}" for algo, abbr in pairs]
    return "\n".join(lines)


def _best_beta_cell_means(fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (best_cell_stats, best_pairs) with one row per algorithm."""
    agg: dict = {"mean_reward": (COL_REWARD, "mean")}
    for full, short in DIST_SPECS:
        agg[short] = (full, "mean")
    g = fin.groupby([COL_ALGO, COL_BETA], dropna=False).agg(**agg).reset_index()
    idx = g.groupby(COL_ALGO, sort=False)["mean_reward"].idxmax()
    best = g.loc[idx].sort_values("mean_reward", ascending=False).reset_index(drop=True)
    pairs = best[[COL_ALGO, COL_BETA]]
    return best, pairs


def _pooled_runs(fin: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    key = pd.merge(
        fin,
        pairs,
        on=[COL_ALGO, COL_BETA],
        how="inner",
    )
    return key


def _use_log_x(series: pd.Series) -> bool:
    s = series.dropna()
    if (s <= 0).any() or len(s) < 2:
        return False
    r = float(s.max() / s.min())
    return r > 50


def _add_abbr_key_below(fig: plt.Figure, abbr_text: str) -> None:
    """Place abbreviation key below subplot grid (assumes tight_layout left bottom margin)."""
    fig.text(
        0.02,
        0.018,
        abbr_text,
        transform=fig.transFigure,
        fontsize=10,
        family="monospace",
        va="bottom",
        ha="left",
        linespacing=1.28,
        bbox={
            "facecolor": "white",
            "alpha": 0.95,
            "edgecolor": "0.75",
            "linewidth": 0.9,
            "pad": 6,
        },
    )


def plot_grid(
    best: pd.DataFrame,
    pooled: pd.DataFrame,
    out_algo: Path,
    out_pool: Path,
) -> None:
    abbr_text = _abbr_legend_text()
    # Tighter gaps between panels only (figsize, markers, legend unchanged).
    _gap = {"pad": 0.25, "h_pad": 0.08, "w_pad": 0.06}
    fig, axes = plt.subplots(3, 2, figsize=(14, 17), dpi=150)
    for ax, (_, short), nice_title in zip(axes.flat, DIST_SPECS, [s for _, s in DIST_SPECS]):
        sub = best.dropna(subset=[short])
        if _use_log_x(sub[short]):
            ax.set_xscale("log")
        ax.scatter(
            sub[short],
            sub["mean_reward"],
            s=100,
            c="steelblue",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        for _, row in sub.iterrows():
            label = SHORT_ALGO.get(str(row[COL_ALGO]), str(row[COL_ALGO])[:10])
            ax.annotate(
                label,
                (row[short], row["mean_reward"]),
                fontsize=6,
                xytext=(4, 4),
                textcoords="offset points",
            )
        if len(sub) >= 2:
            r, p = stats.pearsonr(sub["mean_reward"], sub[short])
            rho, _ = stats.spearmanr(sub["mean_reward"], sub[short])
            ax.set_title(
                f"{nice_title}\n"
                f"Pearson r={r:.3f} (p={p:.3f}), Spearman ρ={rho:.3f}, n={len(sub)}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{nice_title}\n(insufficient points)", fontsize=10)
        ax.set_xlabel("distance (cell mean @ best β)")
        ax.set_ylabel("mean eval extrinsic reward")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_box_aspect(1)

    fig.suptitle(
        "Algorithm-level: one point = mean over finished runs in (algorithm, β*)\n"
        "β* maximizes mean reward; distance metrics missing for rnd_sa / rnd_sans / rnd_e",
        fontsize=11,
        y=0.97,
    )
    fig.tight_layout(rect=[0.05, 0.34, 0.99, 0.91], **_gap)
    _add_abbr_key_below(fig, abbr_text)
    out_algo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_algo, bbox_inches="tight", pad_inches=0.38)
    plt.close(fig)

    # Pooled
    fig, axes = plt.subplots(3, 2, figsize=(14, 17), dpi=150)
    for ax, (full, short), nice_title in zip(axes.flat, DIST_SPECS, [s for _, s in DIST_SPECS]):
        sub = pooled.dropna(subset=[full, COL_REWARD])
        if _use_log_x(sub[full]):
            ax.set_xscale("log")
        for _, chunk in sub.groupby(COL_ALGO):
            ax.scatter(
                chunk[full],
                chunk[COL_REWARD],
                s=22,
                alpha=0.55,
                edgecolors="none",
            )
        # Overlay algorithm means
        b = best.dropna(subset=[short])
        ax.scatter(
            b[short],
            b["mean_reward"],
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        for _, row in b.iterrows():
            label = SHORT_ALGO.get(str(row[COL_ALGO]), str(row[COL_ALGO])[:10])
            ax.annotate(
                label,
                (row[short], row["mean_reward"]),
                fontsize=6,
                xytext=(4, 4),
                textcoords="offset points",
            )
        if len(sub) >= 2:
            r, p = stats.pearsonr(sub[COL_REWARD], sub[full])
            rho, _ = stats.spearmanr(sub[COL_REWARD], sub[full])
            ax.set_title(
                f"{nice_title}\n"
                f"Pearson r={r:.3f} (p={p:.2e}), Spearman ρ={rho:.3f}, n={len(sub)}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{nice_title}\n(insufficient points)", fontsize=10)
        ax.set_xlabel("distance (per run @ best β)")
        ax.set_ylabel("eval extrinsic reward")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_box_aspect(1)

    fig.suptitle(
        "Pooled finished runs at (algorithm, β*) only; open circles = cell means\n"
        "(Runs are not independent across seeds — use with hierarchical caution.)",
        fontsize=11,
        y=0.97,
    )
    fig.tight_layout(rect=[0.05, 0.34, 0.99, 0.91], **_gap)
    _add_abbr_key_below(fig, abbr_text)
    fig.savefig(out_pool, bbox_inches="tight", pad_inches=0.38)
    plt.close(fig)


def best_with_distances_markdown(best: pd.DataFrame) -> str:
    cols = [short for _, short in DIST_SPECS]
    lines = [
        "",
        "| algorithm | eval/mean_extrinsic_reward | min_c_l1_diff | min_c_l2_diff | min_c_l1_inv | min_c_l2_inv | normalized_l2 | normalized_angle_rad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append("" if pd.isna(v) else f"{v:.2f}")
        lines.append(
            f"| {r[COL_ALGO]} | {r['mean_reward']:.2f} | "
            + " | ".join(vals)
            + " |"
        )
    return "\n".join(lines)


def stats_markdown(best: pd.DataFrame, pooled: pd.DataFrame) -> str:
    lines = [
        "",
        "| distance (summary key) | algo means: Pearson *r* | *p* | Spearman ρ | *n* algos | pooled runs: Pearson *r* | *p* | *n* runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for full, short in DIST_SPECS:
        sa = best.dropna(subset=[short])
        sp = pooled.dropna(subset=[full, COL_REWARD])
        ra = pa = na = rhoa = ""
        if len(sa) >= 2:
            r_a, p_a = stats.pearsonr(sa["mean_reward"], sa[short])
            rh_a, _ = stats.spearmanr(sa["mean_reward"], sa[short])
            ra, pa, rhoa, na = f"{r_a:.3f}", f"{p_a:.3f}", f"{rh_a:.3f}", str(len(sa))
        rp, pp, nr = "", "", ""
        if len(sp) >= 2:
            r_p, p_p = stats.pearsonr(sp[COL_REWARD], sp[full])
            rp, pp, nr = f"{r_p:.3f}", f"{p_p:.2e}", str(len(sp))
        lines.append(
            f"| `{full.replace('summary.', '')}` "
            f"| {ra} | {pa} | {rhoa} | {na} "
            f"| {rp} | {pp} | {nr} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--stats-md",
        action="store_true",
        help="Print correlation table (markdown) to stdout",
    )
    p.add_argument(
        "--best-distance-md",
        action="store_true",
        help="Print best-β table with six distance columns (markdown) to stdout",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Missing {args.csv}", file=sys.stderr)
        sys.exit(1)

    usecols = [COL_STATE, COL_ALGO, COL_BETA, COL_REWARD] + [f for f, _ in DIST_SPECS]
    df = pd.read_csv(args.csv, usecols=usecols, low_memory=False)
    fin = df[df[COL_STATE] == "finished"].copy()
    fin[COL_BETA] = pd.to_numeric(fin[COL_BETA], errors="coerce")
    fin[COL_REWARD] = pd.to_numeric(fin[COL_REWARD], errors="coerce")
    for full, _ in DIST_SPECS:
        fin[full] = pd.to_numeric(fin[full], errors="coerce")

    best, pairs = _best_beta_cell_means(fin)
    pooled = _pooled_runs(fin, pairs)

    out_a = PLOT_DIR / "corr_reward_vs_distance_algorithm_means.png"
    out_p = PLOT_DIR / "corr_reward_vs_distance_pooled_runs.png"
    plot_grid(best, pooled, out_a, out_p)
    print(f"Wrote {out_a}", file=sys.stderr)
    print(f"Wrote {out_p}", file=sys.stderr)

    if args.stats_md:
        print(stats_markdown(best, pooled))
    if args.best_distance_md:
        print(best_with_distances_markdown(best))


if __name__ == "__main__":
    main()
