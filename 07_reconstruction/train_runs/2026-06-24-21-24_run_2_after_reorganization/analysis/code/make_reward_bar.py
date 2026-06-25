#!/usr/bin/env python3
"""
Train run 2 reward bar plot (pooled across both logging modes).

Horizontal bar chart of the pooled final eval extrinsic reward Rbar per algorithm
with standard-error bars, best bar at the top, one stable color per algorithm.
Rbar and SE are pooled over both W&B logging modes and all seeds:

    Rbar = (1/n) sum_i R_i ,  SE = s / sqrt(n)   (s = sample std, ddof=1).

Output: <plots_dir>/bar_final_reward.{pdf,png}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_reward_table import reward_summary  # noqa: E402


def make_bar(summary, out_base: Path) -> None:
    """Horizontal bar chart of pooled Rbar +- SE, best at top, colored per common.ALGO_COLOR."""
    # summary is Rbar-descending; reverse so the largest sits at the top of a barh
    b = summary.iloc[::-1].reset_index(drop=True)
    labels = [f"{r['algorithm']}  ($\\beta$={r['beta']:g})" for _, r in b.iterrows()]
    colors = [C.ALGO_COLOR.get(r["algorithm"], "0.5") for _, r in b.iterrows()]

    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    y = range(len(b))
    # draw each algorithm's pooled mean reward with a standard-error whisker
    ax.barh(y, b["Rbar"], xerr=b["SE"], capsize=3, color=colors, edgecolor="none", alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Final eval extrinsic reward (mean $\\pm$ standard error over seeds)")
    ax.set_title("Train run 2: final eval reward per algorithm")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def build(data_dir: Path, plots_dir: Path) -> None:
    """Load records, compute the pooled reward summary, and save the bar plot (no-op if no reward data)."""
    records = C.load_records(data_dir)
    summary = reward_summary(records)
    if summary.empty:
        print("[make_reward_bar] no final-reward data found; skipping bar plot", file=sys.stderr)
        return
    make_bar(summary, plots_dir / "bar_final_reward")


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 2: pooled reward bar plot.")
    p.add_argument("data_dir", nargs="?", type=Path, default=C.DEFAULT_DATA_DIR)
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR)
    args = p.parse_args()
    build(args.data_dir, args.plots_dir)


if __name__ == "__main__":
    main()
