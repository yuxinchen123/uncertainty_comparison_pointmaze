#!/usr/bin/env python3
"""
Bar chart for the “best β per algorithm” table (mean eval extrinsic reward ± SE).

All analysis in ../analysis.md assumes finished W&B runs only (`state == finished`); uses
the same aggregation as `grouped_finished_results.py`.

Output: ../plot/best_per_algorithm_extrinsic_reward.png

Dependencies: pip install pandas matplotlib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
PLOT_DIR = ANALYSIS_ROOT / "plot"
DEFAULT_OUT = PLOT_DIR / "best_per_algorithm_extrinsic_reward.png"

# Import after path setup
sys.path.insert(0, str(SCRIPT_DIR))
import grouped_finished_results as gr  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Plot best-β-per-algorithm rewards ± SE.")
    p.add_argument("--csv", type=Path, default=gr.DEFAULT_CSV)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Missing {args.csv}; run combine_sweep_runs.py first.", file=sys.stderr)
        sys.exit(1)

    need = [
        gr.COL_STATE,
        gr.COL_ALGO,
        gr.COL_BETA,
        gr.COL_EVAL_EXT,
        gr.COL_STEP,
    ]
    df = pd.read_csv(args.csv, usecols=need, low_memory=False)
    b = gr.best_per_algorithm(gr.build_table(df))

    labels = [
        f"{row[gr.COL_ALGO]} (β={row[gr.COL_BETA]:.4g})" for _, row in b.iterrows()
    ]
    means = b["mean_eval_ext"].to_numpy()
    se = b["se_eval_ext"].to_numpy()
    # Bars top-to-bottom = best-first (matplotlib y first row at bottom, so reverse)
    labels = list(reversed(labels))
    means = means[::-1]
    se = se[::-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    y_pos = range(len(labels))
    ax.barh(y_pos, means, xerr=se, capsize=3, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("eval / mean_extrinsic_reward (mean ± SE across runs)")
    ax.set_title("Best config.β per algorithm (finished runs), rank by mean reward")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
