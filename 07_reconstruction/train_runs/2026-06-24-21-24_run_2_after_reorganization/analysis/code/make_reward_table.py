#!/usr/bin/env python3
"""
Train run 2 reward table (pooled over all finished seeds; single local logging mode, no wandb).

Run 2 has one logging mode ("local", use_wandb=False), so reward is pooled over all finished
seeds. For each algorithm we report its fixed best coefficient beta-star, the mean final eval
extrinsic reward Rbar, its standard error SE, and the seed count n:

    Rbar = (1/n) sum_i R_i ,  SE = s / sqrt(n)   (s = sample std, ddof=1)

where R_i is run i's final (largest-step) eval/mean_extrinsic_reward.

Output: a LaTeX tabular fragment at <plots_dir>/reward_table.tex (models in rows,
sorted by Rbar descending; best Rbar bold, second underlined; higher reward is
better). The fragment is also printed to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def reward_summary(records: list[C.RunRecord]) -> pd.DataFrame:
    """Per-algorithm pooled mean / SE / n of the final eval reward, sorted by mean descending.

    Before: records pooled over all finished seeds (single local mode), each with a final_reward.
    After:  DataFrame [algorithm, beta, Rbar, SE, n], one row per algorithm with >=1 reward,
            sorted by Rbar descending (best first).
    """
    df = C.records_to_frame(records)
    # drop runs without a final reward (empty / partial eval_history) before aggregating
    df = df[df["final_reward"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["algorithm", "beta", "Rbar", "SE", "n"])
    # group by algorithm: mean / standard-error / count of final reward, pooled over all finished seeds
    g = (
        df.groupby("algorithm")
        .agg(beta=("beta", "first"), Rbar=("final_reward", "mean"), SE=("final_reward", "sem"), n=("final_reward", "count"))
        .reset_index()
    )
    # sem is NaN when n == 1; report 0 standard error in that case
    g["SE"] = g["SE"].fillna(0.0)
    return g.sort_values("Rbar", ascending=False).reset_index(drop=True)


def reward_tabular(summary: pd.DataFrame) -> str:
    """LaTeX tabular block for the pooled reward table (best Rbar bold, second underlined).

    Matches the train-run-1 column layout: Algorithm, beta-star, Rbar (sort column,
    descending -> down arrow), SE, n. Higher reward is better.
    """
    # mark best (largest) and second-best mean reward across the algorithm rows
    best, second = C.rank_marks(summary["Rbar"].tolist(), lower_is_better=False)
    rows = []
    for i, r in summary.iterrows():
        # format one row; wrap the winning / runner-up Rbar cell in bold / underline
        rbar = f"{r['Rbar']:.2f}"
        if i in best:
            rbar = f"\\textbf{{{rbar}}}"
        elif i in second:
            rbar = f"\\underline{{{rbar}}}"
        rows.append(
            f"\\texttt{{{C.tex_escape_algo(r['algorithm'])}}} & ${r['beta']:g}$ & {rbar} & {r['SE']:.2f} & {int(r['n'])} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Train run 2 reward table (pooled over all finished seeds; single local logging mode, no wandb).\n"
        "% Higher reward is better; rows sorted by Rbar descending; best bold, second-best underlined.\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.6cm} r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Algorithm} & \\textbf{$\\beta^\\star$} & \\textbf{$\\bar{R}$ ($\\downarrow$)} & \\textbf{SE} & \\textbf{$n$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def build(data_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Load records, build the pooled reward table, write reward_table.tex, return the summary frame."""
    records = C.load_records(data_dir)
    summary = reward_summary(records)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tabular = reward_tabular(summary)
    (plots_dir / "reward_table.tex").write_text(tabular + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 2: pooled reward table (LaTeX fragment).")
    p.add_argument("data_dir", nargs="?", type=Path, default=C.DEFAULT_DATA_DIR, help="run-2 data dir (holds the mode subdirs)")
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR, help="output plots dir")
    args = p.parse_args()

    summary = build(args.data_dir, args.plots_dir)
    # console summary so the table is visible without opening the .tex file
    print((args.plots_dir / "reward_table.tex").read_text())
    print("pooled reward summary:", file=sys.stderr)
    print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
