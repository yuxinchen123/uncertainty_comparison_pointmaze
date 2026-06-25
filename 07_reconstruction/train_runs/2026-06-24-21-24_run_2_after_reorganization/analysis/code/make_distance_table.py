#!/usr/bin/env python3
"""
Train run 2 distance-to-ground-truth table (pooled over all finished seeds; single local mode, no wandb).

For each algorithm that logs distance data, the mean of the six
distance_to_gt/* metrics (lower is better) over all finished
seeds, using each run's final (largest-step) distance snapshot:

    Dbar_m(a) = (1/n) sum_i D_{m,i}   for metric m, algorithm a.

gt_position_velocity is the oracle field, so its distances are ~0. Algorithms
whose runs carry no distance data are omitted (action-conditioned bonus fields).

Output: a LaTeX tabular fragment at <plots_dir>/distance_table.tex (models in
rows, the six metrics in columns; per column the smallest mean is bold and the
second smallest underlined; lower distance is better). Also printed to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# Sort rows by this metric ascending (lower-is-better -> best first), matching the
# best-first ordering of the reward table. Only this column gets a sort arrow.
SORT_METRIC = "distance_to_gt/normalized_l2"


def distance_summary(records: list[C.RunRecord]) -> pd.DataFrame:
    """Per-algorithm pooled mean of the six distance metrics + n, for algorithms with distance data.

    Before: per-run frame with the 6 distance_to_gt/* columns (None where a run lacks distance data).
    After:  [algorithm, <6 metric means>, n_dist], one row per algorithm that has >=1 distance run,
            sorted ascending by SORT_METRIC (best first).
    """
    df = C.records_to_frame(records)
    # a run counts as "has distance" if it carries at least the sort metric's final value
    has_dist = df[df[SORT_METRIC].notna()]
    if has_dist.empty:
        return pd.DataFrame(columns=["algorithm", *C.DISTANCE_METRICS, "n_dist"])
    # pooled mean per metric per algorithm, plus the per-algorithm distance-run count
    agg = {m: (m, "mean") for m in C.DISTANCE_METRICS}
    agg["n_dist"] = (SORT_METRIC, "count")
    g = has_dist.groupby("algorithm").agg(**agg).reset_index()
    return g.sort_values(SORT_METRIC, ascending=True).reset_index(drop=True)


def distance_tabular(summary: pd.DataFrame) -> str:
    """LaTeX tabular block for the distance table (per-column smallest bold, second smallest underlined)."""
    # per metric column, find which algorithm rows hold the best (smallest) / second-smallest mean
    marks = {m: C.rank_marks(summary[m].tolist(), lower_is_better=True) for m in C.DISTANCE_METRICS}

    # header: algorithm + six metric names. Each long metric name is FOLDED onto TWO right-aligned lines
    # (a \shortstack), split at the underscore that best balances the two line lengths -- so the identifier
    # wraps within its column instead of overrunning it, but does not explode into one token per line.
    # before: short="min_c_l1_diff"        -> after: \shortstack[r]{\texttt{min\_c\_}\\ \texttt{l1\_diff}}
    # before: short="normalized_angle_rad" -> after: \shortstack[r]{\texttt{normalized\_}\\ \texttt{angle\_rad}}
    header_cells = ["\\textbf{Algorithm}"]
    for m in C.DISTANCE_METRICS:
        toks = C.DISTANCE_SHORT[m].split("_")
        # choose the split index k minimizing the longer of the two resulting lines (balanced fold)
        k = min(range(1, len(toks)), key=lambda j: max(len("_".join(toks[:j])), len("_".join(toks[j:]))))
        line1 = ("_".join(toks[:k]) + "_").replace("_", "\\_")  # trailing "_" marks the mid-name break
        line2 = "_".join(toks[k:]).replace("_", "\\_")
        arrow = " ($\\uparrow$)" if m == SORT_METRIC else ""
        stack = f"\\shortstack[r]{{\\texttt{{{line1}}}\\\\ \\texttt{{{line2}}}{arrow}}}"
        header_cells.append(f"\\textbf{{{stack}}}")
    header = " & ".join(header_cells) + " \\\\"

    rows = []
    for i, r in summary.iterrows():
        # one row: algorithm name then each metric mean, bold/underline per its column ranking
        cells = [f"\\texttt{{{C.tex_escape_algo(r['algorithm'])}}}"]
        for m in C.DISTANCE_METRICS:
            val = r[m]
            txt = "--" if pd.isna(val) else f"{val:.3g}"
            best, second = marks[m]
            if i in best:
                txt = f"\\textbf{{{txt}}}"
            elif i in second:
                txt = f"\\underline{{{txt}}}"
            cells.append(txt)
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)

    colspec = "@{}>{\\raggedright\\arraybackslash}p{3.0cm} " + " ".join(["r"] * len(C.DISTANCE_METRICS)) + "@{}"
    return (
        "% Train run 2 distance-to-ground-truth table (pooled over all finished seeds; single local mode, no wandb).\n"
        "% Lower distance is better; rows sorted ascending by normalized_l2; per column best bold, second underlined.\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def build(data_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Load records, build the pooled distance table, write distance_table.tex, return the summary."""
    records = C.load_records(data_dir)
    summary = distance_summary(records)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        # still write a marker fragment so downstream LaTeX includes never fail on missing file
        (plots_dir / "distance_table.tex").write_text("% Train run 2: no distance data found yet.\n")
        print("[make_distance_table] no distance data found; wrote placeholder fragment", file=sys.stderr)
        return summary
    (plots_dir / "distance_table.tex").write_text(distance_tabular(summary) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 2: pooled distance table (LaTeX fragment).")
    p.add_argument("data_dir", nargs="?", type=Path, default=C.DEFAULT_DATA_DIR)
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR)
    args = p.parse_args()

    summary = build(args.data_dir, args.plots_dir)
    # console summary so the table is visible without opening the .tex file
    print((args.plots_dir / "distance_table.tex").read_text())
    if not summary.empty:
        print("pooled distance summary:", file=sys.stderr)
        print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
