#!/usr/bin/env python3
"""
Aggregate finished runs of sweep y24vyh06 and produce, for "train run 1":
  - data/best_beta_per_algorithm.csv : best beta per algorithm, mean/SEM/n of eval reward
  - data/all_groups.csv              : every (algorithm, beta) group's mean/SEM/n
  - data/verify_train_level.csv      : algorithm-level mean train reward (matches W&B dashboard)
  - plots/bar_best_beta_eval_reward.{pdf,png} : bar chart, best beta per algorithm, mean +- SE

It also splices the best-beta results table directly into ../../main.tex, between the
markers `% >>> AUTO-GENERATED TABLE START: trainrun1-best` / `... END: ...` (the
generate-latex-table convention): only the tabular block is replaced, the caption,
label, and surrounding prose stay hand-edited. No separate .tex fragment is written.

Filter: state == "finished" only (crashed / running excluded).
Group:  by (algorithm, beta); keep the beta with the largest mean eval reward per algorithm.

Mean +- standard error of the mean over seeds:
  mean = (1/n) sum_i R_i ,  SE = s / sqrt(n)  (s = sample std, ddof=1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load_finished(slim_csv: Path) -> pd.DataFrame:
    """Load the slim CSV, keep finished runs, coerce beta and the two reward columns to numeric."""
    df = pd.read_csv(slim_csv, low_memory=False)
    fin = df[df["state"] == "finished"].copy()
    fin["beta"] = pd.to_numeric(fin["config.beta"], errors="coerce")
    fin["eval_ext"] = pd.to_numeric(fin[f"summary.{C.METRIC}"], errors="coerce")
    fin["train_ext"] = pd.to_numeric(fin["summary.train/mean_extrinsic_reward"], errors="coerce")
    return fin


def group_by_algo_beta(fin: pd.DataFrame) -> pd.DataFrame:
    """Mean / SEM / n of eval reward for each (algorithm, beta) group."""
    g = (
        fin.groupby(["config.algorithm", "beta"])
        .agg(mean_eval=("eval_ext", "mean"), se_eval=("eval_ext", "sem"), n=("eval_ext", "count"))
        .reset_index()
    )
    g["se_eval"] = g["se_eval"].fillna(0.0)
    return g


def best_per_algorithm(g: pd.DataFrame) -> pd.DataFrame:
    """One row per algorithm: the beta with the largest mean eval reward; sorted reward-desc."""
    best = g.loc[g.groupby("config.algorithm")["mean_eval"].idxmax()].copy()
    return best.sort_values("mean_eval", ascending=False).reset_index(drop=True)


def verify_train_level(fin: pd.DataFrame) -> pd.DataFrame:
    """Algorithm-level mean train reward + run count — the quantity shown on the W&B dashboard."""
    v = (
        fin.groupby("config.algorithm")
        .agg(n_runs=("train_ext", "count"), train_mean=("train_ext", "mean"))
        .reset_index()
        .sort_values("train_mean", ascending=False)
        .reset_index(drop=True)
    )
    return v


def make_bar(best: pd.DataFrame, out_base: Path) -> None:
    """Horizontal bar chart of best-beta mean eval reward +- SE, best at top, per-algorithm colors."""
    # best is reward-desc; reverse so the top bar (first row) is the largest in a barh
    b = best.iloc[::-1].reset_index(drop=True)
    labels = [f"{r['config.algorithm']}  ($\\beta$={r['beta']:g})" for _, r in b.iterrows()]
    colors = [C.ALGO_COLOR[r["config.algorithm"]] for _, r in b.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    y = range(len(b))
    ax.barh(y, b["mean_eval"], xerr=b["se_eval"], capsize=3, color=colors, edgecolor="none", alpha=0.9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Final eval extrinsic reward at 2M steps (mean $\\pm$ standard error over seeds)")
    ax.set_title("Best intrinsic coefficient per algorithm (finished runs, sweep y24vyh06)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base.with_suffix('.pdf')} and .png", file=sys.stderr)


def results_tabular(best: pd.DataFrame) -> str:
    """Full LaTeX tabular block for the best-beta results table.

    Models in rows, sorted by mean eval reward descending; best mean bold, second
    underlined (analysis-convention rule). Returned text replaces the body between
    the trainrun1-best AUTO-GENERATED markers in main.tex.
    """
    # rank for marking: best (largest mean) bold, second underlined
    order = best["mean_eval"].rank(ascending=False, method="min")
    rows = []
    for i, r in best.iterrows():
        algo = r["config.algorithm"].replace("_", "\\_")
        beta = f"{r['beta']:g}"
        mean = f"{r['mean_eval']:.2f}"
        se = f"{r['se_eval']:.2f}"
        n = int(r["n"])
        if order.iloc[i] == 1:
            mean = f"\\textbf{{{mean}}}"
        elif order.iloc[i] == 2:
            mean = f"\\underline{{{mean}}}"
        rows.append(f"\\texttt{{{algo}}} & ${beta}$ & {mean} & {se} & {n} \\\\")
    body = "\n".join(rows)
    # column spec + header are author-editable too, but the script owns the full block;
    # keep them in sync with the header wording in main.tex.
    return (
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.2cm} r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Algorithm} & \\textbf{$\\beta^\\star$} & \\textbf{$\\bar{R}$ ($\\downarrow$)} & \\textbf{SE} & \\textbf{$n$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def inject_table(tex_path: Path, table_name: str, tabular_block: str) -> None:
    """Replace the body between the AUTO-GENERATED TABLE markers for `table_name` in main.tex.

    Hard-fails if either marker is missing or duplicated, so a typo surfaces immediately
    rather than silently doing nothing. Only the bracketed tabular is touched; caption,
    label, and prose are preserved.
    """
    begin = f"% >>> AUTO-GENERATED TABLE START: {table_name}"
    end = f"% <<< AUTO-GENERATED TABLE END: {table_name}"
    text = tex_path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(
            f"Marker mismatch for table {table_name!r} in {tex_path}: "
            f"BEGIN={text.count(begin)}, END={text.count(end)} (expect 1 each). "
            "Add the markers in main.tex first."
        )
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    tex_path.write_text(f"{before}{begin}\n{tabular_block}\n{end}{after}")


def main() -> None:
    fin = load_finished(C.SWEEP_DIR / "runs_slim.csv")
    g = group_by_algo_beta(fin)
    best = best_per_algorithm(g)
    verify = verify_train_level(fin)

    # write the data products
    (C.SWEEP_DIR).mkdir(parents=True, exist_ok=True)
    g.to_csv(C.SWEEP_DIR / "all_groups.csv", index=False)
    best.to_csv(C.SWEEP_DIR / "best_beta_per_algorithm.csv", index=False)
    verify.to_csv(C.SWEEP_DIR / "verify_train_level.csv", index=False)

    # bar plot
    make_bar(best, C.PLOTS_DIR / "bar_best_beta_eval_reward")

    # splice the results tabular directly into main.tex (no separate .tex fragment)
    inject_table(C.MAIN_TEX, "trainrun1-best", results_tabular(best))
    print(f"spliced tab:trainrun1-best into {C.MAIN_TEX}", file=sys.stderr)

    # console summary
    print("BEST BETA PER ALGORITHM (eval reward, finished):", file=sys.stderr)
    print(best.to_string(index=False), file=sys.stderr)
    print(f"\ntotal finished runs: {len(fin)}", file=sys.stderr)
    print(f"best-beta groups total runs: {int(best['n'].sum())}", file=sys.stderr)


if __name__ == "__main__":
    main()
