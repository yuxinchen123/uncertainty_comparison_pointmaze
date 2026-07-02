#!/usr/bin/env python3
"""Train run 3.1.1 beta-star table (LaTeX fragment): one row per METHOD, its single best configuration.

For each method (A1 batch elliptical, A2 global-sample, A3 global-add, A4 RND next-state) we pool the final
TRAINING-episode extrinsic reward over seeds within each configuration (beta x ridge x input), keep only
configurations with >= MIN_SEEDS finished seeds, and report the configuration with the highest mean reward:

    Rbar = (1/n) sum_i R_i ,  SE = s / sqrt(n)   (s = sample std, ddof=1)

where R_i is run i's final (largest-step) train/mean_extrinsic_reward. Columns: Method, input*, lambda*,
beta*, Rbar, SE, n. Rows sorted by Rbar descending; best Rbar bold, second underlined (higher is better).
Output: <plots_dir>/reward_table.tex (also printed to stdout).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# table-ready method display (long texttt names get \allowbreak so they wrap in the p{} column)
METHOD_DISPLAY = {
    "A1_batch": r"\texttt{rnd\_\allowbreak elliptical} (batch)",
    "A2_global_sample": r"\texttt{rnd\_\allowbreak elliptical\_\allowbreak global} (sample)",
    "A3_global_add": r"\texttt{rnd\_\allowbreak elliptical\_\allowbreak global} (add)",
    "A4_rnd_next_state": r"\texttt{rnd\_\allowbreak next\_\allowbreak state}",
}


def _fmt_pow10(x: float) -> str:
    """Format a (positive) power of ten as $10^{k}$ ($1$ for k=0); fall back to $x:g$ otherwise."""
    if x is None or not C._finite(x) or x <= 0:
        return "---"
    k = math.log10(x)
    if abs(k - round(k)) < 1e-9:
        kk = int(round(k))
        return "$1$" if kk == 0 else f"$10^{{{kk}}}$"
    return f"${x:g}$"


def _fmt_input(method: str, inp) -> str:
    """Format the feature-input cell: (s,a) / s' for elliptical; fixed next-state for RND."""
    if method == "A4_rnd_next_state":
        return r"$s'$ (fixed)"
    return r"$(s,a)$" if inp == "state_action" else r"$s'$"


def final_reward_frame(records) -> pd.DataFrame:
    """One row per run with a final training reward: [method, beta, ridge, input, seed, reward]."""
    rows = []
    for r in records:
        mid = r.method_id()
        if mid is None or mid == C.GT_METHOD or r.final_train_reward is None:
            continue
        # ridge/input as canonical strings ('none' for RND) so the groupby never produces NaN keys
        rows.append({"method": mid, "beta": r.beta, "ridge": C.ridge_str(r.regularization),
                     "input": C.input_str(r.feature_input), "seed": r.seed, "reward": r.final_train_reward})
    return pd.DataFrame(rows, columns=["method", "beta", "ridge", "input", "seed", "reward"])


def best_config_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method,beta,ridge,input) pool mean/SE/n over seeds; per method keep the highest-mean config with
    n >= MIN_SEEDS. Returns [method, beta, ridge, input, Rbar, SE, n] sorted by Rbar descending."""
    if df.empty:
        return pd.DataFrame(columns=["method", "beta", "ridge", "input", "Rbar", "SE", "n"])
    # pool over seeds within each configuration (dropna=False keeps RND's None ridge/input as a group)
    g = (df.groupby(["method", "beta", "ridge", "input"], dropna=False)["reward"]
         .agg(Rbar="mean", SE="sem", n="count").reset_index())
    g["SE"] = g["SE"].fillna(0.0)
    g = g[g["n"] >= C.MIN_SEEDS]
    rows = []
    for m in C.METHODS:
        sub = g[g["method"] == m]
        if sub.empty:
            continue
        rows.append(sub.loc[sub["Rbar"].idxmax()])
    if not rows:
        return pd.DataFrame(columns=["method", "beta", "ridge", "input", "Rbar", "SE", "n"])
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out.sort_values("Rbar", ascending=False).reset_index(drop=True)


def reward_tabular(summary: pd.DataFrame) -> str:
    """LaTeX tabular for the per-method best-config table (best Rbar bold, second underlined)."""
    best, second = C.rank_marks(summary["Rbar"].tolist(), lower_is_better=False)
    rows = []
    for i, r in summary.iterrows():
        rbar = f"{r['Rbar']:.2f}"
        if i in best:
            rbar = f"\\textbf{{{rbar}}}"
        elif i in second:
            rbar = f"\\underline{{{rbar}}}"
        lam = "---" if r["method"] == "A4_rnd_next_state" else _fmt_pow10(float(r["ridge"]))
        rows.append(
            f"{METHOD_DISPLAY[r['method']]} & {_fmt_input(r['method'], r['input'])} & {lam} & "
            f"{_fmt_pow10(r['beta'])} & {rbar} & {r['SE']:.2f} & {int(r['n'])} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% Train run 3.1.1 beta-star table: one row per method, its best config (training-episode reward).\n"
        "% Higher reward is better; rows sorted by Rbar descending; best bold, second-best underlined.\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.3cm} c c c r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{input$^\\star$} & \\textbf{$\\lambda^\\star$} & \\textbf{$\\beta^\\star$} & "
        "\\textbf{$\\bar{R}$ ($\\uparrow$)} & \\textbf{SE} & \\textbf{$n$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def build(data_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Load records, build the per-method best-config table, write reward_table.tex, return the summary frame."""
    records = C.load_records(data_dir)
    summary = best_config_per_method(final_reward_frame(records))
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "reward_table.tex").write_text(reward_tabular(summary) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 3.1.1: per-method best-config reward table (LaTeX fragment).")
    p.add_argument("data_dir", type=Path, help="run-3.1.1 sweep data dir (holds the local/ subdir), e.g. data/<sweep_id>")
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR, help="output plots dir")
    args = p.parse_args()
    summary = build(args.data_dir, args.plots_dir)
    print((args.plots_dir / "reward_table.tex").read_text())
    print("per-method best-config summary:", file=sys.stderr)
    print(summary.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
