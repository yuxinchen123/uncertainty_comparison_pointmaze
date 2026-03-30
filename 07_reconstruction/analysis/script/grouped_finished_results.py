#!/usr/bin/env python3
"""
Group finished runs by (algorithm, beta): mean eval reward, SEM of the mean (pandas
`sem`, ddof=1), mean summary step, counts. `--markdown-best` picks the beta with the
largest mean reward per algorithm (first group if tied), then sorts rows by that mean
descending.

All analysis in ../analysis.md assumes W&B state `finished` only; this script filters
accordingly before aggregating.

Input: combined_runs_index.csv (or any runs_index with the same flat column names).

Dependencies: pip install pandas
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = ANALYSIS_ROOT / "data" / "combined_runs_index.csv"

COL_ALGO = "config.algorithm"
COL_BETA = "config.beta"
COL_STATE = "state"
COL_EVAL_EXT = "summary.eval/mean_extrinsic_reward"
COL_STEP = "summary.step"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    fin = df[df[COL_STATE] == "finished"].copy()
    fin[COL_BETA] = pd.to_numeric(fin[COL_BETA], errors="coerce")
    fin[COL_EVAL_EXT] = pd.to_numeric(fin[COL_EVAL_EXT], errors="coerce")
    fin[COL_STEP] = pd.to_numeric(fin[COL_STEP], errors="coerce")
    g = (
        fin.groupby([COL_ALGO, COL_BETA], dropna=False)
        .agg(
            mean_eval_ext=(COL_EVAL_EXT, "mean"),
            se_eval_ext=(COL_EVAL_EXT, "sem"),
            mean_step=(COL_STEP, "mean"),
            n_runs=(COL_STATE, "count"),
        )
        .reset_index()
        .sort_values([COL_ALGO, COL_BETA])
    )
    g["se_eval_ext"] = g["se_eval_ext"].fillna(0.0)
    return g


def best_per_algorithm(g: pd.DataFrame) -> pd.DataFrame:
    """One row per algorithm: the `config.beta` with largest mean eval reward (ties: first)."""
    idx = g.groupby(COL_ALGO, sort=False)["mean_eval_ext"].idxmax()
    b = g.loc[idx]
    return (
        b.sort_values(["mean_eval_ext", COL_ALGO], ascending=[False, True])
        .reset_index(drop=True)
    )


def to_markdown(g: pd.DataFrame) -> str:
    lines = [
        "| algorithm | beta | eval/mean_extrinsic_reward | step | n_runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in g.iterrows():
        algo = str(r[COL_ALGO]).replace("|", "\\|")
        beta = r[COL_BETA]
        beta_s = f"{beta:.4g}" if pd.notna(beta) else ""
        ext = r["mean_eval_ext"]
        ext_s = f"{ext:.6f}" if pd.notna(ext) else ""
        step = r["mean_step"]
        step_s = str(int(round(step))) if pd.notna(step) else ""
        lines.append(f"| {algo} | {beta_s} | {ext_s} | {step_s} | {int(r['n_runs'])} |")
    return "\n".join(lines)


def to_markdown_best(b: pd.DataFrame) -> str:
    lines = [
        "| algorithm | beta | eval/mean_extrinsic_reward | SE | step | n_runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in b.iterrows():
        algo = str(r[COL_ALGO]).replace("|", "\\|")
        beta = r[COL_BETA]
        beta_s = f"{beta:.4g}" if pd.notna(beta) else ""
        ext = r["mean_eval_ext"]
        ext_s = f"{ext:.6f}" if pd.notna(ext) else ""
        se = r["se_eval_ext"]
        se_s = f"{se:.6f}" if pd.notna(se) else ""
        step = r["mean_step"]
        step_s = str(int(round(step))) if pd.notna(step) else ""
        lines.append(f"| {algo} | {beta_s} | {ext_s} | {se_s} | {step_s} | {int(r['n_runs'])} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Grouped metrics for finished runs only.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Full grouped table (markdown) to stdout",
    )
    p.add_argument(
        "--markdown-best",
        action="store_true",
        help="Best beta per algorithm (max mean reward) with SE; markdown to stdout",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Missing {args.csv}; run combine_sweep_runs.py first.", file=sys.stderr)
        sys.exit(1)

    need = [COL_STATE, COL_ALGO, COL_BETA, COL_EVAL_EXT, COL_STEP]
    df = pd.read_csv(args.csv, usecols=need, low_memory=False)
    g = build_table(df)

    if args.markdown and args.markdown_best:
        print("Choose only one of --markdown / --markdown-best.", file=sys.stderr)
        sys.exit(2)
    if args.markdown:
        print(to_markdown(g))
    elif args.markdown_best:
        print(to_markdown_best(best_per_algorithm(g)))
    else:
        print(g.to_string(index=False))
        print(
            f"\n({len(g)} groups, {g['n_runs'].sum():.0f} finished runs)",
            file=sys.stderr,
        )
        print(
            "Use --markdown or --markdown-best for analysis.md tables.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
