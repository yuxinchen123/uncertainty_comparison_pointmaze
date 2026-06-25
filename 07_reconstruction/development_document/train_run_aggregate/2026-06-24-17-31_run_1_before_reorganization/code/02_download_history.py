#!/usr/bin/env python3
"""
Download the eval-reward learning curve (time series) for the runs in each
algorithm's best-beta group, for the line plot.

"Best beta" = the beta whose finished runs have the largest mean
`eval/mean_extrinsic_reward` (the project primary metric), computed from the slim
summary CSV (01_download_slim.py output). Only those runs need a time series, so
this pulls history for ~400 runs, not all 8000.

Speed: uses the sampled-history endpoint `run.history(...)` (one REST call per
run) instead of `run.scan_history(...)` (streams every logged step, minutes per
run), and fetches runs concurrently with a thread pool. ~400 runs in a few
minutes.

Output (under <sweep_dir>):
  best_beta_per_algorithm.csv   one row per algorithm: chosen beta, mean, SEM, n
  history_best_beta.csv         long format: algorithm, beta, a_seed, run_id, step, eval_reward
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = SCRIPT_DIR.parent / "data" / "y24vyh06"

METRIC = "eval/mean_extrinsic_reward"  # primary metric; defines "best beta" and the curve
XKEY = "step"  # env-timestep x-axis logged next to the eval metric


def best_beta_table(slim_csv: Path) -> pd.DataFrame:
    """Finished-only mean eval reward per (algorithm, beta); keep the best beta per algorithm."""
    df = pd.read_csv(slim_csv, low_memory=False)
    fin = df[df["state"] == "finished"].copy()
    fin["beta"] = pd.to_numeric(fin["config.beta"], errors="coerce")
    fin["ev"] = pd.to_numeric(fin[f"summary.{METRIC}"], errors="coerce")
    # group -> mean/sem/n per (algorithm, beta), then argmax mean per algorithm
    g = (
        fin.groupby(["config.algorithm", "beta"])
        .agg(mean_ev=("ev", "mean"), se_ev=("ev", "sem"), n=("ev", "count"))
        .reset_index()
    )
    best = g.loc[g.groupby("config.algorithm")["mean_ev"].idxmax()].copy()
    best = best.sort_values("mean_ev", ascending=False).reset_index(drop=True)
    return best


def runs_in_best_groups(slim_csv: Path, best: pd.DataFrame) -> pd.DataFrame:
    """Run ids (with algorithm, beta, a_seed) belonging to the best-beta group of each algorithm."""
    df = pd.read_csv(slim_csv, low_memory=False)
    fin = df[df["state"] == "finished"].copy()
    fin["beta"] = pd.to_numeric(fin["config.beta"], errors="coerce")
    # inner-join finished runs to the (algorithm, best-beta) pairs
    keys = best[["config.algorithm", "beta"]]
    sel = fin.merge(keys, on=["config.algorithm", "beta"], how="inner")
    return sel[["config.algorithm", "beta", "config.a_seed", "run_id"]].reset_index(drop=True)


def fetch_one(api, entity: str, project: str, row: pd.Series) -> pd.DataFrame:
    """Fetch one run's (step, eval reward) curve as a long-format frame; empty frame if no history."""
    run = api.run(f"{entity}/{project}/{row['run_id']}")
    h = run.history(keys=[METRIC, XKEY], samples=4000, pandas=True)
    if h is None or len(h) == 0 or METRIC not in h.columns:
        return pd.DataFrame()
    # keep only the two columns and tag with run identity
    out = h[[XKEY, METRIC]].dropna().rename(columns={XKEY: "step", METRIC: "eval_reward"}).copy()
    out["algorithm"] = row["config.algorithm"]
    out["beta"] = row["beta"]
    out["a_seed"] = row["config.a_seed"]
    out["run_id"] = row["run_id"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download best-beta-group eval curves for the line plot.")
    p.add_argument("--entity", default="catresearch")
    p.add_argument("--project", default="rnd_07_reconstruction")
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    import wandb

    slim_csv = args.sweep_dir / "runs_slim.csv"
    best = best_beta_table(slim_csv)
    best.to_csv(args.sweep_dir / "best_beta_per_algorithm.csv", index=False)
    print("best beta per algorithm:\n" + best.to_string(index=False), file=sys.stderr)

    runs = runs_in_best_groups(slim_csv, best)
    print(f"fetching history for {len(runs)} runs ...", file=sys.stderr)

    api = wandb.Api(timeout=180)
    # fan out history fetches; each is one REST call, so threads overlap the network waits
    frames = []
    n_empty = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, api, args.entity, args.project, r): i for i, r in runs.iterrows()}
        done = 0
        for fut in as_completed(futs):
            f = fut.result()
            if len(f) == 0:
                n_empty += 1
            else:
                frames.append(f)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(runs)} runs fetched", file=sys.stderr, flush=True)

    hist = pd.concat(frames, ignore_index=True)
    out_path = args.sweep_dir / "history_best_beta.csv"
    hist.to_csv(out_path, index=False)
    print(
        f"Done: {len(hist)} rows from {hist['run_id'].nunique()} runs "
        f"({n_empty} runs had no history) -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
