#!/usr/bin/env python3
"""
Convert combined_runs_index.csv -> Parquet for fast reload.

Typical speedup (3062 rows × ~63k columns, ~239MB CSV on one machine):
  read_csv full  ~500s, ~6GB RAM
  read_parquet   ~10s, smaller on-disk size (~260MB Snappy)

Dependencies: pip install pandas pyarrow

Optional --slim keeps only common analysis columns (small file, sub-second loads).

All analysis in ../analysis.md assumes finished W&B runs only (`state == finished`) when
summarizing metrics; this script does not filter states.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = ANALYSIS_ROOT / "data" / "combined_runs_index.csv"
DEFAULT_PQ = ANALYSIS_ROOT / "data" / "combined_runs_index.parquet"
DEFAULT_SLIM_PQ = ANALYSIS_ROOT / "data" / "combined_runs_index.slim.parquet"

# Adjust to match columns present in your export (missing cols are ignored).
SLIM_COLUMNS = [
    "sweep_id",
    "run_id",
    "run_name",
    "state",
    "url",
    "config.algorithm",
    "config.beta",
    "config.a_seed",
    "config.discount_factor",
    "config.env_max_episode",
    "config.goal_position",
    "config.total_timesteps",
    "config.apply_termination_wrapper",
    "summary.eval/mean_extrinsic_reward",
    "summary.eval/mean_ep_length",
    "summary.eval/mean_total_reward",
]


def main() -> None:
    p = argparse.ArgumentParser(description="CSV -> Parquet for W&B combined runs table")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_PQ)
    p.add_argument(
        "--slim",
        action="store_true",
        help=f"Also write {DEFAULT_SLIM_PQ.name} with a small column subset",
    )
    p.add_argument(
        "--slim-output",
        type=Path,
        default=DEFAULT_SLIM_PQ,
        help="Path for --slim file",
    )
    args = p.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("Need: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    if not args.csv.is_file():
        print(f"Missing {args.csv}; run combine_sweep_runs.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {args.csv} …", file=sys.stderr)
    df = pd.read_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, compression="snappy", index=False)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    if args.slim:
        use = [c for c in SLIM_COLUMNS if c in df.columns]
        miss = [c for c in SLIM_COLUMNS if c not in df.columns]
        if miss:
            print(f"Slim: skipped missing columns: {miss[:10]}{'…' if len(miss)>10 else ''}", file=sys.stderr)
        slim = df[use]
        slim.to_parquet(args.slim_output, compression="snappy", index=False)
        print(
            f"Wrote {args.slim_output} ({args.slim_output.stat().st_size / 1e6:.2f} MB), "
            f"shape {slim.shape}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
