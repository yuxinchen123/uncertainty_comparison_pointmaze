#!/usr/bin/env python3
"""
Train run 2: build every analysis artifact from the local per-run JSON data.

Runs all four builders against one data directory and writes their outputs into one
plots directory:
  - reward_table.tex             (make_reward_table)
  - bar_final_reward.{pdf,png}   (make_reward_bar)
  - line_reward_curve.{pdf,png}  (make_reward_curve; eval solid + train dashed, each >=30-seed truncated)
  - distance_table.tex           (make_distance_table)

Safe to re-run at any sweep milestone: the loader skips partial / missing JSON and
each builder no-ops (rather than crashing) when an algorithm or metric has
no data yet.

Usage:
  conda run -n exploration python make_all.py <data_dir> <out_plots_dir>
Both arguments are optional; they default to the run-2 data dir and analysis/plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import make_distance_table  # noqa: E402
import make_reward_bar  # noqa: E402
import make_reward_curve  # noqa: E402
import make_reward_table  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train run 2: build all reward / distance / runtime artifacts.")
    p.add_argument("data_dir", nargs="?", type=Path, default=C.DEFAULT_DATA_DIR, help="run-2 data dir (holds the mode subdirs)")
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR, help="output plots dir")
    args = p.parse_args()

    data_dir, plots_dir = args.data_dir, args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[make_all] data_dir={data_dir}", file=sys.stderr)
    print(f"[make_all] plots_dir={plots_dir}", file=sys.stderr)

    # report how much data was found so a milestone run is self-documenting
    records = C.load_records(data_dir)
    print(f"[make_all] loaded {len(records)} run record(s)", file=sys.stderr)

    # build each artifact in turn; each builder is independently safe on partial data.
    # No timing bar: run 2 no longer uses wandb, so there is no full-vs-param-only runtime comparison.
    make_reward_table.build(data_dir, plots_dir)
    make_reward_bar.build(data_dir, plots_dir)
    make_reward_curve.build(data_dir, plots_dir)
    make_distance_table.build(data_dir, plots_dir)

    print(f"[make_all] done; artifacts in {plots_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
