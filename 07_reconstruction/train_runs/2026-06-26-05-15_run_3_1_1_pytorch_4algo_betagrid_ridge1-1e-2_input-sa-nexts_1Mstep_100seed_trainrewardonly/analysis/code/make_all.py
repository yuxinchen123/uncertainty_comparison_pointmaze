#!/usr/bin/env python3
"""Driver: build the Train-run-3.1.1 beta-star table + training-reward curve from one sweep's data.

Usage: make_all.py data/<sweep_id> [plots_dir]   (plots default to ../plots next to this code dir)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import make_reward_table  # noqa: E402
import make_reward_curve  # noqa: E402


def main() -> None:
    """Run both builders against one sweep's data dir."""
    p = argparse.ArgumentParser(description="Train run 3.1.1: build reward table + curve.")
    p.add_argument("data_dir", type=Path, help="run-3.1.1 sweep data dir (holds local/), e.g. data/<sweep_id>")
    p.add_argument("plots_dir", nargs="?", type=Path, default=C.DEFAULT_PLOTS_DIR)
    args = p.parse_args()
    make_reward_table.build(args.data_dir, args.plots_dir)
    make_reward_curve.build(args.data_dir, args.plots_dir)


if __name__ == "__main__":
    main()
