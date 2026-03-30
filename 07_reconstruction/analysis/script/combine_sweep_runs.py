#!/usr/bin/env python3
"""Merge runs_index.csv from both sweeps; adds column sweep_id for filtering.

All analysis in ../analysis.md assumes finished W&B runs only (`state == finished`) when
summarizing metrics; this script does not filter states.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = ANALYSIS_ROOT / "data"
DEFAULT_SWEEPS = ("y24vyh06", "j40bkl6y")
DEFAULT_OUT = DEFAULT_DATA / "combined_runs_index.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate sweep runs_index.csv files.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--sweeps",
        nargs="+",
        default=list(DEFAULT_SWEEPS),
        help="Sweep folder names under data/",
    )
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    fieldnames: list[str] | None = None
    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as outf:
        writer: csv.DictWriter | None = None
        for sid in args.sweeps:
            path = args.data_root / sid / "runs_index.csv"
            if not path.is_file():
                print(f"skip (missing): {path}", file=sys.stderr)
                continue
            with open(path, encoding="utf-8", newline="") as inf:
                r = csv.DictReader(inf)
                if writer is None:
                    fieldnames = ["sweep_id"] + list(r.fieldnames or [])
                    writer = csv.DictWriter(outf, fieldnames=fieldnames)
                    writer.writeheader()
                assert writer is not None
                for row in r:
                    row = dict(row)
                    row["sweep_id"] = sid
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
                    n += 1

    print(f"Wrote {n} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
