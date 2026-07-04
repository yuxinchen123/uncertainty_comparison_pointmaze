#!/usr/bin/env python
"""Report the unit-norm ridge-1e-8 follow-up sweep's progress: completed seeds per beta, partial
checkpoints, queue-state tallies, stale running/ markers, and whether every beta has reached the full
50-seed target.

Usage: progress_unit_ridge_1e8.py <sweep_id>

A "config" is one of the 5 beta points (the cell — unit norm, ridge 1e-8, no clip — is fixed). Only
records with completed=true count as finished; completed=false records are checkpoint partials. A
running/ marker older than PER_RUN_TIMEOUT (24 h) counts as stale/lost.
"""
import os
import sys
import json
import glob
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_queue_unit_ridge_1e8 import BETAS, RUN_TOTAL  # noqa: E402

TARGET = 50  # seeds per beta = complete
STALE_SECONDS = 24 * 60 * 60  # matches worker.py PER_RUN_TIMEOUT


def scan(sweep_id):
    """Scan one sweep's data/ and queue/: seeds per beta, partial/bad counts, queue tallies, stale markers."""
    seeds_by_beta = defaultdict(set)
    n_bad = n_partial = 0
    for f in glob.glob(os.path.join(RUN_DIR, "data", sweep_id, "local", "*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            n_bad += 1
            continue
        # checkpoint records are unfinished runs; only completed=true counts as done
        if r.get("completed", True) is False:
            n_partial += 1
            continue
        seeds_by_beta[float(r["beta"])].add(int(r["a_seed"]))
    # queue-state tallies incl. stale running/ markers (a job killed at walltime leaves its marker forever)
    q = os.path.join(RUN_DIR, "queue", sweep_id)
    counts = {s: len(os.listdir(os.path.join(q, s))) if os.path.isdir(os.path.join(q, s)) else 0
              for s in ("pending", "running", "done", "failed")}
    now = time.time()
    stale = sum(1 for p in glob.glob(os.path.join(q, "running", "*.json"))
                if now - os.path.getmtime(p) > STALE_SECONDS)
    return seeds_by_beta, n_bad, n_partial, counts, stale


def main():
    """Print per-beta completed-seed coverage and the TARGET_MET / TARGET_NOT_MET trigger line."""
    sweep_id = sys.argv[1]
    seeds_by_beta, n_bad, n_partial, counts, stale = scan(sweep_id)
    done_runs = sum(len(v) for v in seeds_by_beta.values())
    print(f"sweep {sweep_id}: completed={done_runs}/{RUN_TOTAL} partial={n_partial} bad={n_bad} "
          f"queue(p/r/d/f)={counts['pending']}/{counts['running']}/{counts['done']}/{counts['failed']} "
          f"stale_running={stale}")
    per_beta = {float(b): len(seeds_by_beta.get(float(b), set())) for b in BETAS}
    print("per-beta completed seeds: " + "  ".join(f"beta={b:g}: {n}/{TARGET}" for b, n in sorted(per_beta.items())))
    mn = min(per_beta.values())
    if mn >= TARGET:
        print(f"TARGET_MET min={mn}")
    else:
        print(f"TARGET_NOT_MET min={mn}")


if __name__ == "__main__":
    main()
