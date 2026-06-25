#!/usr/bin/env python
"""Resume helper for a cancelled sweep.

Rebuilds the queue under the REVISED convention (per-run integer id, seed-outermost order; see
build_queue.py) and marks every config that already has a finished per-run JSON as done, so relaunching
the fleet runs ONLY the not-yet-finished configs. Completion is keyed on (algorithm, a_seed), which is
filename-independent, so it works even though the already-finished JSONs use the old descriptive names.

Usage:  RUN_DIR=<train_runs/<run>/> python slurm/resume_setup.py   (then: bash slurm/launch_queue.sh)
"""
import os
import sys
import json
import glob
import subprocess

RUN = os.environ["RUN_DIR"]
QUEUE = os.path.join(RUN, "queue")


def finished_pairs() -> set:
    """The set of (algorithm, a_seed) that already have a finished JSON in data/local/.

    before: data/local/*.json, each a record like {"algorithm": "rnd_state", "a_seed": 17, ...}
    after:  {("rnd_state", 17), ("gt_position_velocity", 4), ...}
    """
    # one finished run == one readable JSON carrying its algorithm + seed
    pairs = set()
    for f in glob.glob(os.path.join(RUN, "data", "local", "*.json")):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue  # a half-written file is not a completed run; skip it
        if "algorithm" in d and "a_seed" in d:
            pairs.add((d["algorithm"], int(d["a_seed"])))
    return pairs


def main():
    """Regenerate the revised-convention queue, then pre-mark already-finished configs as done."""
    # 1. read who is already finished BEFORE clearing the queue (data/local is the durable completion record)
    finished = finished_pairs()
    print(f"{len(finished)} runs already finished (have a JSON in data/local/)")

    # 2. regenerate all configs under the revised convention (run_id, seed-outermost); --reset clears queue/*
    subprocess.check_call([sys.executable, os.path.join(RUN, "slurm", "build_queue.py"), "--reset"])

    # 3. move each already-finished config pending -> done so workers skip it; the rest stay pending to run
    pending, done = os.path.join(QUEUE, "pending"), os.path.join(QUEUE, "done")
    moved = 0
    for name in os.listdir(pending):
        cfg = json.load(open(os.path.join(pending, name)))
        algo = cfg["g_algo_beta"].split("|")[0]
        if (algo, int(cfg["a_seed"])) in finished:
            os.rename(os.path.join(pending, name), os.path.join(done, name))
            moved += 1
    print(f"marked {moved} finished configs as done; {len(os.listdir(pending))} remain to run")


if __name__ == "__main__":
    main()
