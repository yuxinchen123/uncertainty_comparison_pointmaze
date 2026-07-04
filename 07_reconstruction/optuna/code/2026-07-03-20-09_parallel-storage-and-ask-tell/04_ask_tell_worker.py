"""Worker process for point 4: mimic the project's file work queue exactly. Repeatedly claim a
pending config JSON by atomic os.rename into running/, compute a small noisy objective from its
params, write the result JSON into results/ atomically, then move the marker into done/. This is
the compute half; the controller (04_ask_tell_controller.py) does the Optuna ask/tell half."""

import os
import sys
import json
import time
import hashlib
import numpy as np


def substream(base_seed, *parts):
    """Return a numpy generator seeded by hashing a stable name, one stream per named quantity."""
    # stable key -> 32-bit seed so each trial's noise is independent and reproducible
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def evaluate(params, trial_number):
    """Compute the noisy quadratic objective (x-2)^2 + (y+1)^2 + per-trial noise."""
    # deterministic per-trial noise so a rerun gives the same value for the same trial
    noise = substream("obj", trial_number).normal(0.0, 0.5)
    return (params["x"] - 2.0) ** 2 + (params["y"] + 1.0) ** 2 + noise


def try_claim(pending_dir, running_dir, name):
    """Atomically claim one pending file via os.rename; return True if this worker won the race."""
    # os.rename is atomic on one filesystem: exactly one worker moves the file, the rest lose
    src = os.path.join(pending_dir, name)
    dst = os.path.join(running_dir, name)
    try:
        os.rename(src, dst)
        return True
    except FileNotFoundError:
        # another worker claimed it first; this specific race is the whole point of the pattern
        return False


def main():
    """Loop claiming and running pending trials until the stop sentinel appears and none remain."""
    # queue directory layout is passed on the command line
    queue_dir = sys.argv[1]
    pending_dir = os.path.join(queue_dir, "pending")
    running_dir = os.path.join(queue_dir, "running")
    done_dir = os.path.join(queue_dir, "done")
    results_dir = os.path.join(queue_dir, "results")
    stop_path = os.path.join(queue_dir, "STOP")

    # keep working until told to stop AND there is nothing left to claim
    while True:
        # list ONLY finalized markers; a ".json.tmp" file is still being written by the controller,
        # so claiming it would read a partial file (and steal it from the controller's rename)
        names = sorted(n for n in os.listdir(pending_dir) if n.endswith(".json"))
        if not names:
            # nothing to do: exit only if the controller has signalled completion
            if os.path.exists(stop_path):
                break
            time.sleep(0.05)
            continue

        # try to claim the first pending marker; on a lost race just retry the loop
        name = names[0]
        if not try_claim(pending_dir, running_dir, name):
            continue

        # read the claimed config: {trial_number, params}
        with open(os.path.join(running_dir, name)) as f:
            config = json.load(f)
        trial_number = config["trial_number"]
        params = config["params"]

        # mimic real training work taking a fraction of a second (reproducible per-trial duration)
        work_seconds = 0.1 + 0.2 * substream("work", trial_number).random()
        time.sleep(work_seconds)
        value = evaluate(params, trial_number)

        # write the result atomically (temp file then os.rename) so the controller never sees a partial file
        result_tmp = os.path.join(results_dir, f"{trial_number}.json.tmp")
        result_final = os.path.join(results_dir, f"{trial_number}.json")
        with open(result_tmp, "w") as f:
            json.dump({"trial_number": trial_number, "value": value}, f)
        os.rename(result_tmp, result_final)

        # move the marker from running/ to done/ to record the claim as finished
        os.rename(os.path.join(running_dir, name), os.path.join(done_dir, name))


main()
