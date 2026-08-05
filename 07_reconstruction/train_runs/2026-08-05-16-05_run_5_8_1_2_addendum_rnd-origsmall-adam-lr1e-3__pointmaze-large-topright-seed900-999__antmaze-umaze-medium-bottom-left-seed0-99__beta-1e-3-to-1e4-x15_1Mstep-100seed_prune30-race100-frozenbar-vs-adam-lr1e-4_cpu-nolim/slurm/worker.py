#!/usr/bin/env python
"""Local work-queue worker for the Adam learning-rate 1e-3 addendum, scoped by SWEEP_ID.

Loop: atomically claim a pending config, run train.py, move the marker to done/ (exit code 0) or
failed/ (anything else), repeat until pending/ is empty. One pool only — every run in this sweep is
1,000,000 env steps on cpu, so there is no walltime guard and no pool priority to decide.

No per-run stopwatch: the only time limit is the job's Slurm walltime. A slow-but-progressing run is
never killed near the finish; a run the walltime does kill leaves its partial checkpoint on disk and
its marker is re-pended by requeue_orphans.py.

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root holding train.py), SWEEP_ID.
"""
import os
import sys
import json
import time
import random
import subprocess

RUN = os.environ["RUN_DIR"]
PROJ = os.environ.get("PROJ_DIR", "/p/rlprojects/RND/07_reconstruction")
SWEEP_ID = os.environ["SWEEP_ID"]
QUEUE = os.path.join(RUN, "queue", SWEEP_ID)
PENDING = os.path.join(QUEUE, "pending")
RUNNING = os.path.join(QUEUE, "running")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
DATA = os.path.join(RUN, "data", SWEEP_ID)  # train.py appends /local -> data/<sweep_id>/local/<id>.json

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'


def log(msg):
    """Print a timestamped worker-tagged line, flushed, so the job log streams live and claim/finish
    times are recomputable from the log alone."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp} worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


def claim():
    """Atomically claim one pending config, approximately in run-id order. Returns (name, path) or
    (None, None) when pending/ is empty."""
    try:
        names = sorted(os.listdir(PENDING))
    except FileNotFoundError:
        return None, None
    # Names are zero-padded ids, so lexical order IS id order. Pick randomly among only the FIRST 32
    # so every worker stays inside the earliest-id window (early seeds FINISH first, which is what
    # lets the controller decide wave by wave) while the random pick keeps two workers from
    # colliding on the same file every time.
    while names:
        name = random.choice(names[:32])
        try:
            os.rename(os.path.join(PENDING, name), os.path.join(RUNNING, name))
            return name, os.path.join(RUNNING, name)
        except OSError:
            names.remove(name)  # another worker won this file; retry inside the refreshed window
    return None, None


def build_cmd(cfg):
    """Build the train.py argv for one claimed config; its JSON lands in
    data/<sweep_id>/local/<run_id>_of_<run_total>.json.

    The environment comes entirely from --env_setup, the arm knobs from the config's `params`, and
    the run-wide knobs from `fixed` — all pass through verbatim, so the queue marker is a complete
    record of the command that ran.
    """
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--algorithm={cfg["algorithm"]}',
        f'--beta={cfg["beta"]}',
        f'--a_seed={cfg["a_seed"]}',
        f'--env_setup={cfg["env_setup"]}',
        '--z_logging_mode=local',
        '--use_wandb=False',
        f'--local_log_dir={DATA}',
        f'--run_id={cfg["run_id"]}',
        f'--run_total={cfg["run_total"]}',
    ]
    for k, v in cfg.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    return args


def main():
    """Claim-run-mark loop until pending/ is empty."""
    # small startup jitter so the workers of one job do not all import torch at the same instant
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    log("started")
    n_done = n_fail = 0
    while True:
        name, path = claim()
        if name is None:
            log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: {cfg['env_setup']} beta={cfg['beta']} seed={cfg['a_seed']} "
            f"steps={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        rc = subprocess.call(build_cmd(cfg), cwd=PROJ)
        dest = DONE if rc == 0 else FAILED
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        n_done += (rc == 0)
        n_fail += (rc != 0)
        log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)} "
            f"(running totals: {n_done} done, {n_fail} failed)")


if __name__ == "__main__":
    main()
