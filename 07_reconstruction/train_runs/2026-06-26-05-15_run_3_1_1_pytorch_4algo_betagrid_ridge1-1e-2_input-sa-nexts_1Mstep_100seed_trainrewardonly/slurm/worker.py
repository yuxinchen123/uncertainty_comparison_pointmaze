#!/usr/bin/env python
"""Local work-queue worker for Train run 3.1.1 (PyTorch / SB3 SAC), scoped to ONE sweep by SWEEP_ID.

Drives the live trainer train.py (NOT the JAX run4_train.py). The run folder holds many sweeps, each under
its own queue/<sweep_id>/ and data/<sweep_id>/ subtree (.claude/rules/run-id-and-logging.md); this worker
only ever touches the sweep named by SWEEP_ID, so concurrent/legacy sweeps never collide.

Each worker repeatedly:
  1. atomically claims one pending config (rename queue/<sweep_id>/pending/X -> .../running/X),
  2. runs train.py with that config (logging its JSON under data/<sweep_id>/local/),
  3. renames the file to .../done/X (rc 0) or .../failed/X (otherwise),
and loops until no pending configs remain. 8 workers run per Slurm job (srun ntasks=8).

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root with train.py), SWEEP_ID (this sweep's id).
sys.executable is the exploration-env python (worker.slurm invokes this file with it).
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
PER_RUN_TIMEOUT = 16 * 60 * 60  # 16 h cap: above the real ~5-8 h/run (1e6 steps, no eval, 2 CPUs), below the job limit

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'


def log(msg):
    """Print a worker-tagged line, flushed (so the slurm .log streams live)."""
    print(f"[worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


def claim():
    """Atomically claim one pending config. Returns (name, running_path) or (None, None) when empty."""
    # list pending, shuffle to spread workers across files, then win one via atomic rename
    try:
        names = os.listdir(PENDING)
    except FileNotFoundError:
        return None, None
    random.shuffle(names)
    for name in names:
        try:
            os.rename(os.path.join(PENDING, name), os.path.join(RUNNING, name))
            return name, os.path.join(RUNNING, name)
        except OSError:
            continue  # another worker won the race; try the next file
    return None, None


def build_cmd(cfg):
    """Build the train.py argv for one claimed config; JSON lands in data/<sweep_id>/local/<id>.json.
    The per-config `params` (elliptical knobs) and the global `fixed` args are passed through verbatim."""
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--algorithm={cfg["algorithm"]}',
        f'--beta={cfg["beta"]}',
        f'--a_seed={cfg["a_seed"]}',
        '--z_logging_mode=local',
        '--use_wandb=False',
        f'--local_log_dir={DATA}',
        f'--run_id={cfg["run_id"]}',        # sweep position -> id-based per-run JSON filename in train.py
        f'--run_total={cfg["run_total"]}',  # sweep size -> the "/total" of the run's id
    ]
    # per-config method knobs (elliptical only; empty for RND), then the global fixed args
    for k, v in cfg.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    return args


def main():
    """Claim-run-mark loop over this sweep's queue until its pending set is empty."""
    # small startup jitter so workers don't all import torch / build the env at the same instant
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    n_done = n_fail = 0
    while True:
        name, path = claim()
        if name is None:
            log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: {cfg['algorithm']} beta={cfg['beta']} seed={cfg['a_seed']} params={cfg.get('params', {})}")
        t0 = time.time()
        try:
            rc = subprocess.call(build_cmd(cfg), cwd=PROJ, timeout=PER_RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            rc = 124
            log(f"{name} TIMED OUT after {PER_RUN_TIMEOUT}s")
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
