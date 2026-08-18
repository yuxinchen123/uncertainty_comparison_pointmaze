#!/usr/bin/env python
"""Local work-queue worker for this run folder's sweeps (PyTorch / SB3 SAC), scoped by SWEEP_ID.

Loop: atomically claim a pending config (rename into running/; since the 2026-07-03 follow-up sweep the
claim is approximately run-id-ORDERED — see claim()), run train.py, move the marker to done/ (rc 0) or
failed/ (otherwise), repeat until pending/ is empty. No per-run stopwatch: the only time limit is the
job's Slurm walltime. (Removed 2026-07-06 by user decision: the old 24 h PER_RUN_TIMEOUT killed 71
run-3.2.1 runs that had reached 750k-950k of 1M steps — nearly done, just slowed to ~half speed by node
contention, NOT hung — so the cap threw away ~24 CPU-hours of near-complete work each. A genuinely hung
run now holds one worker of ~1000 until the job's walltime, an acceptable cost vs. killing progressing runs.)

This run's two additions over the run-5 worker (both in build_cmd):
- the queue JSONs carry --env_setup (run 8.1 sweeps 8 point/ant maze env setups), passed through verbatim;
- a per-node DEVICE override: when WORKER_DEVICE is set (cpu / cuda), --device=<WORKER_DEVICE> is appended
  LAST so it wins over the queue's device-agnostic --device=cpu (argparse takes the last occurrence). GPU
  worker jobs export WORKER_DEVICE=cuda; CPU jobs export cpu; the queue JSONs stay device-neutral.

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root with train.py), SWEEP_ID (this arm's sweep id),
     WORKER_DEVICE (optional cpu / cuda device override for the claimed run).
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
    """Print a timestamped worker-tagged line, flushed (so the slurm .log streams live and claim/finish
    times are recomputable from the log alone)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp} worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


def claim():
    """Atomically claim one pending config, approximately in run-id order. Returns (name, running_path)
    or (None, None) when empty."""
    # Sort pending names (zero-padded ids -> lexical order == id order) and pick randomly among only the
    # FIRST 32: workers stay inside the earliest-id window so early seeds FINISH first (a cancelled sweep
    # then leaves complete early-seed coverage), while the random pick within the window keeps the
    # rename-collision protection. Required for sweeps after 3.1.2 (.claude/rules/run-id-and-logging.md;
    # the old fully-shuffled claim executed runs in random order).
    try:
        names = sorted(os.listdir(PENDING))
    except FileNotFoundError:
        return None, None
    while names:
        name = random.choice(names[:32])
        try:
            os.rename(os.path.join(PENDING, name), os.path.join(RUNNING, name))
            return name, os.path.join(RUNNING, name)
        except OSError:
            names.remove(name)  # another worker won this file; retry within the refreshed window
    return None, None


def build_cmd(cfg):
    """Build the train.py argv for one claimed config; JSON lands in data/<sweep_id>/local/<id>.json.
    The per-config `params` (the ablation knobs) and the global `fixed` args pass through verbatim; the
    env setup and (optionally) the per-node device are appended below."""
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--algorithm={cfg["algorithm"]}',
        f'--beta={cfg["beta"]}',
        f'--a_seed={cfg["a_seed"]}',
        f'--env_setup={cfg["env_setup"]}',   # run 8.1 sweeps 8 point/ant maze env setups (queue-carried)
        '--z_logging_mode=local',
        '--use_wandb=False',
        f'--local_log_dir={DATA}',
        f'--run_id={cfg["run_id"]}',
        f'--run_total={cfg["run_total"]}',
    ]
    # per-config ablation knobs (normalization / ridge / clip / timing / input), then the global fixed args
    for k, v in cfg.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    # per-node device override: WORKER_DEVICE=cuda (GPU jobs) / cpu (CPU jobs) appended LAST so argparse's
    # last-occurrence-wins beats the queue's device-agnostic --device=cpu from `fixed`.
    #   before: argv has ... --device=cpu (from fixed)
    #   after (WORKER_DEVICE=cuda): argv has ... --device=cpu ... --device=cuda  -> train.py sees cuda
    if os.environ.get("WORKER_DEVICE"):
        args.append(f"--device={os.environ['WORKER_DEVICE']}")
    return args


def main():
    """Claim-run-mark loop over this arm's queue until its pending set is empty."""
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
        log(f"claimed {name} :: {cfg['env_setup']} {cfg['algorithm']} beta={cfg['beta']} "
            f"seed={cfg['a_seed']} params={cfg.get('params', {})}")
        t0 = time.time()
        # no per-run timeout: run train.py to natural completion (or until the job's Slurm walltime
        # kills the whole step). A slow-but-progressing run is never guillotined near the finish.
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
