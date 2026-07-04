#!/usr/bin/env python
"""Local work-queue worker for the run-3.1.2 folder's sweeps (PyTorch / SB3 SAC), scoped by SWEEP_ID.

Loop: atomically claim a pending config (rename into running/; since the 2026-07-03 follow-up sweep the
claim is approximately run-id-ORDERED — see claim()), run train.py, move the marker to done/ (rc 0) or
failed/ (otherwise), repeat until pending/ is empty. The
per-run stopwatch PER_RUN_TIMEOUT force-stops a single hung training run (the worker survives and claims
the next config); healthy runs take ~10-15 h eval-off and up to ~19 h with eval ON, so 24 h only ever
fires on genuinely stuck runs.

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root with train.py), SWEEP_ID (this arm's sweep id).
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
PER_RUN_TIMEOUT = 24 * 60 * 60  # 24 h stopwatch per RUN: above the ~19 h worst healthy case (eval ON)

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
    The per-config `params` (the ablation knobs) and the global `fixed` args pass through verbatim."""
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--algorithm={cfg["algorithm"]}',
        f'--beta={cfg["beta"]}',
        f'--a_seed={cfg["a_seed"]}',
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
