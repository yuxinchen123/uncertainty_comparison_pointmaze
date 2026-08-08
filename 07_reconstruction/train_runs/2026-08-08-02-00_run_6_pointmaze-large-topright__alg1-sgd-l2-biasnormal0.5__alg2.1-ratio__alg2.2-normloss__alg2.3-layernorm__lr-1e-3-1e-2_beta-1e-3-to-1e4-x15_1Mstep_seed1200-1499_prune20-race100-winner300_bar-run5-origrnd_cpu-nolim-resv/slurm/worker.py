#!/usr/bin/env python
"""Local work-queue worker for the Adam learning-rate 1e-3 addendum, scoped by SWEEP_ID.

Loop: atomically claim a pending config, run train.py, move the marker to done/ (exit code 0) or
failed/ (anything else), repeat until nothing more can be claimed. A worker therefore keeps taking
new runs for the whole life of its job — it does NOT exit after one run.

WALLTIME GUARD. A worker stops claiming once its job has less remaining walltime than a run needs,
and exits cleanly instead. This run writes no model checkpoints, so a run the walltime kills
restarts from zero somewhere else: starting a 16-hour run with 3 hours left is guaranteed waste.
Exiting early instead lets the job end, frees the pool, and lets the monitor's top-up submit a fresh
full-walltime job that picks the work up. The cpu partition caps jobs at 4 days against a ~15.8 h
median run, so without this guard roughly one full fleet of in-flight runs would be killed every 4
days (measured 2026-08-06: median 15.8 h over the first 405 completions).

The guard FAILS OPEN: if the job's end time cannot be read, the worker claims anyway. The guard is an
optimization, not a correctness rule, and a scontrol hiccup must never idle the fleet.

No per-run stopwatch: within the guard, a run is left to finish. A slow-but-progressing run is never
killed near the finish; a run the walltime does kill anyway leaves its partial checkpoint on disk and
its marker is re-pended by requeue_orphans.py.

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root holding train.py), SWEEP_ID,
     WORKER_REQUIRED_HOURS (optional; hours of walltime a claim requires, default 20).
"""
import os
import re
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


# hours of remaining walltime a claim requires: the ~15.8 h median run plus margin for the tail
REQUIRED_SECONDS = float(os.environ.get("WORKER_REQUIRED_HOURS", "20")) * 3600


def job_end_epoch():
    """The job's end time as a unix epoch, or None when it cannot be determined.

    SLURM_JOB_END_TIME (already an epoch integer in the batch environment) first, `scontrol show job`
    EndTime as the fallback.
    """
    v = os.environ.get("SLURM_JOB_END_TIME")
    if v and v.isdigit():
        return int(v)
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"EndTime=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", out)
    if not m:
        return None
    return int(time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")))


JOB_END = job_end_epoch()   # fixed for the job's life; None = unknown -> the guard never fires


def enough_walltime():
    """Whether this job still has room for a whole run. Unknown end time claims anyway (fail open)."""
    if JOB_END is None:
        return True
    return (JOB_END - time.time()) >= REQUIRED_SECONDS


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
    """Claim-run-mark loop: keep taking new runs until the queue is empty or the job is running out
    of walltime."""
    # small startup jitter so the workers of one job do not all import torch at the same instant
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    end = "unknown" if JOB_END is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(JOB_END))
    log(f"started; job ends {end}; a claim needs {REQUIRED_SECONDS/3600:.0f} h of walltime left")
    n_done = n_fail = 0
    while True:
        # stop taking work this job cannot finish: the run would be killed at the walltime and
        # restarted from zero elsewhere, since this run writes no checkpoints
        if not enough_walltime():
            left = (JOB_END - time.time()) / 3600
            log(f"only {left:.1f} h of walltime left, less than the {REQUIRED_SECONDS/3600:.0f} h a "
                f"run needs; not claiming again. Exiting after {n_done} done / {n_fail} failed")
            return
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
