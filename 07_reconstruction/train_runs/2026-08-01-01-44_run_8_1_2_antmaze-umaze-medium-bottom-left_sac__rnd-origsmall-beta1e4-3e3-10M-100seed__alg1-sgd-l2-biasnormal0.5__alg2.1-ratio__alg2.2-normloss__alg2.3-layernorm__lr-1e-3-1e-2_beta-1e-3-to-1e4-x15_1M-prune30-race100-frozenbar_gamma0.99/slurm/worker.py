#!/usr/bin/env python
"""Local work-queue worker for train run 8.1.2, scoped by SWEEP_ID — two pools, walltime-guarded.

Loop: atomically claim a pending config from the pools listed in WORKER_POOLS (in that priority
order), run train.py, move the marker to done/ (rc 0) or failed/ (otherwise), repeat until every
claimable pool is empty. No per-run stopwatch (the 2026-07-06 decision): the only time limit is the
job's Slurm walltime.

Run-8.1.2 changes over the run-8.1 worker:
- TWO pending pools under one sweep: pending_1m/ (task S, 1M-step runs) and pending_10m/ (task R,
  10M-step baseline runs). WORKER_POOLS (space-separated, priority order) says which pool(s) this
  worker claims from: gpu cuda workers "pending_10m", gnolim cpu workers "pending_10m pending_1m"
  (10M first, 1M fallback — the user rule), cpu/nolim/jaguar03 workers "pending_1m" (default).
- WALLTIME GUARD: a pending_10m claim is allowed only when the job's remaining walltime covers a
  full 10M run — >= 90 h on cuda workers, >= 170 h on cpu workers (measured ~64-70 h / ~154 h plus
  margin). The job end time comes from SLURM_JOB_END_TIME (or scontrol as fallback) once at start;
  if neither is available the 10M pool is skipped with a loud log line (never a guessed claim).
  There are no checkpoints in this run: a run that cannot finish inside the job must not start.

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (package root with train.py), SWEEP_ID,
     WORKER_POOLS (optional, default "pending_1m"), WORKER_DEVICE (optional cpu / cuda override).
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
POOLS = os.environ.get("WORKER_POOLS", "pending_1m").split()
RUNNING = os.path.join(QUEUE, "running")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
DATA = os.path.join(RUN, "data", SWEEP_ID)  # train.py appends /local -> data/<sweep_id>/local/<id>.json

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'

# remaining-walltime the 10M pool requires, by this worker's device (hours -> seconds). The
# WORKER_REQUIRED_10M_HOURS override exists for the canary jobs only (short walltime, short runs):
# they set it to 0 so the pending_10m claim path is exercised end to end.
_default_hours = 90 if os.environ.get("WORKER_DEVICE") == "cuda" else 170
REQUIRED_10M_SECONDS = float(os.environ.get("WORKER_REQUIRED_10M_HOURS", _default_hours)) * 3600


def log(msg):
    """Print a timestamped worker-tagged line, flushed (so the slurm .log streams live and claim/finish
    times are recomputable from the log alone)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp} worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


def job_end_epoch():
    """The Slurm job's end time as a unix epoch, or None when it cannot be determined.
    SLURM_JOB_END_TIME (batch env, newer Slurm) first; `scontrol show job` EndTime as fallback."""
    # source 1: the env var is already an epoch integer, e.g. "1754537400"
    v = os.environ.get("SLURM_JOB_END_TIME")
    if v and v.isdigit():
        return int(v)
    # source 2: scontrol prints EndTime=2026-08-05T06:59:00 (cluster-local time)
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


JOB_END = job_end_epoch()  # fixed for the job's life; None = unknown -> 10M pool never claimable
_warned_no_end = False


def pool_claimable(pool):
    """Whether this worker may claim from `pool` RIGHT NOW (the 10M walltime guard)."""
    global _warned_no_end
    if pool != "pending_10m":
        return True
    # an explicit WORKER_REQUIRED_10M_HOURS=0 (canary jobs, portal debugging) bypasses the guard
    # entirely, including the unknown-job-end block below
    if REQUIRED_10M_SECONDS <= 0:
        return True
    if JOB_END is None:
        if not _warned_no_end:
            log("job end time unknown (no SLURM_JOB_END_TIME, scontrol failed): "
                "pending_10m is NOT claimable by this worker")
            _warned_no_end = True
        return False
    remaining = JOB_END - time.time()
    return remaining >= REQUIRED_10M_SECONDS


def claim():
    """Atomically claim one pending config from the first claimable non-empty pool in POOLS order,
    approximately in run-id order. Returns (name, running_path) or (None, None) when nothing is
    claimable. `all_empty` distinguishes 'nothing left anywhere' from 'blocked by the guard'."""
    any_blocked = False
    for pool in POOLS:
        pending = os.path.join(QUEUE, pool)
        try:
            names = sorted(os.listdir(pending))
        except FileNotFoundError:
            continue
        if not names:
            continue
        if not pool_claimable(pool):
            any_blocked = True
            continue
        # Sort pending names (zero-padded ids -> lexical order == id order) and pick randomly among
        # only the FIRST 32: workers stay inside the earliest-id window so early seeds FINISH first,
        # while the random pick keeps the rename-collision protection (run-8.1 convention).
        while names:
            name = random.choice(names[:32])
            try:
                os.rename(os.path.join(pending, name), os.path.join(RUNNING, name))
                return name, os.path.join(RUNNING, name), False
            except OSError:
                names.remove(name)  # another worker won this file; retry within the refreshed window
    return None, None, any_blocked


def build_cmd(cfg):
    """Build the train.py argv for one claimed config; JSON lands in data/<sweep_id>/local/<id>.json.
    The per-config `params` and the global `fixed` args (which carry this task's total_timesteps)
    pass through verbatim; the per-node device override is appended last."""
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
    # per-config arm knobs, then the global fixed args (incl. this task's total_timesteps)
    for k, v in cfg.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    # per-node device override: WORKER_DEVICE=cuda (GPU jobs) / cpu (CPU jobs) appended LAST so
    # argparse's last-occurrence-wins beats the queue's device-agnostic --device=cpu from `fixed`.
    if os.environ.get("WORKER_DEVICE"):
        args.append(f"--device={os.environ['WORKER_DEVICE']}")
    return args


def main():
    """Claim-run-mark loop over this worker's pools until nothing is left to claim."""
    # small startup jitter so workers don't all import torch / build the env at the same instant
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    log(f"pools={POOLS} device={os.environ.get('WORKER_DEVICE', '(queue default)')} "
        f"job_end={'unknown' if JOB_END is None else time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(JOB_END))}")
    n_done = n_fail = 0
    while True:
        name, path, blocked = claim()
        if name is None:
            if blocked:
                # 10M items remain but this job can no longer finish one: exit rather than idle-spin
                log(f"only walltime-blocked items remain; exiting after {n_done} done / {n_fail} failed")
            else:
                log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: task={cfg.get('task', '?')} {cfg['env_setup']} {cfg['arm']} "
            f"beta={cfg['beta']} seed={cfg['a_seed']} steps={cfg['fixed']['total_timesteps']}")
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
