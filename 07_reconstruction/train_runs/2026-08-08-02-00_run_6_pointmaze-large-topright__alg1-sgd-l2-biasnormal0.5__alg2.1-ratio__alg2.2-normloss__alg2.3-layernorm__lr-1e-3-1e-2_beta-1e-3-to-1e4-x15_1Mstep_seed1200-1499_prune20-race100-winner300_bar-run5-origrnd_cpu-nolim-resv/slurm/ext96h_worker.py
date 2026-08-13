#!/usr/bin/env python
"""Worker for the run-6 96-hour extension sweep (ext96h): claim ONE fresh run and train it with
train96h.py until the job's walltime (or the 10M cap) — no checkpoints, no resume.

Effectively one-shot: the claim guard requires >= 90 h of remaining walltime, which only a
fresh job has, so each worker slot runs exactly one 96-hour run. The loop exists solely so a
worker whose run CRASHES in the first few hours can pick up another instead of idling four days.
A cleanly ended run (walltime or cap) exits 0 -> done/; anything else -> failed/.

Env: RUN_DIR, PROJ_DIR, SWEEP_ID, WORKER_REQUIRED_HOURS (default 90).
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
DATA = os.path.join(RUN, "data", SWEEP_ID)

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'


def log(msg):
    """Print a timestamped worker-tagged line, flushed (claim lines feed the orphan requeue)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp} worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


# a claim needs essentially a whole fresh job: runs are single 96-hour attempts
REQUIRED_SECONDS = float(os.environ.get("WORKER_REQUIRED_HOURS", "90")) * 3600


def job_end_epoch():
    """The job's end time as a unix epoch (for the trainer's walltime stop), or None."""
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


JOB_END = job_end_epoch()


def enough_walltime():
    """Whether a fresh 96-hour attempt still fits this job. Unknown end time claims anyway."""
    if JOB_END is None:
        return True
    return (JOB_END - time.time()) >= REQUIRED_SECONDS


def claim():
    """Atomically claim one pending config, approximately in run-id order."""
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
            names.remove(name)
    return None, None


def build_cmd(cfg):
    """The train96h.py argv: train.py's args + the adopted optimization switches + the wall."""
    args = [
        sys.executable, os.path.join(PROJ, "train96h.py"),
        f'--walltime_end_epoch={JOB_END or 0}',
        '--opt_polyak_foreach=True',
        '--opt_torch_reward=True',
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
    """Claim-run-mark; in practice one run per worker (the 90 h guard), retrying only after an
    early crash."""
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    end = "unknown" if JOB_END is None else time.strftime("%Y-%m-%dT%H:%M:%S",
                                                          time.localtime(JOB_END))
    log(f"started (96h single-attempt); job ends {end}")
    n_done = n_fail = 0
    while True:
        if not enough_walltime():
            left = 0 if JOB_END is None else (JOB_END - time.time()) / 3600
            log(f"only {left:.1f} h left, under the {REQUIRED_SECONDS/3600:.0f} h a fresh "
                f"attempt needs; exiting after {n_done} done / {n_fail} failed")
            return
        name, path = claim()
        if name is None:
            log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: {cfg['env_setup']} beta={cfg['beta']} seed={cfg['a_seed']} "
            f"cap={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        rc = subprocess.call(build_cmd(cfg), cwd=PROJ)
        dest = DONE if rc == 0 else FAILED
        n_done += (rc == 0)
        n_fail += (rc != 0)
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)} "
            f"(totals: {n_done} done, {n_fail} failed)")


if __name__ == "__main__":
    main()
