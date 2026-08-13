#!/usr/bin/env python
"""ONE-SHOT work-queue worker for the run-6 4M extension sweep (ext4m): each worker claims AT
MOST ONE run, finishes it, and exits — it never claims a second run (user rule 2026-08-13).

Why one-shot: the cpu partition caps a job at 96 hours and a 4M run takes 60–76 h, so a worker
that claimed a second run late in its job's life would hand that run a fraction of the walltime
it needs. Instead every claim owns a FULL fresh job's walltime: the job (srun --wait=0) ends when
the last of its workers finishes its single run, and replacement jobs already PENDING in the
Slurm queue start in its place and take the next runs. ext4m_plan_jobs.py keeps that pending
ladder stocked, capped at this submitter's share of the workload.

The trainer's checkpoint suspend stays as a safety net only (exit code 3 -> the marker returns to
pending with its checkpoint; the next fresh job resumes it). A run on any current node class fits
one job's walltime, so in normal operation a worker's run simply completes.

Env: RUN_DIR, PROJ_DIR, SWEEP_ID.
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
CKPTS = os.path.join(RUN, "checkpoints", SWEEP_ID)
SUSPEND_EXIT_CODE = 3

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'


def log(msg):
    """Print a timestamped worker-tagged line, flushed (claim lines feed the orphan requeue)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp} worker {WID} sweep={SWEEP_ID}] {msg}", flush=True)


def job_end_epoch():
    """The job's end time as a unix epoch (for the trainer's suspend safety net), or None."""
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


def claim():
    """Atomically claim one pending config, approximately in run-id order (the first-32 window
    keeps early seeds finishing first while avoiding same-file collisions)."""
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


def build_cmd(cfg, job_end):
    """The train4m.py argv for one claimed config: train.py's args plus the checkpoint args."""
    width = len(str(cfg["run_total"]))
    ckpt_dir = os.path.join(CKPTS, f'{cfg["run_id"]:0{width}d}')
    args = [
        sys.executable, os.path.join(PROJ, "train4m.py"),
        f'--ckpt_dir={ckpt_dir}',
        '--ckpt_every=500000',
        f'--suspend_end_epoch={job_end or 0}',
        '--buffer_tail=100000',
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
    """Claim ONE run, finish it, mark it, exit. The job ends when every worker has done this."""
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    job_end = job_end_epoch()
    end = "unknown" if job_end is None else time.strftime("%Y-%m-%dT%H:%M:%S",
                                                          time.localtime(job_end))
    log(f"started (one-shot); job ends {end}")
    name, path = claim()
    if name is None:
        log("queue empty; exiting with no claim")
        return
    with open(path) as fh:
        cfg = json.load(fh)
    log(f"claimed {name} :: {cfg['env_setup']} beta={cfg['beta']} seed={cfg['a_seed']} "
        f"steps={cfg['fixed']['total_timesteps']}")
    t0 = time.time()
    rc = subprocess.call(build_cmd(cfg, job_end), cwd=PROJ)
    # 0 -> done; 3 -> suspended at a checkpoint (safety net), back to pending; else failed
    if rc == 0:
        dest = DONE
    elif rc == SUSPEND_EXIT_CODE:
        dest = PENDING
    else:
        dest = FAILED
    try:
        os.rename(path, os.path.join(dest, name))
    except OSError:
        pass
    log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)}; one-shot worker "
        f"exiting")


if __name__ == "__main__":
    main()
