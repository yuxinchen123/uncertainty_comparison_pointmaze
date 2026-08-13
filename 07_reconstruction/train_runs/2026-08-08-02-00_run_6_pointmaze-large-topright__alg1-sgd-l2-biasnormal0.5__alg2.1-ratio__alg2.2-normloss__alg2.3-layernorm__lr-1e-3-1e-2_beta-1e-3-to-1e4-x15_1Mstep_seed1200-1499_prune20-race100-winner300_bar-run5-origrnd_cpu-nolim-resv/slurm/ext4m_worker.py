#!/usr/bin/env python
"""Work-queue worker for the run-6 4M extension sweep (ext4m): claims resumable 4M-step runs
in a loop for the job's FULL 96-hour walltime, finishing as many as it can (the user's final
design, 2026-08-13, replacing the brief one-shot variant).

- runs train4m.py (checkpoint every 0.5M steps, resume-on-start; slurm/EXT4M_DESIGN.md) with
  the throughput-research switches applied (APPLIED_CHANGES.md in the research folder: the
  foreach polyak update and the torch-only reward combine — bit-exact on every configuration);
- exit code 3 from the trainer means SUSPENDED at a checkpoint (the job's walltime could not
  fit another 0.5M-step chunk): the marker goes BACK TO PENDING so the next claimer resumes it;
- the walltime guard needs only ONE chunk of walltime (default 12 h), not a whole run — the
  trainer itself suspends before the wall, so late claims still make durable progress.

Env: RUN_DIR, PROJ_DIR, SWEEP_ID, WORKER_REQUIRED_HOURS (default 12).
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


# a claim needs walltime for ONE 0.5M-step chunk plus margin, not a whole 4M run
REQUIRED_SECONDS = float(os.environ.get("WORKER_REQUIRED_HOURS", "12")) * 3600


def job_end_epoch():
    """The job's end time as a unix epoch, or None when it cannot be determined."""
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
    """Whether this job still has room for one more chunk. Unknown end time claims anyway."""
    if JOB_END is None:
        return True
    return (JOB_END - time.time()) >= REQUIRED_SECONDS


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


def build_cmd(cfg):
    """The train4m.py argv for one claimed config: train.py's args plus the checkpoint args."""
    width = len(str(cfg["run_total"]))
    ckpt_dir = os.path.join(CKPTS, f'{cfg["run_id"]:0{width}d}')
    args = [
        sys.executable, os.path.join(PROJ, "train4m.py"),
        f'--ckpt_dir={ckpt_dir}',
        '--ckpt_every=500000',
        f'--suspend_end_epoch={JOB_END or 0}',
        '--buffer_tail=100000',
        # the throughput research's bit-exact switches (research folder APPLIED_CHANGES.md)
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
    """Claim-run-mark loop; suspended runs (rc=3) return to pending with their checkpoint."""
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    end = "unknown" if JOB_END is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(JOB_END))
    log(f"started; job ends {end}; a claim needs {REQUIRED_SECONDS/3600:.0f} h of walltime left")
    n_done = n_fail = n_susp = 0
    while True:
        if not enough_walltime():
            left = (JOB_END - time.time()) / 3600
            log(f"only {left:.1f} h of walltime left, less than the {REQUIRED_SECONDS/3600:.0f} h "
                f"one chunk needs; exiting after {n_done} done / {n_susp} suspended / {n_fail} failed")
            return
        name, path = claim()
        if name is None:
            log(f"queue empty; exiting after {n_done} done / {n_susp} suspended / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: {cfg['env_setup']} beta={cfg['beta']} seed={cfg['a_seed']} "
            f"steps={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        rc = subprocess.call(build_cmd(cfg), cwd=PROJ)
        # 0 -> done; 3 -> suspended at a checkpoint, back to pending for the next claimer; else failed
        if rc == 0:
            dest = DONE
            n_done += 1
        elif rc == SUSPEND_EXIT_CODE:
            dest = PENDING
            n_susp += 1
        else:
            dest = FAILED
            n_fail += 1
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)} "
            f"(running totals: {n_done} done, {n_susp} suspended, {n_fail} failed)")


if __name__ == "__main__":
    main()
