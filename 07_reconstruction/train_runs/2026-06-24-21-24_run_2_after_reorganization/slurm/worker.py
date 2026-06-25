#!/usr/bin/env python
"""Local work-queue worker — replaces `wandb agent` for config distribution.

Each worker repeatedly:
  1. atomically claims one pending config by renaming queue/pending/X -> queue/running/X
     (os.rename is atomic on the shared filesystem; the loser of a race just tries another file),
  2. runs train.py with that config,
  3. renames the file to queue/done/X (success) or queue/failed/X (failure / timeout),
and loops until no pending configs remain.

8 workers run per slurm job (srun ntasks=8); 64 jobs => 512 workers but only 64 slurm IDs.
This run is local-only: config distribution never touches wandb and train.py logs to its own per-run
JSON (use_wandb=False). Each worker passes the config's run_id/run_total through so train.py names the
JSON by its sweep id (see .claude/rules/run-id-and-logging.md).

Env: RUN_DIR (train_runs/<run>/), PROJ_DIR (the package root containing train.py).
"""
import os
import sys
import json
import time
import random
import subprocess

RUN = os.environ["RUN_DIR"]
PROJ = os.environ.get("PROJ_DIR", "/p/rlprojects/RND/07_reconstruction")
QUEUE = os.path.join(RUN, "queue")
PENDING = os.path.join(QUEUE, "pending")
RUNNING = os.path.join(QUEUE, "running")
DONE = os.path.join(QUEUE, "done")
FAILED = os.path.join(QUEUE, "failed")
DATA = os.path.join(RUN, "data")
PER_RUN_TIMEOUT = 40 * 60 * 60  # 40 h cap: above the real ~14 h/run (even on a slow node) but below the
                                # 48 h slurm job limit, so it only ever fires on a genuinely hung run.
                                # (Was 3 h, which wrongly killed every real run at the 3 h mark.)

WID = f'{os.environ.get("SLURM_JOB_ID", "x")}.{os.environ.get("SLURM_PROCID", "0")}.{os.getpid()}'


def log(msg):
    print(f"[worker {WID}] {msg}", flush=True)


def claim():
    """Atomically claim one pending config. Returns (name, running_path) or (None, None) when empty."""
    try:
        names = os.listdir(PENDING)
    except FileNotFoundError:
        return None, None
    random.shuffle(names)  # spread workers across files to cut claim collisions
    for name in names:
        try:
            os.rename(os.path.join(PENDING, name), os.path.join(RUNNING, name))
            return name, os.path.join(RUNNING, name)
        except OSError:
            continue  # another worker won the race; try the next file
    return None, None


def build_cmd(cfg):
    """Build the train.py argv for one claimed config, passing its sweep id through for id-based logging."""
    # mode is always "local" in this run (no wandb); use_wandb stays False so train.py never calls wandb.init
    mode = cfg["z_logging_mode"]
    use_wandb = (mode == "wandb_full")
    args = [
        sys.executable, os.path.join(PROJ, "train.py"),
        f'--a_seed={cfg["a_seed"]}',
        f'--g_algo_beta={cfg["g_algo_beta"]}',
        f'--z_logging_mode={mode}',
        f'--use_wandb={use_wandb}',
        f'--local_log_dir={DATA}',
        f'--run_id={cfg["run_id"]}',        # sweep position -> id-based per-run JSON filename in train.py
        f'--run_total={cfg["run_total"]}',  # sweep size -> the "/total" of the run's id
    ]
    # fixed args (env, timesteps, eval cadence, ...) are passed verbatim from the queue config
    for k, v in cfg["fixed"].items():
        args.append(f"--{k}={v}")
    return args


def main():
    # startup jitter so the wandb_full runs across 512 workers don't all wandb.init at once
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "75"))))
    n_done = n_fail = 0
    while True:
        name, path = claim()
        if name is None:
            log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return
        with open(path) as fh:
            cfg = json.load(fh)
        log(f"claimed {name} :: {cfg['g_algo_beta']} {cfg['z_logging_mode']} seed={cfg['a_seed']}")
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
        if rc == 0:
            n_done += 1
        else:
            n_fail += 1
        log(f"{name} rc={rc} in {time.time()-t0:.0f}s -> {os.path.basename(dest)} "
            f"(running totals: {n_done} done, {n_fail} failed)")


if __name__ == "__main__":
    main()
