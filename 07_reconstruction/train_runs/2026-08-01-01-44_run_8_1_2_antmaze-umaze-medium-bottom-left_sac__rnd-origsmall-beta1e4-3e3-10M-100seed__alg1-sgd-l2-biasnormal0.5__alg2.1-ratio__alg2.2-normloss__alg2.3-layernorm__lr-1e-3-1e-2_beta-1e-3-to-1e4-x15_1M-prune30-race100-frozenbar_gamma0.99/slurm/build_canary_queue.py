#!/usr/bin/env python
"""Build the CANARY queue for train run 8.1.2 (the canary phase of the submission skills): SHORT runs
of the real worker path, in the run's real TWO-POOL layout, under a sweep id of their OWN so canary
records never mix with the racing data (the stage-1 controller reads only the real sweep id and never
sees them).

What it builds (default --count 40, split roughly 32 task-S / 8 task-R):
- pending_1m/  : the task-S claim path — all 4 arms (alg1, alg2.1, alg2.2, alg2.3) x both env setups
                 at learning rate 0.01 and bonus weight 1 = 8 specs, repeated over as many seeds as
                 the count needs (seed outermost).
- pending_10m/ : the task-R claim path — the 2 baseline specs (run-8.1 winner per env, orig-small
                 stack) with the SAME short length, repeated over seeds. They exist so the gpu and
                 gnolim workers exercise the pending_10m claim path end to end.

Every canary run is SHORT: total_timesteps 60000, eval_freq 20000, n_eval_episodes 10 — long enough
for two eval snapshots, a per-episode history, and a speed/memory signal per node class.

IMPORTANT — the canary worker jobs must be submitted with WORKER_REQUIRED_10M_HOURS=0, e.g.
  sbatch --export=ALL,SWEEP_ID=<canary id>,WORKER_REQUIRED_10M_HOURS=0 ... worker_gpu_10m_1x2.slurm
Without it worker.py's walltime guard refuses every pending_10m claim on a short canary job (the
guard requires >= 90 h of remaining walltime on cuda, >= 170 h on cpu, sized for real 10M runs), and
the task-R claim path would go untested.

Canary outputs land in data/<canary_sweep_id>/local/ (the worker derives that path from SWEEP_ID), and
the sweep is recorded in data/SWEEPS.md with status "canary".

Usage: python build_canary_queue.py --sweep_id <canary sweep id> [--count 40]
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue  # noqa: E402  (CONFIGS / BASELINE_CONFIGS / label / config_key — source of truth)

CANARY_STEPS = 60000        # ~1/17 of a task-S run: two eval snapshots plus a speed/memory signal
CANARY_EVAL_FREQ = 20000
CANARY_EVAL_EPISODES = 10
CANARY_LR = "0.01"          # the task-S canary picks one learning rate ...
CANARY_BETA = "1"           # ... and one middle bonus weight, so the 8 specs differ only by arm/env
TASK_S_SHARE = 0.8          # of --count: 32 of 40 task-S markers, the rest task-R


def canary_task_s_specs():
    """The 8 task-S canary specs: every arm x every env setup at CANARY_LR / CANARY_BETA."""
    return [c for c in build_queue.CONFIGS
            if c["lr"] == CANARY_LR and c["beta"] == CANARY_BETA]


def canary_task_r_specs():
    """The 2 task-R canary specs: the real baseline configs (run only for CANARY_STEPS here)."""
    return list(build_queue.BASELINE_CONFIGS)


def split_counts(count):
    """(task-S markers, task-R markers) from the total marker count, ~80/20.
      before: count = 40   -> after: (32, 8)
      before: count = 10   -> after: (8, 2)"""
    n_s = int(round(count * TASK_S_SHARE))
    return n_s, count - n_s


def markers(specs, n_markers):
    """[(spec, seed)] of length n_markers, seed OUTERMOST over the spec list (so a small count still
    covers every spec once before repeating a spec at the next seed).
      before: specs = [A, B, C], n_markers = 5
      after : [(A,0), (B,0), (C,0), (A,1), (B,1)]"""
    out = []
    seed = 0
    while len(out) < n_markers:
        for spec in specs:
            if len(out) == n_markers:
                break
            out.append((spec, seed))
        seed += 1
    return out


def write_manifest_row(sweep_id, count, n_s, n_r):
    """Append this canary sweep's row to data/SWEEPS.md (create with the same header build_queue uses
    when the file is missing), with status "canary" so it is never mistaken for racing data."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder (train run 8.1.2)\n\n"
                     "| sweep_id | runs | layout | status |\n|---|---|---|---|\n")
    with open(manifest, "a") as fh:
        fh.write(f"| {sweep_id} | {count} | canary: {n_s} task-S + {n_r} task-R markers, all at "
                 f"{CANARY_STEPS} steps | canary |\n")


def main():
    """Write the canary markers into queue/<canary_sweep_id>/{pending_1m,pending_10m}/."""
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="the CANARY sweep id, e.g. <ts>_run812-canary")
    p.add_argument("--count", type=int, default=40, help="total canary markers (~80%% task S)")
    args = p.parse_args()
    n_s, n_r = split_counts(args.count)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    # the short-run overrides on top of the real fixed args (device stays queue-neutral)
    fixed = dict(build_queue.FIXED_COMMON, total_timesteps=CANARY_STEPS,
                 eval_freq=CANARY_EVAL_FREQ, n_eval_episodes=CANARY_EVAL_EPISODES)
    plan = ([("pending_1m", "S", spec, seed) for spec, seed in markers(canary_task_s_specs(), n_s)]
            + [("pending_10m", "R", spec, seed) for spec, seed in markers(canary_task_r_specs(), n_r)])
    width = len(str(args.count))
    for run_id, (pool, task, spec, seed) in enumerate(plan):
        cfg = {
            "sweep_id": args.sweep_id, "run_id": run_id, "run_total": args.count,
            "task": task, "pool": pool,
            "env_setup": spec["env_setup"], "algorithm": spec["algorithm"],
            "arm": spec["arm"], "beta": spec["beta"], "a_seed": seed,
            "config_key": build_queue.config_key(spec),
            "params": dict(spec["params"]),
            "fixed": fixed,
        }
        name = f"{run_id:0{width}d}_of_{args.count}_{build_queue.label(spec)}_seed{seed}.json"
        with open(os.path.join(sweep_queue, pool, name), "w") as fh:
            json.dump(cfg, fh)
    write_manifest_row(args.sweep_id, args.count, n_s, n_r)
    print(f"canary queue built: {n_s} task-S markers in pending_1m/, {n_r} task-R markers in "
          f"pending_10m/ under {sweep_queue} (all {CANARY_STEPS} steps)")
    print("submit the canary worker jobs with WORKER_REQUIRED_10M_HOURS=0, e.g.\n"
          f"  sbatch --export=ALL,SWEEP_ID={args.sweep_id},WORKER_REQUIRED_10M_HOURS=0 "
          "--partition=gpu --time=00:40:00 --job-name=r812can "
          f"{os.path.join(HERE, 'worker_gpu_10m_1x2.slurm')}")


if __name__ == "__main__":
    main()
