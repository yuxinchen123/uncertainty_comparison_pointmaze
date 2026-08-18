#!/usr/bin/env python
"""RL-pilot queue builder (adapted from run-8.1's build_queue.py, trimmed to four configs).

60 runs = 4 configs x 15 seeds (0-14), fixed grid, no pruning:
  coinflip_count at beta 0.3 / 1 / 3, and gt_position_velocity at beta 1 (the in-pilot
  control; run-8.1's winner at this env), all on PointMaze_Large-v3_start_bottom_left at
  run-8.1's fixed knobs (1M steps, eval every 50k, 100 eval episodes).
Seed OUTERMOST, config inner: seed s owns ids [4*s .. 4*s+3] in CONFIGS order (the shared
work-queue convention; slurm/worker.py claims in approximate id order).

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python build_queue.py --sweep_id <id>
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

ENV = "PointMaze_Large-v3_start_bottom_left"
CONFIGS = [
    {"env_setup": ENV, "algorithm": "coinflip_count", "beta": 0.3, "params": {}},
    {"env_setup": ENV, "algorithm": "coinflip_count", "beta": 1.0, "params": {}},
    {"env_setup": ENV, "algorithm": "coinflip_count", "beta": 3.0, "params": {}},
    {"env_setup": ENV, "algorithm": "gt_position_velocity", "beta": 1.0, "params": {}},
]
SEEDS = list(range(15))
RUN_TOTAL = len(CONFIGS) * len(SEEDS)

FIXED = {
    "total_timesteps": 1000000, "eval_freq": 50000, "n_eval_episodes": 100,
    "eval_standalone": "False", "log_distance": "False", "device": "cpu",
    "rnd_obs_norm": "True", "rnd_distance": "mse", "rnd_output_dim": 128, "n_predictors": 1,
    "use_wandb": "False",
}


def config_key(cfg_spec) -> str:
    """Canonical grouping key env_setup|algorithm|beta (the run-8.1 convention)."""
    return f"{cfg_spec['env_setup']}|{cfg_spec['algorithm']}|{cfg_spec['beta']:g}"


def label(cfg_spec) -> str:
    """Short filename fragment for a config."""
    return f"{cfg_spec['algorithm']}_b{cfg_spec['beta']:g}"


def write_manifest_row(sweep_id):
    """Append this sweep's row to data/SWEEPS.md (create with a header if missing)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder (11_decay_rate RL pilot)\n\n"
                     "4 configs x 15 seeds, fixed grid, no pruning.\n\n"
                     "| sweep_id | runs | configs x seeds x steps | status |\n|---|---|---|---|\n")
    with open(manifest, "a") as fh:
        fh.write(f"| {sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEEDS)}seeds x "
                 f"{FIXED['total_timesteps']} | active |\n")


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/."""
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators use for_collaborator/ if present")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    args = p.parse_args()
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    for seed in SEEDS:
        for cfg_spec in CONFIGS:
            cfg = {
                "sweep_id": args.sweep_id, "run_id": run_id, "run_total": RUN_TOTAL,
                "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
                "beta": cfg_spec["beta"], "a_seed": seed,
                "config_key": config_key(cfg_spec),
                "params": cfg_spec["params"], "fixed": dict(FIXED),
            }
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{label(cfg_spec)}_seed{seed}.json"
            with open(os.path.join(pending, name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    write_manifest_row(args.sweep_id)
    print(f"queue built: {run_id} entries under {pending}")


if __name__ == "__main__":
    main()
