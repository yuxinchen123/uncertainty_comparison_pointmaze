#!/usr/bin/env python
"""Build the work queue for the run-6 4M-step extension sweep (ext4m): three configurations x
300 fresh seeds x 4,000,000 steps = 900 resumable runs. Design: slurm/EXT4M_DESIGN.md.

The three configurations (each at its exact published parameters):
- run-5 original RND (Adam 1e-4, mse-mean readout, reward norm ON, beta=1000) — re-run at 4M;
  parameters verbatim from run 5's own queue markers (test_ext4m_queue_convention.py pins this
  against a run-5 marker on disk).
- algorithm 2.3 at its best run-6 configuration (plain SGD lr 0.01, beta=30) — parameter dict
  imported from build_queue.py (byte-identical to the 1M sweep's).
- the best ground-truth bonus: gt_position_velocity, bonus min(1, 1/sqrt(n)), beta=1 —
  parameters verbatim from the run-3.2.2 oracle re-run's markers.

Seeds are a_seed 1500-1799 — disjoint from run 5 (600-899), the addendum (900-1199) and the run-6
1M sweep (1200-1499). Seed index OUTERMOST: index i owns run ids [3i .. 3i+2]. No truncation race:
all 900 runs go to completion.
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue as bq   # noqa: E402  (the 1M sweep's builder: ALG1_PARAMS + ARM_EXTRAS)

SEED_INDICES = list(range(0, 300))
SEED_OFFSET = 1500          # seed_index 0 -> a_seed 1500; index 299 -> a_seed 1799
ENV_SETUP = bq.ENV_SETUP    # initial_single_large_pointmaze_max_400, the run-5 task verbatim
STEPS = 4000000

# run-5 original RND parameters, VERBATIM from run 5's queue markers (pinned by unit test)
RUN5_ORIGRND_PARAMS = {
    "rnd_optimizer": "adam", "rnd_bonus_readout": "mse_mean", "rnd_lr": "0.0001",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "True", "rnd_reward_norm_gamma": "0.99",
    "rnd_bias_init": "zero", "rnd_weight_init": "orthogonal",
}

# algorithm 2.3 at lr 0.01: the 1M sweep's dict, imported so it cannot drift
ALG23_PARAMS = dict(bq.ALG1_PARAMS)
ALG23_PARAMS.update(bq.ARM_EXTRAS["alg2.3"])
ALG23_PARAMS["rnd_lr"] = "0.01"

# the fixed per-run args of the 1M sweep with only the step budget changed
FIXED_COMMON = dict(bq.FIXED_COMMON)
FIXED_COMMON["total_timesteps"] = STEPS

# the three configurations in their fixed per-seed order (index i owns ids 3i..3i+2)
CONFIGS = [
    {"env_setup": ENV_SETUP, "arm": "run5-origrnd", "algorithm": "rnd_next_state",
     "beta": "1000", "params": dict(RUN5_ORIGRND_PARAMS),
     "config_key": f"{ENV_SETUP}|run5-origrnd|lr0.0001|b1000",
     "label": "PointMaze_Large-topright_run5origrnd_adam1e-4_b1000_4M"},
    {"env_setup": ENV_SETUP, "arm": "alg2.3", "algorithm": "rnd_next_state",
     "beta": "30", "params": dict(ALG23_PARAMS),
     "config_key": f"{ENV_SETUP}|alg2.3|lr0.01|b30",
     "label": "PointMaze_Large-topright_alg2.3_lr0.01_b30_4M"},
    {"env_setup": ENV_SETUP, "arm": "gt-position-velocity", "algorithm": "gt_position_velocity",
     "beta": "1", "params": {"visit_count_decay": "-0.5"},
     "config_key": f"{ENV_SETUP}|gt_position_velocity|b1",
     "label": "PointMaze_Large-topright_gtposvel_sqrtn_b1_4M"},
]

RUN_TOTAL = len(SEED_INDICES) * len(CONFIGS)   # 300 * 3 = 900


def main():
    """Write the 900 marker JSONs into queue/<sweep_id>/pending/, seed index outermost."""
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators use for_collaborator/")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    args = p.parse_args()
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "data", args.sweep_id, "local"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "checkpoints", args.sweep_id), exist_ok=True)
    width = len(str(RUN_TOTAL))

    # seed index OUTERMOST, configuration inner: index i owns ids [3i .. 3i+2]
    run_id = 0
    for seed_index in SEED_INDICES:
        seed = SEED_OFFSET + seed_index
        for spec in CONFIGS:
            cfg = {
                "sweep_id": args.sweep_id, "run_id": run_id, "run_total": RUN_TOTAL,
                "pool": "pending",
                "env_setup": spec["env_setup"], "algorithm": spec["algorithm"],
                "arm": spec["arm"], "beta": spec["beta"],
                "a_seed": seed, "seed_index": seed_index,
                "config_key": spec["config_key"],
                "params": spec["params"],
                "fixed": dict(FIXED_COMMON),
            }
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{spec['label']}_seed{seed}.json"
            with open(os.path.join(sweep_queue, "pending", name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    assert run_id == RUN_TOTAL

    # manifest row in the run folder's data/SWEEPS.md (created by the 1M sweep's builder)
    with open(os.path.join(RUN_DIR, "data", "SWEEPS.md"), "a") as fh:
        fh.write(f"| {args.sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEED_INDICES)}seeds "
                 f"x {STEPS} steps, checkpointed/resumable | active |\n")
    print(f"queue built: {RUN_TOTAL} entries in {sweep_queue}/pending/ "
          f"({len(CONFIGS)} configurations x {len(SEED_INDICES)} seeds x {STEPS} steps)")


if __name__ == "__main__":
    main()
