#!/usr/bin/env python
"""Build the work queue for the run-6 96-HOUR extension sweep (ext96h): the same three
configurations and 300 fresh seeds as the retired ext4m sweep, but each run is a FRESH single
attempt with a 10,000,000-step upper bound, ending at its job's 96-hour walltime (no
checkpoints, no resume — the user's final design 2026-08-13; see slurm/EXT96H_DESIGN.md).

Configurations (parameters verbatim from their sources, pinned by test_ext96h_queue_convention):
run-5 original RND (Adam 1e-4, beta=1000); algorithm 2.3 (SGD lr 0.01, beta=30);
gt_position_velocity min(1, 1/sqrt(n)) (beta=1). Seeds a_seed 1500-1799, seed index OUTERMOST
(index i owns ids 3i..3i+2).
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue as bq   # noqa: E402

SEED_INDICES = list(range(0, 300))
SEED_OFFSET = 1500
ENV_SETUP = bq.ENV_SETUP
STEPS = 10000000            # upper bound only: no 96-hour job reaches it (~6.5M on the fastest)

RUN5_ORIGRND_PARAMS = {
    "rnd_optimizer": "adam", "rnd_bonus_readout": "mse_mean", "rnd_lr": "0.0001",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "True", "rnd_reward_norm_gamma": "0.99",
    "rnd_bias_init": "zero", "rnd_weight_init": "orthogonal",
}
ALG23_PARAMS = dict(bq.ALG1_PARAMS)
ALG23_PARAMS.update(bq.ARM_EXTRAS["alg2.3"])
ALG23_PARAMS["rnd_lr"] = "0.01"

FIXED_COMMON = dict(bq.FIXED_COMMON)
FIXED_COMMON["total_timesteps"] = STEPS

CONFIGS = [
    {"env_setup": ENV_SETUP, "arm": "run5-origrnd", "algorithm": "rnd_next_state",
     "beta": "1000", "params": dict(RUN5_ORIGRND_PARAMS),
     "config_key": f"{ENV_SETUP}|run5-origrnd|lr0.0001|b1000",
     "label": "PointMaze_Large-topright_run5origrnd_adam1e-4_b1000_96h"},
    {"env_setup": ENV_SETUP, "arm": "alg2.3", "algorithm": "rnd_next_state",
     "beta": "30", "params": dict(ALG23_PARAMS),
     "config_key": f"{ENV_SETUP}|alg2.3|lr0.01|b30",
     "label": "PointMaze_Large-topright_alg2.3_lr0.01_b30_96h"},
    {"env_setup": ENV_SETUP, "arm": "gt-position-velocity", "algorithm": "gt_position_velocity",
     "beta": "1", "params": {"visit_count_decay": "-0.5"},
     "config_key": f"{ENV_SETUP}|gt_position_velocity|b1",
     "label": "PointMaze_Large-topright_gtposvel_sqrtn_b1_96h"},
]

RUN_TOTAL = len(SEED_INDICES) * len(CONFIGS)   # 900


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
    width = len(str(RUN_TOTAL))
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
    with open(os.path.join(RUN_DIR, "data", "SWEEPS.md"), "a") as fh:
        fh.write(f"| {args.sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEED_INDICES)}seeds"
                 f" x 96h (10M-step cap), fresh single attempts | active |\n")
    print(f"queue built: {RUN_TOTAL} entries in {sweep_queue}/pending/")


if __name__ == "__main__":
    main()
