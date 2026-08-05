#!/usr/bin/env python
"""Build the work queue for the Adam learning-rate 1e-3 addendum to train run 5 and train run 1.2.

One sweep, one pool (pending/): 45 configurations x 100 seeds = 4,500 runs at 1,000,000 env steps.
The 45 configurations = 3 environments x 15 bonus weights. Every configuration is the SAME stack:
the train-run-5 "original-small" RND (Adam, mean-squared-error-over-dimensions readout, LeakyReLU
0.2, predictor one block deeper, 6400 env-step state-normalization warm-up, reward normalization on)
with ONE knob changed from the executed runs — the predictor's Adam learning rate is 1e-3 instead of
1e-4. Nothing else moves, so each environment's column is directly comparable to the Adam 1e-4
column already in the writeup.

The three environments and where their Adam 1e-4 comparison comes from:
- initial_single_large_pointmaze_max_400 -> train run 5 (writeup subsection "Train run 5"), the
  PointMaze Large top-right task at discount 0.999, 400-step episodes, raw sparse reward;
- AntMaze_UMaze-v5_start_bottom_left     -> train run 1.2 (writeup subsubsection "Train run 1.2");
- AntMaze_Medium-v5_start_bottom_left    -> train run 1.2.

Seeds are indexed 0..99 and mapped per environment: PointMaze runs seeds 900-999 (fresh, disjoint
from train run 5's 600-899 and from every earlier PointMaze run), AntMaze runs seeds 0-99 (the seed
space train runs 1.1 and 1.2 used on those two environments). Seed index is OUTERMOST: index i owns
run ids [45*i .. 45*i+44] in the fixed CONFIGS order, so early seeds finish first across every
configuration and the truncation controller can decide wave by wave.

config_key = "<env_setup>|<arm>|lr<lr>|b<beta>" — the same 4-field shape train run 1.2 uses, with
arm "origsmall" and lr 0.001, so keys from the two runs line up if the data is ever pooled.
The unit test slurm/test_run_queue_convention.py pins the ordering, ids, and counts.
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

SEED_INDICES = list(range(0, 100))   # 100 seeds per configuration, no racing target above this

# the 15 bonus weights of train run 1.2, swept identically on all three environments (the strings
# reach train.py verbatim so the queue records exactly what was asked for)
BETAS = ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30",
         "100", "300", "1000", "3000", "10000"]

# The train-run-5 "original-small" stack, verbatim from that run's build_queue.py ORIGSMALL_PARAMS,
# with the ONE changed knob: rnd_lr 0.0001 -> 0.001. Everything else is byte-identical, which is what
# makes this an addendum column rather than a new arm.
ORIGSMALL_ADAM_LR1E3_PARAMS = {
    "rnd_optimizer": "adam", "rnd_bonus_readout": "mse_mean", "rnd_lr": "0.001",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "True", "rnd_reward_norm_gamma": "0.99",
    "rnd_bias_init": "zero", "rnd_weight_init": "orthogonal",
}

ARM = "origsmall"     # the arm tag written into every config_key and every queue marker
LR = "0.001"          # the swept-in learning rate (constant across this run; it IS the change)

# environment order used by every table and plot of this run: the train-run-5 environment first,
# then the two train-run-1.2 environments in their writeup order
ENV_SETUPS = [
    "initial_single_large_pointmaze_max_400",
    "AntMaze_UMaze-v5_start_bottom_left",
    "AntMaze_Medium-v5_start_bottom_left",
]

# seed offset per environment: PointMaze gets fresh seeds 900-999, AntMaze keeps the 0-99 space its
# own train runs used. before: seed_index 7 -> after: a_seed 907 (PointMaze) / 7 (AntMaze).
SEED_OFFSET = {
    "initial_single_large_pointmaze_max_400": 900,
    "AntMaze_UMaze-v5_start_bottom_left": 0,
    "AntMaze_Medium-v5_start_bottom_left": 0,
}


def build_configs():
    """The 45 config specs in their fixed per-seed order (environments outer, bonus weights inner,
    ascending)."""
    configs = []
    for env_setup in ENV_SETUPS:
        for beta in BETAS:
            configs.append({"env_setup": env_setup, "arm": ARM, "lr": LR, "beta": beta,
                            "algorithm": "rnd_next_state",
                            "params": dict(ORIGSMALL_ADAM_LR1E3_PARAMS)})
    return configs


CONFIGS = build_configs()   # 3 envs x 15 bonus weights = 45

# Fixed args shared by every run. The environment itself comes entirely from --env_setup (train.py
# overlays the named EnvSetup onto every env-side field), so no env knob is repeated here and no
# run can silently disagree with its environment's definition.
FIXED_COMMON = {
    "eval_freq": 50000, "n_eval_episodes": 100,
    "eval_standalone": "False", "log_distance": "False", "device": "cpu",
    "rnd_obs_norm": "True", "rnd_distance": "mse", "rnd_output_dim": 128, "n_predictors": 1,
    "use_wandb": "False", "total_timesteps": 1000000,
}
STEPS = FIXED_COMMON["total_timesteps"]

RUN_TOTAL = len(SEED_INDICES) * len(CONFIGS)   # 100 * 45 = 4,500


def a_seed_of(env_setup, seed_index):
    """The actual --a_seed for one (environment, seed index) pair."""
    return SEED_OFFSET[env_setup] + seed_index


def config_key(cfg_spec):
    """Canonical 4-field grouping key: env_setup|arm|lr<lr>|b<beta> (train run 1.2's key shape)."""
    lr = "%g" % float(cfg_spec["lr"])
    beta = "%g" % float(cfg_spec["beta"])
    return f"{cfg_spec['env_setup']}|{cfg_spec['arm']}|lr{lr}|b{beta}"


def label(cfg_spec):
    """Short human tag inside the queue filename (the full identity lives in the JSON)."""
    env_short = (cfg_spec["env_setup"].replace("_start_bottom_left", "")
                 .replace("initial_single_large_pointmaze_max_400", "PointMaze_Large-topright"))
    return f"{env_short}_{cfg_spec['arm']}_lr{cfg_spec['lr']}_b{cfg_spec['beta']}"


def key_from_record(d):
    """The 4-field config_key straight from an output record, so the controller, the checker and the
    analysis all group runs the same way without re-reading the queue."""
    return (f"{d['env_setup']}|{ARM}|lr{'%g' % float(d['rnd_lr'])}|b{'%g' % float(d['beta'])}")


def write_manifest_row(sweep_id):
    """Append this sweep's row to data/SWEEPS.md (create the file with a header if missing)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder (Adam learning-rate 1e-3 addendum)\n\n"
                     "45 configurations (3 environments x 15 bonus weights) x up to 100 seeds at "
                     "1,000,000 env steps, raced against a per-environment bar frozen before launch "
                     "from the matching Adam 1e-4 configuration (slurm/truncation_controller.py + "
                     "slurm/FROZEN_BARS.json).\n\n"
                     "| sweep_id | runs | layout | status |\n|---|---|---|---|\n")
    with open(manifest, "a") as fh:
        fh.write(f"| {sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEED_INDICES)}seeds x "
                 f"{STEPS} steps | active |\n")


def main():
    """Write RUN_TOTAL config JSONs into queue/<sweep_id>/pending/, seed index outermost."""
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators use for_collaborator/")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    args = p.parse_args()
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    width = len(str(RUN_TOTAL))

    def write_marker(run_id, cfg_spec, seed_index):
        # one marker JSON per (configuration, seed); its location in queue/<sweep_id>/ IS the run's
        # state, and requeue_orphans re-pends a killed run from this same marker
        seed = a_seed_of(cfg_spec["env_setup"], seed_index)
        cfg = {
            "sweep_id": args.sweep_id, "run_id": run_id, "run_total": RUN_TOTAL,
            "pool": "pending",
            "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
            "arm": cfg_spec["arm"], "beta": cfg_spec["beta"],
            "a_seed": seed, "seed_index": seed_index,
            "config_key": config_key(cfg_spec),
            "params": cfg_spec["params"],
            "fixed": dict(FIXED_COMMON),
        }
        name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{label(cfg_spec)}_seed{seed}.json"
        with open(os.path.join(sweep_queue, "pending", name), "w") as fh:
            json.dump(cfg, fh)

    # seed index OUTERMOST, configuration inner: index i owns ids [45*i .. 45*i+44]
    run_id = 0
    for seed_index in SEED_INDICES:
        for cfg_spec in CONFIGS:
            write_marker(run_id, cfg_spec, seed_index)
            run_id += 1
    assert run_id == RUN_TOTAL
    write_manifest_row(args.sweep_id)
    print(f"queue built: {RUN_TOTAL} entries in {sweep_queue}/pending/ "
          f"({len(CONFIGS)} configurations x {len(SEED_INDICES)} seeds)")


if __name__ == "__main__":
    main()
