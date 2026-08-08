#!/usr/bin/env python
"""Build the work queue for train run 6: the four train-run-8.1.2 algorithms on the train-run-5
PointMaze task, raced against the run-5 original RND as a frozen reference.

One sweep, one pool (pending/): 120 configurations x 300 seeds = 36,000 markers at 1,000,000 env
steps. The 120 configurations = 4 algorithm arms x 2 predictor learning rates x 15 bonus weights,
all on ONE environment — `initial_single_large_pointmaze_max_400`, the exact task of train run 5
(PointMaze Large, top-right goal, cap 400, discount 0.999, raw sparse reward). The original RND is
NOT re-run: its reference numbers come from train run 5's own final records (300 seeds), frozen in
slurm/FROZEN_BARS.json.

The four arms are the train-run-8.1.2 arms, their parameter dictionaries copied VERBATIM from that
run's slurm/build_queue.py (ALG1_PARAMS + ARM_EXTRAS); slurm/test_arm_equality.py asserts
byte-equality against that file, so this run cannot silently drift from the arms it claims to test:
- alg1:   plain constant-rate SGD predictor (lr swept), l2 bonus readout, bias normal_0.5,
          original training loss, reward normalization OFF;
- alg2.1: alg1 + a frozen copy of the predictor at initialization, ratio bonus
          ||e_theta|| / (||e_0|| + 1e-8);
- alg2.2: alg2.1 with the initialization-normalized training loss;
- alg2.3: alg2.1 with LayerNorm inside both networks (original loss).

Seeds are `a_seed` 1200-1499 — 300 fresh seeds, disjoint from every earlier PointMaze run (run 5
used 600-899, the learning-rate addendum 900-1199, everything before that <= 599). Seed index is
OUTERMOST: index i owns run ids [120*i .. 120*i+119] in the fixed CONFIGS order, so early seeds
finish first across every configuration and the two-phase truncation controller decides wave by
wave. Most markers are never launched: phase 1 truncates configurations against the frozen bar
from 20 completed seeds, and phase 2 stops every non-best surviving configuration per arm at 100
seeds, so only up to 4 winners (one per arm) consume seeds 100-299.

config_key = "<env_setup>|<arm>|lr<lr>|b<beta>" — the train-run-8.1.2 key shape, so keys from the
two runs line up if the data is ever pooled.
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

SEED_INDICES = list(range(0, 300))   # 300 seeds per configuration = the winner phase's ceiling

# the 15 bonus weights of train runs 1.2 / the addendum, swept identically on every arm (the
# strings reach train.py verbatim so the queue records exactly what was asked for)
BETAS = ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30",
         "100", "300", "1000", "3000", "10000"]

# the 2 swept plain-SGD learning rates (constant rate, momentum 0, no decay) — 8.1.2's LRS
LRS = ["0.001", "0.01"]

# The task-S single stack of train run 8.1.2, VERBATIM from that run's build_queue.py ALG1_PARAMS
# (test_arm_equality.py pins this): original-small architecture + state normalization, reward norm
# OFF, plain constant-rate SGD (rnd_lr is added per config from LRS), l2 readout, bias normal_0.5.
ALG1_PARAMS = {
    "rnd_optimizer": "sgd", "rnd_bonus_readout": "l2",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "False",
    "rnd_bias_init": "normal_0.5", "rnd_weight_init": "orthogonal",
}

# arm name -> the EXTRA params on top of ALG1_PARAMS, VERBATIM from 8.1.2's ARM_EXTRAS
ARM_EXTRAS = {
    "alg1": {},
    "alg2.1": {"rnd_readout_norm_init": "True"},
    "alg2.2": {"rnd_readout_norm_init": "True", "rnd_predictor_loss": "mse_init_normalized"},
    "alg2.3": {"rnd_readout_norm_init": "True", "rnd_layer_norm": "True"},
}
ARMS = ["alg1", "alg2.1", "alg2.2", "alg2.3"]

# the ONE environment: train run 5's task, exactly as its registry entry defines it
ENV_SETUP = "initial_single_large_pointmaze_max_400"
SEED_OFFSET = 1200   # seed_index 0 -> a_seed 1200; index 299 -> a_seed 1499


def build_configs():
    """The 120 config specs in their fixed per-seed order (arms outer, learning rates, then bonus
    weights, ascending) — the same axis order train run 8.1.2 used inside one environment."""
    configs = []
    for arm in ARMS:
        for lr in LRS:
            for beta in BETAS:
                params = dict(ALG1_PARAMS)
                params.update(ARM_EXTRAS[arm])
                params["rnd_lr"] = lr
                configs.append({"env_setup": ENV_SETUP, "arm": arm, "lr": lr, "beta": beta,
                                "algorithm": "rnd_next_state", "params": params})
    return configs


CONFIGS = build_configs()   # 4 arms x 2 learning rates x 15 bonus weights = 120

# Fixed args shared by every run. The environment itself comes entirely from --env_setup, so no
# env knob is repeated here and no run can silently disagree with the environment's definition.
FIXED_COMMON = {
    "eval_freq": 50000, "n_eval_episodes": 100,
    "eval_standalone": "False", "log_distance": "False", "device": "cpu",
    "rnd_obs_norm": "True", "rnd_distance": "mse", "rnd_output_dim": 128, "n_predictors": 1,
    "use_wandb": "False", "total_timesteps": 1000000,
}
STEPS = FIXED_COMMON["total_timesteps"]

RUN_TOTAL = len(SEED_INDICES) * len(CONFIGS)   # 300 * 120 = 36,000


def a_seed_of(seed_index):
    """The actual --a_seed for one seed index."""
    return SEED_OFFSET + seed_index


def config_key(cfg_spec):
    """Canonical 4-field grouping key: env_setup|arm|lr<lr>|b<beta> (train run 8.1.2's shape)."""
    lr = "%g" % float(cfg_spec["lr"])
    beta = "%g" % float(cfg_spec["beta"])
    return f"{cfg_spec['env_setup']}|{cfg_spec['arm']}|lr{lr}|b{beta}"


def arm_from_record(d):
    """The arm name from a per-run JSON record's flag fields (the inverse of ARM_EXTRAS)."""
    # before: {"rnd_optimizer": "sgd", "rnd_layer_norm": true, ...} -> "alg2.3"
    if str(d.get("rnd_layer_norm")) == "True" or d.get("rnd_layer_norm") is True:
        return "alg2.3"
    if d.get("rnd_predictor_loss") == "mse_init_normalized":
        return "alg2.2"
    if str(d.get("rnd_readout_norm_init")) == "True" or d.get("rnd_readout_norm_init") is True:
        return "alg2.1"
    return "alg1"


def key_from_record(d):
    """The 4-field config_key straight from an output record, so the controller, the checker and
    the analysis all group runs the same way without re-reading the queue."""
    return (f"{d['env_setup']}|{arm_from_record(d)}|lr{'%g' % float(d['rnd_lr'])}"
            f"|b{'%g' % float(d['beta'])}")


def label(cfg_spec):
    """Short human tag inside the queue filename (the full identity lives in the JSON)."""
    return f"PointMaze_Large-topright_{cfg_spec['arm']}_lr{cfg_spec['lr']}_b{cfg_spec['beta']}"


def write_manifest_row(sweep_id):
    """Append this sweep's row to data/SWEEPS.md (create the file with a header if missing)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder (train run 6)\n\n"
                     "120 configurations (4 train-run-8.1.2 algorithm arms x 2 predictor learning "
                     "rates x 15 bonus weights) on the train-run-5 PointMaze task, raced against "
                     "the run-5 original RND frozen bar (slurm/truncation_controller.py + "
                     "slurm/FROZEN_BARS.json): truncation from 20 completed seeds; at 100 seeds "
                     "only the best configuration per arm continues, to 300 seeds.\n\n"
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
        # one marker JSON per (configuration, seed); its location in queue/<sweep_id>/ IS the
        # run's state, and requeue_orphans re-pends a killed run from this same marker
        seed = a_seed_of(seed_index)
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

    # seed index OUTERMOST, configuration inner: index i owns ids [120*i .. 120*i+119]
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
