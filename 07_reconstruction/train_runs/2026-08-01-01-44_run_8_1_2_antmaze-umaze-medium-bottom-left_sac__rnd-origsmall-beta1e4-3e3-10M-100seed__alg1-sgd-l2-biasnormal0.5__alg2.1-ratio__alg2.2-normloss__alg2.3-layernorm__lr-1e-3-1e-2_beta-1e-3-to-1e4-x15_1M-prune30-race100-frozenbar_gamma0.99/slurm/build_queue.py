#!/usr/bin/env python
"""Build the work queue for train run 8.1.2 (writeup subsubsection 8.1.2) — two tasks, two pools.

TASK S (stage 1, pool pending_1m/): 240 sweep configs x 100 seeds = 24,000 entries at 1M steps,
seed OUTERMOST: seed s owns ids [240*s .. 240*s+239] in the fixed CONFIGS order, so early seeds
finish first across every config and the frozen-bar prune decisions (stage1_controller.py) archive
the un-run seeds of pruned configs wave by wave.

The 240 configs = 2 env setups (AntMaze UMaze / Medium, start bottom-left) x 4 algorithm arms
x 2 SGD learning rates x 15 bonus weights. The four arms (single stack = the run-5 original-small
architecture + state normalization, reward normalization OFF, plain constant-rate SGD):
- alg1:   l2 bonus readout, bias normal_0.5, original training loss;
- alg2.1: alg1 + frozen initial predictor, ratio bonus ||e_theta||/(||e_0||+1e-8);
- alg2.2: alg2.1 with the initialization-normalized training loss;
- alg2.3: alg2.1 with LayerNorm inside both nets.

TASK R (baseline, pool pending_10m/): the run-8.1 RND winner per env (UMaze beta 1e4, Medium 3e3,
run-5 original-small stack, reward norm ON) x 100 seeds = 200 entries at 10M steps, never pruned.
Claimed only by workers whose WORKER_POOLS includes pending_10m (gpu cuda workers; gnolim cpu
workers claim it FIRST and fall back to pending_1m) behind worker.py's remaining-walltime guard.

config_key = "<env_setup>|<arm>|lr<lr>|b<beta>" (4 fields; the arm names the knob bundle).
The unit test slurm/test_run_queue_convention.py pins the ordering, ids, and counts.
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

SEEDS = list(range(0, 100))     # race target 100 seeds per config (both tasks)

# the 15 bonus weights, swept identically by every task-S arm (strings reach train.py verbatim)
BETAS = ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30",
         "100", "300", "1000", "3000", "10000"]

# the 2 swept plain-SGD learning rates (constant rate, momentum 0, no decay)
LRS = ["0.001", "0.01"]

# train-run-5 original-small RND stack (verbatim from run 8.1's build_queue.py ORIGSMALL_PARAMS) —
# the TASK-R baseline arm: adam 1e-4, mse_mean readout, reward normalization ON.
ORIGSMALL_PARAMS = {
    "rnd_optimizer": "adam", "rnd_bonus_readout": "mse_mean", "rnd_lr": "0.0001",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "True", "rnd_reward_norm_gamma": "0.99",
    "rnd_bias_init": "zero", "rnd_weight_init": "orthogonal",
}

# the task-S single stack: original-small architecture + state normalization, reward norm OFF,
# plain constant-rate SGD (rnd_lr is added per config from LRS), l2 readout, bias normal_0.5.
ALG1_PARAMS = {
    "rnd_optimizer": "sgd", "rnd_bonus_readout": "l2",
    "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
    "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
    "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "False",
    "rnd_bias_init": "normal_0.5", "rnd_weight_init": "orthogonal",
}

# arm name -> the EXTRA params on top of ALG1_PARAMS (the plan's algorithm 2.x definitions)
ARM_EXTRAS = {
    "alg1": {},
    "alg2.1": {"rnd_readout_norm_init": "True"},
    "alg2.2": {"rnd_readout_norm_init": "True", "rnd_predictor_loss": "mse_init_normalized"},
    "alg2.3": {"rnd_readout_norm_init": "True", "rnd_layer_norm": "True"},
}
ARMS = ["alg1", "alg2.1", "alg2.2", "alg2.3"]

# the 2 env setups (writeup table order) and the task-R fixed winning bonus weight per env
ENV_SETUPS_RUN12 = [
    "AntMaze_UMaze-v5_start_bottom_left",
    "AntMaze_Medium-v5_start_bottom_left",
]
BASELINE_BETA = {
    "AntMaze_UMaze-v5_start_bottom_left": "10000",
    "AntMaze_Medium-v5_start_bottom_left": "3000",
}


def build_configs():
    """The 240 task-S config specs in their fixed per-seed order (env setups outer, arms, learning
    rates, betas innermost ascending)."""
    configs = []
    for env_setup in ENV_SETUPS_RUN12:
        for arm in ARMS:
            for lr in LRS:
                for beta in BETAS:
                    params = dict(ALG1_PARAMS)
                    params.update(ARM_EXTRAS[arm])
                    params["rnd_lr"] = lr
                    configs.append({"env_setup": env_setup, "arm": arm, "lr": lr, "beta": beta,
                                    "algorithm": "rnd_next_state", "params": params})
    return configs


def build_baseline_configs():
    """The 2 task-R baseline specs (one per env, fixed winner beta, orig-small stack)."""
    configs = []
    for env_setup in ENV_SETUPS_RUN12:
        configs.append({"env_setup": env_setup, "arm": "baseline", "lr": "0.0001",
                        "beta": BASELINE_BETA[env_setup], "algorithm": "rnd_next_state",
                        "params": dict(ORIGSMALL_PARAMS)})
    return configs


CONFIGS = build_configs()                 # 240 task-S configs
BASELINE_CONFIGS = build_baseline_configs()  # 2 task-R configs

# Fixed args shared by every run; per-task total_timesteps below. The worker overrides --device per
# node type (WORKER_DEVICE cpu / cuda); the queue JSONs stay device-neutral.
FIXED_COMMON = {
    "eval_freq": 50000, "n_eval_episodes": 100,
    "eval_standalone": "False", "log_distance": "False", "device": "cpu",
    "rnd_obs_norm": "True", "rnd_distance": "mse", "rnd_output_dim": 128, "n_predictors": 1,
    "use_wandb": "False",
}
STEPS_S = 1000000     # task S: stage-1 screening length
STEPS_R = 10000000    # task R: the baseline's full length

N_S = len(SEEDS) * len(CONFIGS)            # 100 * 240 = 24,000 task-S entries
N_R = len(SEEDS) * len(BASELINE_CONFIGS)   # 100 * 2   = 200 task-R entries
RUN_TOTAL = N_S + N_R                      # one id space: [0, N_S) task S, [N_S, RUN_TOTAL) task R


def config_key(cfg_spec):
    """Canonical 4-field grouping key: env_setup|arm|lr<lr>|b<beta>. The arm names the knob bundle
    (baseline / alg1 / alg2.1 / alg2.2 / alg2.3); lr and beta identify the config inside the arm."""
    lr = "%g" % float(cfg_spec["lr"])
    beta = "%g" % float(cfg_spec["beta"])
    return f"{cfg_spec['env_setup']}|{cfg_spec['arm']}|lr{lr}|b{beta}"


def label(cfg_spec):
    """Short human tag inside the queue filename (full identity lives in the JSON)."""
    env_short = cfg_spec["env_setup"].replace("_start_bottom_left", "")
    return f"{env_short}_{cfg_spec['arm']}_lr{cfg_spec['lr']}_b{cfg_spec['beta']}"


def arm_from_record(d):
    """Rebuild the arm name from a per-run JSON record's flag fields (the inverse of ARM_EXTRAS;
    used by the controller/checker/report so records group without carrying the arm explicitly)."""
    # before: {"rnd_optimizer": "adam", ...} -> "baseline" (only the orig-small arm uses adam)
    # before: {"rnd_optimizer": "sgd", "rnd_layer_norm": true, ...} -> "alg2.3"
    if d.get("rnd_optimizer") == "adam":
        return "baseline"
    if d.get("rnd_layer_norm"):
        return "alg2.3"
    if d.get("rnd_predictor_loss") == "mse_init_normalized":
        return "alg2.2"
    if d.get("rnd_readout_norm_init"):
        return "alg2.1"
    return "alg1"


def key_from_record(d):
    """The 4-field config_key straight from an output record (arm inferred from the flag fields)."""
    return (f"{d['env_setup']}|{arm_from_record(d)}|lr{'%g' % float(d['rnd_lr'])}"
            f"|b{'%g' % float(d['beta'])}")


def write_manifest_row(sweep_id):
    """Append this sweep's row to data/SWEEPS.md (create with a header if missing)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder (train run 8.1.2)\n\n"
                     "Task S: 240 configs x up to 100 seeds at 1M steps, frozen-bar racing from 30 "
                     "seeds (slurm/stage1_controller.py + slurm/FROZEN_BARS.json). Task R: 2 "
                     "baseline configs x 100 seeds at 10M steps, never pruned.\n\n"
                     "| sweep_id | runs | layout | status |\n|---|---|---|---|\n")
    with open(manifest, "a") as fh:
        fh.write(f"| {sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEEDS)}seeds x {STEPS_S} "
                 f"+ {len(BASELINE_CONFIGS)}cfg x {len(SEEDS)}seeds x {STEPS_R} | active |\n")


def main():
    """Write RUN_TOTAL config JSONs: task S (seed-outermost ids) into queue/<sweep_id>/pending_1m/,
    task R into queue/<sweep_id>/pending_10m/."""
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators use for_collaborator/ if present")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    args = p.parse_args()
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    width = len(str(RUN_TOTAL))

    def write_marker(pool, run_id, cfg_spec, seed, steps, task):
        # one marker JSON per (config, seed); `pool` routes it to the claiming worker class and
        # requeue_orphans re-pends a killed run into the same pool it came from.
        cfg = {
            "sweep_id": args.sweep_id, "run_id": run_id, "run_total": RUN_TOTAL,
            "task": task, "pool": pool,
            "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
            "arm": cfg_spec["arm"], "beta": cfg_spec["beta"], "a_seed": seed,
            "config_key": config_key(cfg_spec),
            "params": cfg_spec["params"],
            "fixed": dict(FIXED_COMMON, total_timesteps=steps),
        }
        name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{label(cfg_spec)}_seed{seed}.json"
        with open(os.path.join(sweep_queue, pool, name), "w") as fh:
            json.dump(cfg, fh)

    # task S: SEED outermost, config inner — seed s owns ids [240*s .. 240*s+239] in CONFIGS order
    run_id = 0
    for seed in SEEDS:
        for cfg_spec in CONFIGS:
            write_marker("pending_1m", run_id, cfg_spec, seed, STEPS_S, "S")
            run_id += 1
    # task R: ids [N_S, RUN_TOTAL), env outer, seed inner (no racing, order immaterial)
    for cfg_spec in BASELINE_CONFIGS:
        for seed in SEEDS:
            write_marker("pending_10m", run_id, cfg_spec, seed, STEPS_R, "R")
            run_id += 1
    assert run_id == RUN_TOTAL
    write_manifest_row(args.sweep_id)
    print(f"queue built: {N_S} task-S entries in pending_1m/, {N_R} task-R entries in pending_10m/ "
          f"under {sweep_queue}")


if __name__ == "__main__":
    main()
