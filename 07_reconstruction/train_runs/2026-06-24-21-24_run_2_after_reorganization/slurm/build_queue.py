#!/usr/bin/env python
"""Generate the 600-config work queue as one JSON file per config under queue/pending/.

Replaces the W&B sweep as the source of configs: 3 algorithms x 1 logging mode ("local") x 200 seeds
= 600 runs. Idempotent guard: refuses to clobber a non-empty pending/ unless --reset is passed.

Sweep / id convention (see .claude/rules/run-id-and-logging.md):
- The SEED is the OUTERMOST loop, so each run's integer id is grouped by seed: a_seed s contributes the
  contiguous id block [s*len(ALGO_BETA) .. s*len(ALGO_BETA)+len(ALGO_BETA)-1], one id per algorithm in
  ALGO_BETA order. An earlier seed always has smaller ids than a later seed.
- run_id is the run's 0-based position in this sweep; run_total is the sweep size (600). Together they
  are the run's identity "run_id / run_total" and drive the per-run JSON filename in train.py.

Env: RUN_DIR = the train_runs/<run>/ folder.
"""
import os
import sys
import json

RUN = os.environ["RUN_DIR"]
QUEUE = os.path.join(RUN, "queue")
SUBDIRS = ("pending", "running", "done", "failed")

ALGO_BETA = ["gt_position_velocity|1", "rnd_elliptical|0.01", "rnd_state|100"]
# No wandb at all now: the local queue distributes configs and train.py logs locally to one JSON per run.
# One mode ("local", use_wandb=False) x 3 algos x 200 seeds = 600 runs.
MODE = "local"
SEEDS = list(range(200))
RUN_TOTAL = len(SEEDS) * len(ALGO_BETA)  # 600 = the "/total" in each run's id

# Fixed args passed verbatim to train.py for every config (names match train.py's argparse).
FIXED = {
    "env_name": "PointMaze_Large-v3",
    "total_timesteps": 1000000,
    "eval_freq": 50000,
    "n_eval_episodes": 100,
    "n_eval_episodes_final": 100,
    "device": "cpu",
    "discount_factor": 0.999,
    "env_max_episode": 400,
    "goal_position": "top_right",
    "rnd_obs_norm": "True",
    "rnd_distance": "mse",
    "rnd_output_dim": 128,
    "n_predictors": 1,
    "apply_termination_wrapper": "False",
}


def main():
    reset = "--reset" in sys.argv
    for d in SUBDIRS:
        os.makedirs(os.path.join(QUEUE, d), exist_ok=True)
    pending = os.path.join(QUEUE, "pending")
    if os.listdir(pending) and not reset:
        print(f"queue/pending already has {len(os.listdir(pending))} files; pass --reset to regenerate")
        return
    if reset:
        for d in SUBDIRS:
            dd = os.path.join(QUEUE, d)
            for f in os.listdir(dd):
                os.remove(os.path.join(dd, f))
    # SEED outermost, algorithm inner: run_id increments seed-by-seed, so seed 0 -> ids {0,1,2}
    # (gt_position_velocity, rnd_elliptical, rnd_state), seed 1 -> ids {3,4,5}, ... seed 199 -> {597,598,599}.
    # before: ALGO outermost gave ids grouped by algorithm; after: SEED outermost groups ids by seed.
    width = len(str(RUN_TOTAL))  # zero-pad ids to the width of run_total (600 -> 3 -> "042"), for sortable names
    run_id = 0
    for seed in SEEDS:
        for ab in ALGO_BETA:
            cfg = {
                "run_id": run_id,         # 0-based position of this run in the sweep ("i" of i/total)
                "run_total": RUN_TOTAL,   # sweep size ("total" of i/total)
                "g_algo_beta": ab,
                "z_logging_mode": MODE,
                "a_seed": seed,
                "fixed": FIXED,
            }
            # queue filename is id-leading (sorts in sweep order) and keeps algo+seed for human-readable resume
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{ab.split('|')[0]}_seed{seed}.json"
            with open(os.path.join(pending, name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    print(f"generated {run_id} configs into {pending} (seed-outermost ids 0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
