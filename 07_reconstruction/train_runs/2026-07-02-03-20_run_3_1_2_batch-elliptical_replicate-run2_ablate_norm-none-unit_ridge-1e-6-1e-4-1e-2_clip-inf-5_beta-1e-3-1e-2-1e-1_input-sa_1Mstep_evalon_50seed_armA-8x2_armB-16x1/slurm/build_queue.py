#!/usr/bin/env python
"""Build ONE arm's work queue for Train run 3.1.2 (sweep-id convention, .claude/rules/run-id-and-logging.md).

Run 3.1.2 replicates run-2's batch elliptical and ablates the three implementation deltas, all with the
batch elliptical (`rnd_elliptical`), (s,a) input, sample-time covariance updates, 1M steps, standalone
eval ON (the numbers compare directly to run-2's 44.49 eval reward):

  feature normalization {none (raw), unit} x ridge lambda {1e-6, 1e-4, 1e-2} x bonus clip {inf, 5}
  = 12 cells, each at beta {0.001, 0.01, 0.1}  ->  36 configs per seed.

The cell (none, 1e-6, inf, beta=0.01) is the exact run-2 replica. The 50 seeds are split by PARITY across
two Slurm-shape arms (the infrastructure comparison): arm A (8 tasks x 2 cpus) gets the EVEN seeds
0,2,..,48; arm B (16 tasks x 1 cpu) gets the ODD seeds 1,3,..,49 — 25 seeds/arm, run_total = 900/arm.
Science pools both arms to n=50 per config; per-arm speed/error stats come free via the sweep-id isolation.

Within an arm the run-id is seed-OUTERMOST: seed index s (0-based within the arm's 25 seeds) owns ids
[36*s .. 36*s+35], one id per config in the fixed order (normalization outer, lambda, clip, beta inner).
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

# Sweep axes (spelled out; values as strings so they reach train.py exactly as written).
NORMALIZATIONS = ["none", "unit"]
RIDGES = ["1e-06", "0.0001", "0.01"]
CLIPS = ["inf", "5"]
BETAS = ["0.001", "0.01", "0.1"]
ALL_SEEDS = list(range(50))
ARM_SEEDS = {"A": [s for s in ALL_SEEDS if s % 2 == 0],   # even seeds -> arm A (8 tasks x 2 cpus)
             "B": [s for s in ALL_SEEDS if s % 2 == 1]}   # odd seeds  -> arm B (16 tasks x 1 cpu)

# Fixed args passed verbatim to train.py (names match its argparse). Standalone eval ON: every cell's
# final reward is the same 100-episode deterministic eval as run-2's 44.49. Bools as strings for _str2bool.
FIXED = {
    "env_name": "PointMaze_Large-v3",
    "total_timesteps": 1000000,
    "eval_freq": 50000,
    "n_eval_episodes": 100,
    "eval_standalone": "True",
    "log_distance": "False",
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


def build_configs():
    """Return the fixed-order list of the 36 inner configs (one per seed): normalization outer, then
    lambda, clip, beta inner. Every config is the batch elliptical with (s,a) input and sample timing."""
    configs = []
    for norm in NORMALIZATIONS:
        for ridge in RIDGES:
            for clip in CLIPS:
                for beta in BETAS:
                    configs.append({
                        "algorithm": "rnd_elliptical",
                        "beta": beta,
                        "params": {
                            "elliptical_feature_normalization": norm,
                            "elliptical_regularization": ridge,
                            "elliptical_bonus_clip": clip,
                            "elliptical_update_timing": "sample",
                            "elliptical_feature_input": "state_action",
                        },
                    })
    return configs


CONFIGS = build_configs()
RUN_TOTAL = len(ARM_SEEDS["A"]) * len(CONFIGS)  # 25 * 36 = 900 per arm


def write_manifest_row(sweep_id, arm):
    """Append (or create) this sweep's row in data/SWEEPS.md (id, arm, size, config summary, status)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder\n\n"
                     "One row per sweep launch (`build_queue.py --sweep_id --arm`). Data under "
                     "`data/<sweep_id>/local/`, queue under `queue/<sweep_id>/`. The two arms share the "
                     "36-config grid and split the 50 seeds by parity; pool both for the science, compare "
                     "them for the Slurm-shape question. Mark superseded sweeps `legacy`.\n\n"
                     "| sweep_id | arm (shape / seeds) | runs | configs x seeds x steps | eval | status |\n"
                     "|---|---|---|---|---|---|\n")
    shape = "8 tasks x 2 cpus, even seeds" if arm == "A" else "16 tasks x 1 cpu, odd seeds"
    row = (f"| {sweep_id} | {arm} ({shape}) | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(ARM_SEEDS[arm])}seeds x "
           f"{FIXED['total_timesteps']} | on | active |\n")
    with open(manifest, "a") as fh:
        fh.write(row)


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/ for one arm."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<YYYY-MM-DD-HH-MM>_<tag> identifying this arm's sweep")
    p.add_argument("--arm", required=True, choices=["A", "B"], help="A = even seeds (8x2), B = odd seeds (16x1)")
    args = p.parse_args()
    # all FOUR queue subdirs (workers rename pending -> running -> done/failed; a missing dir breaks claims)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    # SEED outermost, config inner: the arm's s-th seed owns ids [36s .. 36s+35] in CONFIGS order
    for seed in ARM_SEEDS[args.arm]:
        for cfg_spec in CONFIGS:
            cfg = {
                "sweep_id": args.sweep_id,
                "arm": args.arm,
                "run_id": run_id,
                "run_total": RUN_TOTAL,
                "algorithm": cfg_spec["algorithm"],
                "beta": cfg_spec["beta"],
                "a_seed": seed,
                "params": cfg_spec["params"],
                "fixed": dict(FIXED),
            }
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{cfg_spec['algorithm']}_seed{seed}.json"
            with open(os.path.join(pending, name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    write_manifest_row(args.sweep_id, args.arm)
    print(f"generated {run_id} configs into {pending} (arm {args.arm}, {len(CONFIGS)} configs/seed, "
          f"seed-outermost ids 0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
