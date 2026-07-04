#!/usr/bin/env python
"""Build the work queue for the run-3.1.2 FOLLOW-UP sweep: unit normalization at ridge 1e-8, no clip.

Motivation (run-3.1.2 finding iv, writeup section 5.3.5): unit-normalized features cannot reach the
run-2 replica's effective ridge ratio rho inside the 3.1.1/3.1.2 grids — their floor is
rho = lambda*d = 1.28e-4 at lambda=1e-6, 35x the replica's 3.7e-6 — and matching run-2's bonus contrast
under unit norm needs lambda around 3e-8. This sweep tests that regime directly with ONE cell below the
old grid: unit normalization, lambda = 1e-8 (rho = lambda*d = 1.28e-6, BELOW the replica's measured
3.7e-6), no clip, batch elliptical with (s,a) input, sample-time updates, 1M steps, standalone eval ON
(numbers compare directly to run-2's 44.49 and the replica's 52.76). Because the bonus scale grows as
the ridge shrinks, beta is swept over five decades {1e-4, 1e-3, 1e-2, 1e-1, 1e0} (run 3.1.2 finding v:
beta* shifts inversely with bonus scale), each at 50 seeds: 5 x 50 = 250 runs.

Sweep-id convention: .claude/rules/run-id-and-logging.md. Seed OUTERMOST, beta inner (ascending), so
seed s owns ids [5s .. 5s+4] and early seeds finish first under the ordered-window worker claim.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

# Sweep axes (values as strings so they reach train.py exactly as written). ONE cell, beta + seed swept.
NORMALIZATION = "unit"
RIDGE = "1e-08"
CLIP = "inf"
BETAS = ["0.0001", "0.001", "0.01", "0.1", "1.0"]
SEEDS = list(range(50))

# Fixed args passed verbatim to train.py — identical to the 3.1.2 replicate/ablate sweeps (eval ON,
# distance logging off, cpu device); only the cell above differs.
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
    """Return the fixed-order list of the 5 inner configs (one per seed): beta ascending, single cell."""
    configs = []
    for beta in BETAS:
        configs.append({
            "algorithm": "rnd_elliptical",
            "beta": beta,
            "params": {
                "elliptical_feature_normalization": NORMALIZATION,
                "elliptical_regularization": RIDGE,
                "elliptical_bonus_clip": CLIP,
                "elliptical_update_timing": "sample",
                "elliptical_feature_input": "state_action",
            },
        })
    return configs


CONFIGS = build_configs()
RUN_TOTAL = len(SEEDS) * len(CONFIGS)  # 50 * 5 = 250


def write_manifest_row(sweep_id):
    """Append this sweep's row to the run folder's data/SWEEPS.md manifest (columns match the 3.1.2 rows)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    row = (f"| {sweep_id} | follow-up (16 tasks x 1 cpu, seeds 0-49) | {RUN_TOTAL} | "
           f"{len(CONFIGS)}cfg x {len(SEEDS)}seeds x {FIXED['total_timesteps']} | on | active |\n")
    with open(manifest, "a") as fh:
        fh.write(row)


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<YYYY-MM-DD-HH-MM>_<tag> identifying this sweep")
    args = p.parse_args()
    # all FOUR queue subdirs (workers rename pending -> running -> done/failed; a missing dir breaks claims)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    # SEED outermost, beta inner: seed s owns ids [5s .. 5s+4] in CONFIGS (beta-ascending) order
    for seed in SEEDS:
        for cfg_spec in CONFIGS:
            cfg = {
                "sweep_id": args.sweep_id,
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
    write_manifest_row(args.sweep_id)
    print(f"generated {run_id} configs into {pending} ({len(CONFIGS)} betas/seed, "
          f"seed-outermost ids 0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
