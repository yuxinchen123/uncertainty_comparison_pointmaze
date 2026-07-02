#!/usr/bin/env python
"""Build the Train-run-3.1.1 work queue for ONE sweep, scoped by a sweep id (the sweep-id convention,
.claude/rules/run-id-and-logging.md). Drives the live PyTorch trainer train.py (SB3 SAC), not the JAX run-4
trainer.

Run 3.1.1 sweeps four intrinsic-bonus methods on PointMaze_Large-v3 (top_right goal), each over a beta grid,
and (for the three elliptical methods) over a ridge-lambda grid and a feature-input grid:

  - A1  rnd_elliptical          (batch covariance, update_timing=sample)   x input{state_action,next_state} x ridge{1,1e-2} x beta(7)
  - A2  rnd_elliptical_global   (global covariance, update_timing=sample)  x input{state_action,next_state} x ridge{1,1e-2} x beta(7)
  - A3  rnd_elliptical_global   (global covariance, update_timing=add)     x input{state_action,next_state} x ridge{1,1e-2} x beta(7)
  - A4  rnd_next_state          (RND, next-state input, no ridge)          x beta(7)

So 3*2*2*7 = 84 elliptical configs + 7 RND configs = 91 configs per seed. With 100 seeds the sweep is
91*100 = 9100 runs. NOTE: this is only the FILE work-queue size (one pending JSON per run); the number of
Slurm jobs is small and fixed (see launch_queue.sh / .claude/rules/slurm-submission.md) -- workers drain
the queue. Every elliptical run is unit-norm (feature_normalization=unit) and 1e6 steps with standalone eval
OFF (training-episode reward only).

Within a sweep the run-id is the seed-OUTERMOST 0..TOTAL-1 position: seed s owns the contiguous block
[s*91 .. s*91+90], one id per config in CONFIGS order. The run-id is sweep-local; (sweep_id, run_id) is
unique within the run folder. Layout: queue/<sweep_id>/{pending,running,done,failed}/ and per-run JSONs under
data/<sweep_id>/local/<NNNN_of_9100>.json (train.py appends /local to --local_log_dir).
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

# Sweep axes (spelled out; no abstraction). Betas as strings so they reach train.py exactly as written.
BETAS = ["0.001", "0.01", "0.1", "1", "10", "100", "1000"]
RIDGES = ["1", "0.01"]                       # ridge lambda; elliptical only (RND has none)
INPUTS = ["state_action", "next_state"]      # elliptical encoder input: (s,a) vs next-state-only
# (algorithm, update_timing) for the three elliptical variants, in fixed order.
ELLIPTICAL = [
    ("rnd_elliptical", "sample"),            # A1 batch covariance, sample-time update
    ("rnd_elliptical_global", "sample"),     # A2 global covariance, sample-time update
    ("rnd_elliptical_global", "add"),        # A3 global covariance, add-time update
]
SEEDS = list(range(100))

# Fixed args passed verbatim to train.py for every config (names match train.py's argparse). Standalone eval
# OFF (the run-3.1.1 standard) -> training-episode reward only; distance-to-GT gridding OFF. Bools as strings
# so train.py's _str2bool parses them. SAC hyperparameters are left at SB3 defaults (as in run 2).
FIXED = {
    "env_name": "PointMaze_Large-v3",
    "total_timesteps": 1000000,
    "eval_freq": 50000,
    "n_eval_episodes": 100,
    "eval_standalone": "False",
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
    """Return the fixed-order list of the 91 inner configs (one per seed). Each is a dict with the algorithm,
    beta, and a `params` dict of per-config method knobs (elliptical knobs for the elliptical methods, empty
    for RND). Order: the 84 elliptical configs (algo/timing x input x ridge x beta), then the 7 RND configs."""
    configs = []
    # elliptical: 3 (algo,timing) x 2 input x 2 ridge x 7 beta = 84
    for algo, timing in ELLIPTICAL:
        for feature_input in INPUTS:
            for ridge in RIDGES:
                for beta in BETAS:
                    configs.append({
                        "algorithm": algo,
                        "beta": beta,
                        "params": {
                            "elliptical_update_timing": timing,
                            "elliptical_feature_input": feature_input,
                            "elliptical_regularization": ridge,
                            "elliptical_feature_normalization": "unit",
                        },
                    })
    # RND next-state: 7 beta, no ridge / no input axis
    for beta in BETAS:
        configs.append({"algorithm": "rnd_next_state", "beta": beta, "params": {}})
    return configs


CONFIGS = build_configs()
RUN_TOTAL = len(SEEDS) * len(CONFIGS)  # 100 * 91 = 9100


def write_manifest_row(sweep_id):
    """Append (or create) a one-line record of this sweep to data/SWEEPS.md so every sweep in the run folder
    is catalogued (id, size, config summary, status). Status starts 'active'; mark stale ones by hand."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    # header written once; each sweep is one row.
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder\n\n"
                     "Each row is one sweep launch (`build_queue.py --sweep_id`). Data lives under "
                     "`data/<sweep_id>/local/`, queue under `queue/<sweep_id>/`. Analysis loads a sweep by id; "
                     "pool reruns that share a config (same tag) to extend coverage. Mark superseded/invalid "
                     "sweeps `legacy` so they are excluded.\n\n"
                     "| sweep_id | runs | configs (per seed) x seeds x steps | eval_standalone | status |\n"
                     "|---|---|---|---|---|\n")
    row = (f"| {sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEEDS)}seeds x {FIXED['total_timesteps']} | "
           f"{FIXED['eval_standalone']} | active |\n")
    with open(manifest, "a") as fh:
        fh.write(row)


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/ and log the sweep."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<YYYY-MM-DD-HH-MM>_<tag> identifying this sweep")
    args = p.parse_args()
    # create ALL FOUR queue subdirs (workers rename pending -> running -> done/failed; a missing dir makes
    # every claim's os.rename fail and the queue look empty)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    # SEED outermost, config inner: seed s -> ids {91s .. 91s+90}; an earlier seed always has smaller ids.
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
    print(f"generated {run_id} configs into {pending} (sweep_id={args.sweep_id}, {len(CONFIGS)} configs/seed, "
          f"seed-outermost ids 0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
