#!/usr/bin/env python
"""Build the work queue for Train run 3.2.1 (sweep-id convention, .claude/rules/run-id-and-logging.md).

Run 3.2.1 sweeps three RND predictor optimizers on `rnd_next_state` (writeup sec:train-run-3-2-1),
scored on the final training-episode reward (standalone eval OFF). The O1 Adam + mse cell is NOT in
the grid (user decision 2026-07-04): it is exactly run 3.1.1's rnd_next_state reference arm, already
measured there with ~100 seeds. The grid is therefore 184 configurations:

  O1 adam    x readout {l2}       x beta (8)                      =   8
  O2 adagrad x readout {mse, l2}  x beta (8)                      =  16
  O3 sgd1t   x readout {mse, l2}  x eta0 (5) x t0 (2) x beta (8)  = 160

Each configuration gets seeds 0..49 in the queue (9200 entries), but the optuna controller
(optuna_controller.py) stops a configuration early -- moving its remaining pending entries to
queue/<sweep_id>/pruned/ -- once it has n >= 10 finished seeds and its 99% upper confidence limit
U = mean + 2.576*s/sqrt(n) falls below the bar (run 3.1.1's best rnd_next_state mean, 36.95 at the
2026-07-02 aggregation; the controller freezes the recomputed value at first start).

Run ids are seed-OUTERMOST: seed s owns ids [184*s .. 184*s+183], one id per config in the fixed
CONFIGS order (O1 block, then O2 readout-outer/beta-inner, then O3 readout/eta0/t0/beta-innermost).
Early seeds therefore run first (worker.py claims in approximate id order), so every configuration
crosses the 10-seed decision floor at about the same time.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

# Sweep axes (spelled out; values as strings so they reach train.py exactly as written).
BETAS = ["0.001", "0.01", "0.1", "1", "10", "100", "1000", "10000"]
READOUTS = ["mse", "l2"]
ETA0S = ["0.001", "0.003", "0.01", "0.03", "0.1"]
T0S = ["1000", "10000"]
SEEDS = list(range(50))

# Fixed args passed verbatim to train.py (names match its argparse). Standalone eval OFF (run-3.1.1
# standard): scored on the training-episode reward. Bools as strings for _str2bool.
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
    """Return the fixed-order list of the 184 inner configs (one per seed): O1 adam-l2 (beta asc),
    then O2 adagrad (readout mse->l2, beta asc), then O3 sgd1t (readout mse->l2, eta0 asc, t0 asc,
    beta asc innermost). Every config is algorithm rnd_next_state."""
    configs = []
    # O1 Adam: l2 readout only (the mse cell = run-3.1.1 reference arm, not re-run)
    for beta in BETAS:
        configs.append({
            "algorithm": "rnd_next_state",
            "beta": beta,
            "params": {"rnd_optimizer": "adam", "rnd_bonus_readout": "l2"},
        })
    # O2 AdaGrad: both readouts
    for readout in READOUTS:
        for beta in BETAS:
            configs.append({
                "algorithm": "rnd_next_state",
                "beta": beta,
                "params": {"rnd_optimizer": "adagrad", "rnd_bonus_readout": readout},
            })
    # O3 SGD-1/t: both readouts x eta0 x t0
    for readout in READOUTS:
        for eta0 in ETA0S:
            for t0 in T0S:
                for beta in BETAS:
                    configs.append({
                        "algorithm": "rnd_next_state",
                        "beta": beta,
                        "params": {"rnd_optimizer": "sgd1t", "rnd_bonus_readout": readout,
                                   "rnd_sgd_eta0": eta0, "rnd_sgd_t0": t0},
                    })
    return configs


CONFIGS = build_configs()
RUN_TOTAL = len(SEEDS) * len(CONFIGS)  # 50 * 184 = 9200


def config_key(params, beta):
    """Canonical grouping key for one configuration: optimizer|readout|eta0|t0|beta, numbers via %g.
    before: params={'rnd_optimizer': 'sgd1t', 'rnd_bonus_readout': 'l2', 'rnd_sgd_eta0': '0.003',
    'rnd_sgd_t0': '1000'}, beta='0.01'  ->  after: 'sgd1t|l2|0.003|1000|0.01'.
    Non-sgd1t methods have no eta0/t0 slots: 'adagrad|mse|-|-|100'."""
    opt = params["rnd_optimizer"]
    readout = params["rnd_bonus_readout"]
    eta0 = "%g" % float(params["rnd_sgd_eta0"]) if opt == "sgd1t" else "-"
    t0 = "%g" % float(params["rnd_sgd_t0"]) if opt == "sgd1t" else "-"
    return f"{opt}|{readout}|{eta0}|{t0}|%g" % float(beta)


def write_manifest_row(sweep_id):
    """Append (or create) this sweep's row in data/SWEEPS.md (id, size, config summary, status)."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder\n\n"
                     "One row per sweep launch (`build_queue.py --sweep_id`). Data under "
                     "`data/<sweep_id>/local/`, queue under `queue/<sweep_id>/`. Seeds are allocated "
                     "adaptively: the optuna controller moves a stopped configuration's remaining "
                     "pending entries to `queue/<sweep_id>/pruned/`, so most configurations finish "
                     "far fewer than 50 seeds by design. Mark superseded sweeps `legacy`.\n\n"
                     "| sweep_id | runs | configs x seeds x steps | allocation | status |\n"
                     "|---|---|---|---|---|\n")
    row = (f"| {sweep_id} | {RUN_TOTAL} | {len(CONFIGS)}cfg x {len(SEEDS)}seeds x "
           f"{FIXED['total_timesteps']} | bar rule: stop when n>=10 and mean+2.576*s/sqrt(n) < bar "
           f"(36.95 ref); 50-seed cap | active |\n")
    with open(manifest, "a") as fh:
        fh.write(row)


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<YYYY-MM-DD-HH-MM>_<tag> identifying this sweep")
    args = p.parse_args()
    # all FIVE queue subdirs (workers rename pending -> running -> done/failed; the controller
    # renames pending -> pruned; a missing dir breaks the renames)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    # SEED outermost, config inner: seed s owns ids [184s .. 184s+183] in CONFIGS order
    for seed in SEEDS:
        for cfg_spec in CONFIGS:
            cfg = {
                "sweep_id": args.sweep_id,
                "run_id": run_id,
                "run_total": RUN_TOTAL,
                "algorithm": cfg_spec["algorithm"],
                "beta": cfg_spec["beta"],
                "a_seed": seed,
                "config_key": config_key(cfg_spec["params"], cfg_spec["beta"]),
                "params": cfg_spec["params"],
                "fixed": dict(FIXED),
            }
            opt = cfg_spec["params"]["rnd_optimizer"]
            readout = cfg_spec["params"]["rnd_bonus_readout"]
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{opt}-{readout}_seed{seed}.json"
            with open(os.path.join(pending, name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    write_manifest_row(args.sweep_id)
    print(f"generated {run_id} configs into {pending} ({len(CONFIGS)} configs/seed, "
          f"seed-outermost ids 0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
