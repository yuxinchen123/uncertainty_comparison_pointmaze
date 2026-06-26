#!/usr/bin/env python
"""Build a Train-run-4 work queue for ONE sweep, scoped by a sweep id (the sweep-id convention,
.claude/rules/run-id-and-logging.md).

A run folder can hold MANY sweeps -- reruns of the same config (which EXTEND coverage) or different configs --
without their data colliding. Each sweep is identified by a `sweep_id` (`<YYYY-MM-DD-HH-MM>_<tag>`), and:
  - its queue lives under   queue/<sweep_id>/{pending,running,done,failed}/
  - its per-run JSONs under  data/<sweep_id>/local/<NNN_of_TOTAL>.json
so the analysis loads exactly the sweep(s) it wants and is never bothered by legacy data. Each sweep is
recorded as a row in data/SWEEPS.md.

Within a sweep, the run-id is the seed-outermost 0..TOTAL-1 position (gt_position_velocity, rnd_elliptical,
rnd_state per seed): seed s owns ids {3s, 3s+1, 3s+2}. The run-id is sweep-local; the (sweep_id, run_id) pair
is globally unique within the run folder.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

ALGO_BETA = [("gt_position_velocity", 1.0), ("rnd_elliptical", 0.01), ("rnd_state", 100.0)]
SEEDS = list(range(200))
RUN_TOTAL = len(SEEDS) * len(ALGO_BETA)  # 600

# Fixed args passed verbatim to run4_train.py. Matches run-2 EXCEPT total_timesteps (500K vs run-2's 1M).
# log_distance defaults OFF (0): the distance_to_gt/* metric grids the maze each eval and run-4 is a reward
# comparison; run-2's data already carries the distance field. Pass it through so the value is explicit.
FIXED = {
    "total_timesteps": 500000,
    "eval_freq": 50000,
    "n_eval_episodes": 100,
    "threads": 2,
    "log_distance": 0,
}


def write_manifest_row(sweep_id):
    """Append (or create) a one-line record of this sweep to data/SWEEPS.md so every sweep in the run folder
    is catalogued (id, size, config, distance on/off, status). Status starts 'active'; mark stale ones by hand."""
    manifest = os.path.join(RUN_DIR, "data", "SWEEPS.md")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    # header is written once; each sweep is one row. before: file may not exist. after: a row for this sweep.
    if not os.path.exists(manifest):
        with open(manifest, "w") as fh:
            fh.write("# Sweeps in this run folder\n\n"
                     "Each row is one sweep launch (`build_queue.py --sweep_id`). Data lives under "
                     "`data/<sweep_id>/local/`, queue under `queue/<sweep_id>/`. Analysis loads a sweep by id; "
                     "pool reruns that share a config (same tag) to extend coverage. Mark superseded/invalid "
                     "sweeps `legacy` so they are excluded.\n\n"
                     "| sweep_id | runs | algorithms x seeds x steps | log_distance | status |\n"
                     "|---|---|---|---|---|\n")
    algos = ",".join(a for a, _ in ALGO_BETA)
    row = (f"| {sweep_id} | {RUN_TOTAL} | {algos} x {len(SEEDS)}seeds x {FIXED['total_timesteps']} | "
           f"{'on' if FIXED['log_distance'] else 'off'} | active |\n")
    with open(manifest, "a") as fh:
        fh.write(row)


def main():
    """Write RUN_TOTAL config JSONs (seed-outermost ids) into queue/<sweep_id>/pending/ and log the sweep."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True, help="<YYYY-MM-DD-HH-MM>_<tag> identifying this sweep")
    args = p.parse_args()
    # create ALL FOUR queue subdirs (workers atomically rename pending -> running -> done/failed, so the
    # destination dirs must exist or every claim's os.rename fails and workers see an "empty" queue)
    sweep_queue = os.path.join(RUN_DIR, "queue", args.sweep_id)
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(sweep_queue, sub), exist_ok=True)
    pending = os.path.join(sweep_queue, "pending")
    width = len(str(RUN_TOTAL))
    run_id = 0
    # SEED outermost, algorithm inner: seed s -> ids {3s,3s+1,3s+2}; earlier seed -> smaller ids
    for seed in SEEDS:
        for algo, beta in ALGO_BETA:
            cfg = {
                "sweep_id": args.sweep_id,
                "run_id": run_id,
                "run_total": RUN_TOTAL,
                "algorithm": algo,
                "beta": beta,
                "a_seed": seed,
                "fixed": dict(FIXED),
            }
            name = f"{run_id:0{width}d}_of_{RUN_TOTAL}_{algo}_seed{seed}.json"
            with open(os.path.join(pending, name), "w") as fh:
                json.dump(cfg, fh)
            run_id += 1
    write_manifest_row(args.sweep_id)
    print(f"generated {run_id} configs into {pending} (sweep_id={args.sweep_id}, seed-outermost ids "
          f"0..{run_id-1}, total={RUN_TOTAL})")


if __name__ == "__main__":
    main()
