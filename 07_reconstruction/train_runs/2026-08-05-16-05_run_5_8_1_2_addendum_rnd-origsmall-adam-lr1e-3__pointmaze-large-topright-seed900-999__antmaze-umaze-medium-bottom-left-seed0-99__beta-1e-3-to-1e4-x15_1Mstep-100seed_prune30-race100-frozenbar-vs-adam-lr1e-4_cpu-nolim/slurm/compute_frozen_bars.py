#!/usr/bin/env python
"""Compute this run's frozen truncation bars and write slurm/FROZEN_BARS.json.

Run ONCE at launch, then the file is committed and never regenerated: every truncation decision line
records its sha256 and truncation_check.py verifies it, so a mid-run bar change is a loud violation.

BAR(env) = the mean score of the BEST Adam 1e-4 configuration on that environment, under that
environment's own score rule (slurm/score_rules.py), over the completed records of the executed run
that measured it:

- initial_single_large_pointmaze_max_400 -> train run 5's "original-small" arm (Adam 1e-4,
  mse_mean readout, reward normalization on), best of its 8 bonus weights by mean FINAL REWARD.
  Expected winner: bonus weight 1000.
- AntMaze_UMaze-v5_start_bottom_left / AntMaze_Medium-v5_start_bottom_left -> train run 1.1's
  SAC + RND arm (the same original-small stack at Adam 1e-4), best of its 15 bonus weights by mean
  WHOLE-RUN return. Expected winners: 10000 (UMaze) and 3000 (Medium). These are the same two
  numbers train run 1.2 froze, and the recompute must reproduce them.

Usage:  python compute_frozen_bars.py            # writes FROZEN_BARS.json (refuses to overwrite)
        python compute_frozen_bars.py --print    # recompute and print only (no write)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_rules  # noqa: E402  (the ONE score definition, shared with the controller)

OUT_PATH = os.path.join(HERE, "FROZEN_BARS.json")
TRAIN_RUNS = os.path.dirname(os.path.dirname(HERE))

# train run 5's records (the executed PointMaze Adam 1e-4 sweep, seeds 600-899)
RUN5_LOCAL = os.path.join(
    TRAIN_RUNS,
    "2026-07-20-16-44_run_5_baseline_rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-"
    "leaky0.2-warmupenv6400-rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-"
    "b1e4__seed600-899_1Mstep_cpu-nolim",
    "data", "2026-07-20-16-55_set-baseline", "local")
# train run 1.1's records (the executed AntMaze Adam 1e-4 sweep)
RUN81_LOCAL = os.path.join(
    TRAIN_RUNS,
    "2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall__"
    "gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15_"
    "prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99",
    "data", "2026-07-23-02-05_pm-am-run1", "local")

POINTMAZE_ENV = "initial_single_large_pointmaze_max_400"
ANTMAZE_ENVS = ["AntMaze_UMaze-v5_start_bottom_left", "AntMaze_Medium-v5_start_bottom_left"]

# the recompute must land on these bonus weights, else something about the source data changed
EXPECTED_WINNER_BETA = {
    POINTMAZE_ENV: "1000",
    "AntMaze_UMaze-v5_start_bottom_left": "10000",
    "AntMaze_Medium-v5_start_bottom_left": "3000",
}
SOURCE_RUN = {
    POINTMAZE_ENV: "train run 5, sweep 2026-07-20-16-55_set-baseline",
    "AntMaze_UMaze-v5_start_bottom_left": "train run 1.1, sweep 2026-07-23-02-05_pm-am-run1",
    "AntMaze_Medium-v5_start_bottom_left": "train run 1.1, sweep 2026-07-23-02-05_pm-am-run1",
}


def _iter_completed(local_dir):
    """Yield every completed per-run record under one sweep's local/ directory."""
    paths = glob.glob(os.path.join(local_dir, "*.json"))
    if not paths:
        sys.exit(f"no records under {local_dir}")
    for path in paths:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush; the bars are computed from settled data only
        if not d.get("completed", True):
            continue  # partial checkpoint of a killed attempt never enters a bar
        yield d


def collect_pointmaze():
    """Per-bonus-weight FINAL-REWARD score lists for train run 5's original-small Adam 1e-4 arm.

    Train run 5's records carry no env_setup field (they predate the flag) and hold three arms in one
    sweep, so the arm is identified by its knobs: the mse_mean readout is unique to original-small,
    and rnd_lr 1e-4 pins the learning rate this run is the 1e-3 counterpart of.
    """
    # before: 1,913 run-5 JSONs across 10 configurations; after: {"1000": [38.4, ...], ...} for the
    # 8 original-small bonus weights only
    out = {}
    for d in _iter_completed(RUN5_LOCAL):
        if d.get("rnd_bonus_readout") != "mse_mean":
            continue                                  # the benchmark / reward-norm arms
        if abs(float(d.get("rnd_lr", 0)) - 1e-4) > 1e-12:
            continue                                  # not the Adam 1e-4 line
        s = score_rules.score_of_record(d, env_setup=POINTMAZE_ENV)
        if s is not None:
            out.setdefault("%g" % float(d["beta"]), []).append(s)
    return out


def collect_antmaze(env_setup):
    """Per-bonus-weight WHOLE-RUN-MEAN score lists for train run 1.1's SAC + RND arm on one AntMaze
    environment."""
    # before: ~15k run-8.1 JSONs across 8 environments and 4 algorithms; after:
    # {"10000": [-689.1, ...], ...} for one environment's RND arm only
    out = {}
    for d in _iter_completed(RUN81_LOCAL):
        if d.get("env_setup") != env_setup or d.get("algorithm") != "rnd_next_state":
            continue
        s = score_rules.score_of_record(d, env_setup=env_setup)
        if s is not None:
            out.setdefault("%g" % float(d["beta"]), []).append(s)
    return out


def stats(vals):
    """Sample mean, (n-1) standard deviation and count of a non-empty score list."""
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mean, sd, n


def winner(by_beta, env):
    """The best bonus weight of one environment's Adam 1e-4 sweep, checked against the expected one."""
    if not by_beta:
        sys.exit(f"no completed Adam 1e-4 records for {env}")
    scored = {beta: stats(vals) for beta, vals in by_beta.items()}
    beta, (mean, sd, n) = max(scored.items(), key=lambda kv: kv[1][0])
    if beta != EXPECTED_WINNER_BETA[env]:
        sys.exit(f"recomputed winner bonus weight {beta} for {env} != expected "
                 f"{EXPECTED_WINNER_BETA[env]} — investigate before freezing")
    return beta, mean, sd, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true", dest="print_only")
    args = p.parse_args()
    if not args.print_only and os.path.exists(OUT_PATH):
        sys.exit(f"{OUT_PATH} already exists — the bars are FROZEN; use --print to recompute-only")
    bars = {}
    for env in [POINTMAZE_ENV] + ANTMAZE_ENVS:
        by_beta = collect_pointmaze() if env == POINTMAZE_ENV else collect_antmaze(env)
        beta, mean, sd, n = winner(by_beta, env)
        rule = score_rules.rule_for(env)
        bars[env] = {"beta": beta, "mean": mean, "sd": sd, "n": n,
                     "se": sd / math.sqrt(n), "score_rule": rule,
                     "score_rule_description": score_rules.RULE_DESCRIPTION[rule],
                     "source": SOURCE_RUN[env]}
        print(f"{env}: rule={rule} winner bonus weight={beta} mean={mean:.4f} sd={sd:.4f} n={n}")
    if args.print_only:
        return
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=HERE).stdout.strip()
    doc = {"schema": 1, "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_at_compute": git,
           "source_local_dirs": {POINTMAZE_ENV: RUN5_LOCAL,
                                 ANTMAZE_ENVS[0]: RUN81_LOCAL, ANTMAZE_ENVS[1]: RUN81_LOCAL},
           "bars": bars}
    with open(OUT_PATH, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()
    print(f"wrote {OUT_PATH} (sha256 {sha[:16]}...) — commit it; never regenerate")


if __name__ == "__main__":
    main()
