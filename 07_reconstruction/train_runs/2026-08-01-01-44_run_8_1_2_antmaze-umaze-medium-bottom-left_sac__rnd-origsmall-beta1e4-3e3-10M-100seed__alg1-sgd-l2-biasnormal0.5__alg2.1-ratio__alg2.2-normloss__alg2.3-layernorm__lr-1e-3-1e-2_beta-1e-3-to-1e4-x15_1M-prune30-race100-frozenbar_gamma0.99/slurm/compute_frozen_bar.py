#!/usr/bin/env python
"""Compute the run-8.1.2 frozen bars from the run-8.1 records and write slurm/FROZEN_BARS.json.

Run ONCE at launch, then the file is committed to git and never regenerated: BAR-1M(env) = the mean
whole-run training return of run 8.1's winning RND configuration (rnd_next_state, per-env winning
bonus weight) over its completed 1M records. Every stage1_controller decision line records the
file's sha256, and stage1_check verifies it — a mid-run bar change is a loud violation.

Usage:  python compute_frozen_bar.py            # writes FROZEN_BARS.json (refuses to overwrite)
        python compute_frozen_bar.py --print    # recompute and print only (no write)
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
OUT_PATH = os.path.join(HERE, "FROZEN_BARS.json")

# the source sweep: run 8.1's records (the executed train run 1.1)
RUN81_LOCAL = ("/p/rlprojects/RND/07_reconstruction/train_runs/"
               "2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__"
               "rnd-origsmall__gt-position-velocity__gt-position-maze-cell__gt-position-1m__"
               "beta-1e-3-to-1e4-x15_prune20-winner100-max300_1Mstep_default-max-episode_"
               "reward-shift-1_gamma0.99/data/2026-07-23-02-05_pm-am-run1/local")
ENVS = ["AntMaze_UMaze-v5_start_bottom_left", "AntMaze_Medium-v5_start_bottom_left"]
# expected winners from the run-8.1 final tables (Table 69) — the recompute must agree on the beta
EXPECTED_WINNER_BETA = {"AntMaze_UMaze-v5_start_bottom_left": "10000",
                        "AntMaze_Medium-v5_start_bottom_left": "3000"}


def score_of_record(d):
    """The racing score: mean per-episode extrinsic return over ALL training episodes (the run-8.1
    rule, identical to stage1_controller.score_of_record)."""
    rows = d.get("train_episode_history") or []
    vals = [r["train/extrinsic_reward"] for r in rows if "train/extrinsic_reward" in r]
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect():
    """Per (env, beta) score lists for run 8.1's completed rnd_next_state records on the two envs."""
    # before: ~15k JSONs of all algorithms/envs; after: {env: {beta: [score, ...]}} for RND only
    out = {env: {} for env in ENVS}
    paths = glob.glob(os.path.join(RUN81_LOCAL, "*.json"))
    if not paths:
        sys.exit(f"no run-8.1 records under {RUN81_LOCAL}")
    for path in paths:
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("env_setup") not in out or d.get("algorithm") != "rnd_next_state":
            continue
        if not d.get("completed", True):
            continue
        s = score_of_record(d)
        if s is None:
            continue
        beta = "%g" % float(d["beta"])
        out[d["env_setup"]].setdefault(beta, []).append(s)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true", dest="print_only")
    args = p.parse_args()
    if not args.print_only and os.path.exists(OUT_PATH):
        sys.exit(f"{OUT_PATH} already exists — the bars are FROZEN; use --print to recompute-only")
    per_env = collect()
    bars = {}
    for env in ENVS:
        by_beta = per_env[env]
        if not by_beta:
            sys.exit(f"no completed run-8.1 RND records for {env}")
        # the winner = the beta with the highest mean score (the run-8.1 ranking rule)
        stats = {}
        for beta, vals in by_beta.items():
            n = len(vals)
            mean = sum(vals) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
            stats[beta] = (mean, sd, n)
        winner = max(stats.items(), key=lambda kv: kv[1][0])
        beta, (mean, sd, n) = winner
        if beta != EXPECTED_WINNER_BETA[env]:
            sys.exit(f"recomputed winner beta {beta} for {env} != expected "
                     f"{EXPECTED_WINNER_BETA[env]} — investigate before freezing")
        bars[env] = {"beta": beta, "mean": mean, "sd": sd, "n": n, "se": sd / math.sqrt(n)}
        print(f"{env}: winner beta={beta} mean={mean:.4f} sd={sd:.4f} n={n}")
    if args.print_only:
        return
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=HERE).stdout.strip()
    doc = {"schema": 1, "source_sweep": "2026-07-23-02-05_pm-am-run1",
           "source_local_dir": RUN81_LOCAL,
           "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_at_compute": git,
           "score_rule": "mean of train/extrinsic_reward over all train_episode_history rows, "
                         "completed=true records only",
           "bars": bars}
    with open(OUT_PATH, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    sha = hashlib.sha256(open(OUT_PATH, "rb").read()).hexdigest()
    print(f"wrote {OUT_PATH} (sha256 {sha[:16]}...) — commit it; never regenerate")


if __name__ == "__main__":
    main()
