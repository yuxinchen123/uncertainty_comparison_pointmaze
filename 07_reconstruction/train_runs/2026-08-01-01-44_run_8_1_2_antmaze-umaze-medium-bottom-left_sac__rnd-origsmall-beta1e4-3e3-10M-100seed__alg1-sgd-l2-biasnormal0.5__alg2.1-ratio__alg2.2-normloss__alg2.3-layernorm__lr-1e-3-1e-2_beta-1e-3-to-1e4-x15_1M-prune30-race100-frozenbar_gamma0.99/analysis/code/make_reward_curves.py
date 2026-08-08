#!/usr/bin/env python
"""The train-run-1.2 reward-curve figure (the Figure-23 kind of train run 1.1): a 2x2 grid — one
column per environment (AntMaze UMaze left, Medium right, the tables' order), the SAME data in
both rows, top row on a LINEAR x axis and bottom row on a LOG x axis (user request: both scales).

Series per panel:
- RND baseline of train run 1.1 (the previous run's winner, from the run-8.1 curves cache):
  BLACK DASHED, to 10^6 steps;
- RND baseline of this run (task R, the same configuration rerun): BLACK SOLID, to 10^7 steps;
- the four sweep arms' best configurations so far (best = highest mean whole-run return, the
  tables' rule), fixed Okabe-Ito colors, to 10^6 steps.
Each curve is the mean over the configuration's completed seeds of the per-run mean return over
the past 100 training episodes (train_history, logged every 50k steps); the band is +-1 standard
error over seeds. A dotted vertical rule marks 10^6 steps, the stage-1 screening length.

Usage:  /p/rlprojects/RND/.venvs/exploration/bin/python make_reward_curves.py
Writes: ../plots/reward_curves_run812.pdf and .png
(first run reads every completed per-run JSON of the sweep and caches the curves)
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(os.path.dirname(HERE))
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
SWEEP_ID = "2026-08-01-02-03_run812"
CACHE = os.path.join(os.path.dirname(HERE), "data", "curves_cache.json")

# the previous run's final curves cache (train run 1.1) — the black dashed reference
RUN81_CACHE = os.path.join(
    os.path.dirname(RUN_DIR),
    "2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall"
    "__gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15"
    "_prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99",
    "analysis", "data", "curves_cache.json")
RUN81_BASELINE_KEY = {  # the run-1.1 winner per env (3-field run-8.1 keys)
    "AntMaze_UMaze-v5_start_bottom_left": "AntMaze_UMaze-v5_start_bottom_left|rnd_next_state|10000",
    "AntMaze_Medium-v5_start_bottom_left": "AntMaze_Medium-v5_start_bottom_left|rnd_next_state|3000",
}

# The Adam addendum: the SAME RND baseline stack at two other predictor learning rates, run as its
# own sweep. Added here so each panel shows the baseline at every rate rather than only 1e-4.
LR1E3_RUN = os.path.join(
    os.path.dirname(RUN_DIR),
    "2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-"
    "seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_"
    "prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim")
LR1E3_SWEEP = "2026-08-05-16-05_lr1e3"
LR1E3_MIN_SEEDS = 5
# the addendum's predictor learning rates, drawn in this order, each with its own colour
ADDENDUM_LRS = ["0.001", "0.01"]
ADDENDUM_COLOR = {"0.001": "#D55E00", "0.01": "#56B4E9"}


def load_addendum(env_setup):
    """{lr: (beta, curves, n)} for each addendum rate's best bonus weight on one environment.

    Best is by the whole-run mean per-episode return — the same ranking this figure's companion
    table uses — so the curve drawn is the same configuration that table names. One pass over the
    addendum's records for this environment serves both rates, keeping only the per-eval curve and
    the score. A rate with no weight at LR1E3_MIN_SEEDS finished seeds is absent from the result,
    which is the state of a rate whose runs are all still in flight.
    before: ~9,000 addendum JSONs across three environments and two learning rates
    after : {"0.001": ("3000", [([0, 50000, ...], [-700.0, ...]), ...], 40)}
    """
    import collections
    whole = collections.defaultdict(lambda: collections.defaultdict(list))
    curves = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in glob.glob(os.path.join(LR1E3_RUN, "data", LR1E3_SWEEP, "local", "*.json")):
        try:
            d = json.load(open(path))
        except (ValueError, OSError):
            continue
        if not d.get("completed") or d.get("env_setup") != env_setup:
            continue
        lr = "%g" % float(d.get("rnd_lr", 0))
        if lr not in ADDENDUM_COLOR:
            continue
        eps = [r["train/extrinsic_reward"] for r in (d.get("train_episode_history") or [])]
        rows = d.get("train_history") or []
        if not eps or not rows:
            continue
        beta = "%g" % float(d["beta"])
        whole[lr][beta].append(sum(eps) / len(eps))
        curves[lr][beta].append(([r["step"] for r in rows],
                                 [r["train/mean_extrinsic_reward"] for r in rows]))
    out = {}
    for lr in ADDENDUM_LRS:
        ranked = [(b, sum(v) / len(v)) for b, v in whole[lr].items() if len(v) >= LR1E3_MIN_SEEDS]
        if not ranked:
            continue
        beta = max(ranked, key=lambda kv: kv[1])[0]
        out[lr] = (beta, curves[lr][beta], len(whole[lr][beta]))
    return out


sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue          # noqa: E402  (ENV_SETUPS_RUN12 / CONFIGS / key_from_record)
import stage1_controller    # noqa: E402  (score_of_record — the tables' scoring)

# fixed arm -> (color, display name) across every panel; the two baselines are black (dashed =
# train run 1.1, solid = this run's task R), the sweep arms take Okabe-Ito colors
ARM_STYLE = {
    "alg1":   ("#0072B2", "algorithm 1"),
    "alg2.1": ("#009E73", "algorithm 2.1"),
    "alg2.2": ("#E69F00", "algorithm 2.2"),
    "alg2.3": ("#CC79A7", "algorithm 2.3"),
}


def load_curves():
    """One pass over the completed per-run JSONs: {config_key: {"scores": [..], "curves":
    [(steps, rewards), ..]}} with the 4-field run-8.1.2 keys. Task-S records give 20-point curves,
    task-R records 200-point curves; only the curve and the score are kept per record.

    before: 00042_of_24200.json (full record, long train_episode_history)
    after : by_key["AntMaze_...|alg2.2|lr0.01|b30"]["curves"][k] = (array of steps, array of means)
    """
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            raw = json.load(fh)
        print(f"[load] cache hit: {CACHE}", flush=True)
        return {k: {"scores": e["scores"],
                    "curves": [(np.array(s), np.array(v)) for s, v in e["curves"]]}
                for k, e in raw.items()}
    local = os.path.join(RUN_DIR, "data", SWEEP_ID, "local")
    by_key = {}
    paths = glob.glob(os.path.join(local, "*.json"))
    for i, path in enumerate(paths):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("completed", True):
            continue
        score = stage1_controller.score_of_record(d)
        if score is None:
            continue
        rows = d.get("train_history") or []
        if not rows:
            continue
        steps = np.array([r["step"] for r in rows], dtype=float)
        vals = np.array([r["train/mean_extrinsic_reward"] for r in rows], dtype=float)
        e = by_key.setdefault(build_queue.key_from_record(d), {"scores": [], "curves": []})
        e["scores"].append(score)
        e["curves"].append((steps, vals))
        if (i + 1) % 2000 == 0:
            print(f"[load] {i + 1}/{len(paths)} files", flush=True)
    print(f"[load] done: {sum(len(e['curves']) for e in by_key.values())} completed records "
          f"across {len(by_key)} configurations", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump({k: {"scores": e["scores"],
                       "curves": [(s.tolist(), v.tolist()) for s, v in e["curves"]]}
                   for k, e in by_key.items()}, fh)
    print(f"[load] cache written: {CACHE}", flush=True)
    return by_key


def load_run81_baseline(env_setup):
    """The train-run-1.1 winner's curves for one env from the run-8.1 final cache.
    before: cache entry {"curves": [([50000..1e6], [..20 means..]), x n_seeds]}
    after : list of (steps array, values array)"""
    with open(RUN81_CACHE) as fh:
        raw = json.load(fh)
    e = raw[RUN81_BASELINE_KEY[env_setup]]
    return [(np.array(s), np.array(v)) for s, v in e["curves"]]


def best_key(by_key, env_setup, arm):
    """The arm's best config_key in this env by mean whole-run return (the tables' rule)."""
    keys = [build_queue.config_key(c) for c in build_queue.CONFIGS
            if c["env_setup"] == env_setup and c["arm"] == arm]
    best, best_mean = None, None
    for k in keys:
        e = by_key.get(k)
        if not e:
            continue
        m = float(np.mean(e["scores"]))
        if best_mean is None or m > best_mean:
            best, best_mean = k, m
    return best


def band(curves):
    """(steps, mean, standard error, n) across seed curves on the modal common grid.
    before: [(steps, vals seed A), (steps, vals seed B), ...]
    after : (grid, mean over seeds, SE over seeds), n_used"""
    grid = Counter(tuple(s) for s, _ in curves).most_common(1)[0][0]
    vals = np.array([v for s, v in curves if tuple(s) == grid])
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros_like(mean)
    return np.array(grid), mean, se, len(vals)


def draw_panel(ax, env_setup, by_key, log_x):
    """One panel: both black baselines + the four arms' best configs, band = +-1 SE over seeds."""
    # train-run-1.1 baseline (black dashed, to 1M)
    steps, mean, se, n = band(load_run81_baseline(env_setup))
    beta81 = RUN81_BASELINE_KEY[env_setup].split("|")[2]
    ax.plot(steps, mean, color="black", linestyle="--", linewidth=1.6,
            label=f"RND baseline, Adam $10^{{-4}}$, run 1.1 $\\beta$={beta81} (n={n})")
    ax.fill_between(steps, mean - se, mean + se, color="black", alpha=0.10, linewidth=0)
    # this run's task-R baseline (black solid, to 10M)
    base_spec = next(c for c in build_queue.BASELINE_CONFIGS if c["env_setup"] == env_setup)
    e = by_key.get(build_queue.config_key(base_spec))
    if e:
        steps, mean, se, n = band(e["curves"])
        ax.plot(steps, mean, color="black", linestyle="-", linewidth=1.8,
                label=f"RND baseline, Adam $10^{{-4}}$, this run $\\beta$={base_spec['beta']} (n={n})")
        ax.fill_between(steps, mean - se, mean + se, color="black", alpha=0.10, linewidth=0)
    # the same baseline stack at each addendum rate (to 1M) — each in its own colour and dashed, so
    # they read as siblings of the two black baselines rather than as extra algorithms
    for lr, (beta_a, curves_a, _) in sorted(load_addendum(env_setup).items()):
        steps, mean, se, n = band(curves_a)
        exponent = int(round(math.log10(float(lr))))
        ax.plot(steps, mean, color=ADDENDUM_COLOR[lr], linestyle="--", linewidth=1.8,
                label=f"RND baseline, Adam $10^{{{exponent}}}$ $\\beta$={beta_a} (n={n})")
        ax.fill_between(steps, mean - se, mean + se, color=ADDENDUM_COLOR[lr], alpha=0.15,
                        linewidth=0)
    # the four sweep arms' best configurations (colored, to 1M)
    for arm, (color, name) in ARM_STYLE.items():
        k = best_key(by_key, env_setup, arm)
        if k is None:
            continue
        steps, mean, se, n = band(by_key[k]["curves"])
        _, _, lr_part, beta_part = k.split("|")
        label = f"{name} lr={lr_part[2:]} $\\beta$={beta_part[1:]} (n={n})"
        ax.plot(steps, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.18, linewidth=0)
    # the stage-1 screening length
    ax.axvline(1e6, color="#666666", linestyle=":", linewidth=1.0)
    # linear row: the stage-1 range only (0 to 1M, user request), so the screening region is
    # readable at full width; log row: the full 10M range incl. the task-R baseline's tail
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(5e4, 1.05e7)
    else:
        ax.set_xlim(0, 1.02e6)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    # legend placement clears the data (checked on the render): log panels are empty at the top
    # left; the 1M linear panels differ per env — UMaze's peak sits at 0.3-0.5M so the descending
    # right side is the least-informative region (also shown in the log row), while Medium's upper
    # LEFT quarter is empty (no curve above -992 before 0.35M). Compact, opaque box.
    if log_x:
        loc = "upper left"
    else:
        loc = "upper right" if env_setup.startswith("AntMaze_UMaze") else "upper left"
    ax.legend(fontsize=6, loc=loc, framealpha=1.0, handlelength=1.5, labelspacing=0.25,
              borderpad=0.35)


def main():
    by_key = load_curves()
    envs = build_queue.ENV_SETUPS_RUN12
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, env_setup in enumerate(envs):
        for row, log_x in enumerate([False, True]):
            ax = axes[row][col]
            draw_panel(ax, env_setup, by_key, log_x)
            scale = "log $x$ (to $10^{7}$)" if log_x else "linear $x$ (to $10^{6}$)"
            ax.set_title(env_setup.replace("_start_bottom_left", "") + f" --- {scale}", fontsize=10)
            if row == 1:
                ax.set_xlabel("environment step")
            if col == 0:
                ax.set_ylabel("training reward (mean of past 100 episodes)", fontsize=8)
    # the run's own arms stopped 2026-08-05; the Adam 1e-3 line on top of them is still filling seeds
    fig.suptitle("Train run 1.2 at its 2026-08-05 stop, plus the interim Adam $10^{-3}$ and "
                 "$10^{-2}$ RND baselines: best configuration per arm\n"
                 "mean $\\pm$ 1 standard error over completed seeds; dotted rule at $10^{6}$ "
                 "steps (the stage-1 screening length)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(PLOTS, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(PLOTS, f"reward_curves_run812.{ext}")
        fig.savefig(out, dpi=180)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
