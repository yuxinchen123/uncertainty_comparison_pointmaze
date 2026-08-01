#!/usr/bin/env python
"""The 4x2 reward-curve figure of the run plan (plan step 8, item 3): one panel per environment
(PointMaze column left, AntMaze column right, sizes top to bottom), each panel drawing the
training reward curve of ONLY the best configuration of each algorithm — mean +- one standard
error over that configuration's completed seeds, on the shared 50k-step logging grid.

Curve source: each completed per-run JSON's train_history rows (train/mean_extrinsic_reward =
mean per-episode extrinsic return over the past 100 training episodes, logged every 50k steps).
Best configuration per (environment, algorithm) = highest mean whole-run per-episode return
(prune_controller.score_of_record), the same selection the writeup tables use.

Colors are fixed per algorithm across every panel (Okabe-Ito colorblind-safe palette).

Usage:  /p/rlprojects/RND/.venvs/exploration/bin/python make_reward_curves.py
Writes: ../plots/reward_curves_8env.pdf and .png
(reads every completed per-run JSON of the sweep — takes some minutes)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(os.path.dirname(HERE))
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
SWEEP_ID = "2026-07-23-02-05_pm-am-run1"

sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue          # noqa: E402  (ENV_SETUPS_RUN1 / CONFIGS — env + config order)
import prune_controller     # noqa: E402  (score_of_record / key_from_record — the table scoring)

# fixed algorithm -> (color, display name) across every panel; Okabe-Ito palette
ALGO_STYLE = {
    "no_exploration":        ("#767676", "no_exploration"),
    "rnd_next_state":        ("#0072B2", "rnd_next_state"),
    "gt_position_velocity":  ("#009E73", "gt_position_velocity"),
    "gt_position_maze_cell": ("#E69F00", "gt_position_maze_cell"),
    "gt_position_1m":        ("#CC79A7", "gt_position_1m"),
}


CACHE = os.path.join(os.path.dirname(HERE), "data", "curves_cache.json")


def load_curves():
    """One pass over the completed per-run JSONs: {config_key: {"scores": [..], "curves":
    [(steps, rewards), ..]}}. Each curve is the record's train_history mean-extrinsic-reward
    column; records without a computable score or not completed are skipped (same filter as the
    writeup tables). Memory stays small: only the 20-point curve is kept per record.

    before: 00042_of_92400.json  (full record, ~1400-episode train_episode_history)
    after : by_key["PointMaze_...|rnd_next_state|3000"]["curves"][k] = (array 20 steps, 20 means)
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
        score = prune_controller.score_of_record(d)
        if score is None:
            continue
        rows = d.get("train_history") or []
        steps = np.array([r["step"] for r in rows], dtype=float)
        vals = np.array([r["train/mean_extrinsic_reward"] for r in rows], dtype=float)
        if len(rows) == 0:
            continue
        e = by_key.setdefault(prune_controller.key_from_record(d), {"scores": [], "curves": []})
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


def best_key(by_key, env_setup, algorithm):
    """The algorithm's best config_key in this env by mean whole-run return (the table rule)."""
    keys = [build_queue.config_key(c) for c in build_queue.CONFIGS
            if c["env_setup"] == env_setup and c["algorithm"] == algorithm]
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
    """(steps, mean, standard error) across seed curves on the common grid; curves whose grid
    differs from the modal grid are excluded (none expected — every completed run logs the same
    20 steps).

    before: [(steps 50k..1M, vals seed A), (steps 50k..1M, vals seed B), ...]
    after : (steps 50k..1M, mean over seeds, SE over seeds), n_used
    """
    from collections import Counter
    grid = Counter(tuple(s) for s, _ in curves).most_common(1)[0][0]
    vals = np.array([v for s, v in curves if tuple(s) == grid])
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.zeros_like(mean)
    return np.array(grid), mean, se, len(vals)


def main():
    by_key = load_curves()
    envs = build_queue.ENV_SETUPS_RUN1
    pm = [e for e in envs if e.startswith("PointMaze")]
    am = [e for e in envs if e.startswith("AntMaze")]

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)
    for col, fam in enumerate([pm, am]):
        for row, env_setup in enumerate(fam):
            ax = axes[row][col]
            algos = []
            seen = set()
            for c in build_queue.CONFIGS:                       # env's algorithms in config order
                if c["env_setup"] == env_setup and c["algorithm"] not in seen:
                    seen.add(c["algorithm"])
                    algos.append(c["algorithm"])
            for algorithm in algos:
                k = best_key(by_key, env_setup, algorithm)
                if k is None:
                    continue
                steps, mean, se, n = band(by_key[k]["curves"])
                color, name = ALGO_STYLE[algorithm]
                beta = k.split("|")[2]
                label = name if algorithm == "no_exploration" else f"{name} $\\beta$={beta}"
                ax.plot(steps, mean, color=color, linewidth=1.8, label=f"{label} (n={n})")
                ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.18, linewidth=0)
            ax.set_title(env_setup.replace("_start_bottom_left", ""), fontsize=10)
            if env_setup.startswith("AntMaze_Large"):
                # all four curves coincide at -1000: every episode ends at the 1000-step limit
                ax.set_ylim(-1004, -992)
                ax.annotate("all four algorithms flat at $-1000$:\nno run ever reaches the goal",
                            xy=(0.5, 0.45), xycoords="axes fraction", ha="center",
                            fontsize=9, color="#444444")
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.legend(fontsize=6.5, loc="best", framealpha=0.85)
            if row == 3:
                ax.set_xlabel("environment step")
            if col == 0:
                ax.set_ylabel("training reward (mean of past 100 episodes)", fontsize=8)
    fig.suptitle("Point maze + ant maze train run 1: best configuration per algorithm\n"
                 "mean $\\pm$ 1 standard error over completed seeds (final data, sweep ended "
                 "2026-07-31)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(PLOTS, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(PLOTS, f"reward_curves_8env.{ext}")
        fig.savefig(out, dpi=180)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
