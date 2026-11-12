"""Compare completed Montezuma 1B-step curves on paired independent seeds.

Inputs are *run directories* with run.json and metrics.jsonl (H100 can
contain four copies, RTX cards one). The archived 2B baseline can be
truncated at update 61,035. Only extrinsic episode_return_sum/count is scored.

Usage:
 python scripts/atari_pixel_campaign/analyze_neural_rnd_1b.py \
   --baseline-run /path/to/baseline/h100/run \
   --baseline-run /path/to/baseline/a4500_seed-05/run \
   --candidate-run /path/to/candidate/h100/run \
   --candidate-run /path/to/candidate/a4500_seed-05/run \
   --out-dir /path/to/analysis

Repeat each option for every independent run directory. Candidate and
baseline must contain the same seed IDs for paired analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PER_UPDATE = 128 * 128
UPDATES = 61035
STEP_BUDGET = PER_UPDATE * UPDATES
BIN_STEPS = 20_000_000
N_BINS = 50


def read_runs(paths):
    """Return per-seed arrays: count and sum of extrinsic scores in 50 bins."""
    output = {}
    for run in paths:
        metadata = json.loads((run / "run.json").read_text())
        seeds = metadata["copy_seeds"]
        learner = metadata["learner"]
        if learner.get("n_envs") != 128 or learner.get("num_steps") != 128:
            raise ValueError(f"Non-matching PPO rollout dimensions in {run}")
        if learner.get("total_timesteps") != 2_000_000_000:
            raise ValueError(f"Wrong 2B annealing horizon in {run}")
        if metadata["environment"].get("max_episode_steps", 27000) != 27000:
            raise ValueError(f"Uncorrected episode cap in {run}")
        per_seed = {int(seed): np.zeros((N_BINS, 2), dtype=np.float64) for seed in seeds}
        previous_update = 0
        with (run / "metrics.jsonl").open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                update = int(row["update"])
                if update <= previous_update:
                    raise ValueError(f"Nonmonotonic updates in {run}: {update}")
                previous_update = update
                if update > UPDATES:
                    break
                b = ((update - 1) * PER_UPDATE) // BIN_STEPS
                n = np.asarray(row["episode_count"], dtype=np.float64).reshape(-1)
                s = np.asarray(row["episode_return_sum"], dtype=np.float64).reshape(-1)
                if len(n) != len(seeds) or len(s) != len(seeds):
                    raise ValueError(f"Wrong copy axis at update {update} in {run}")
                for j, seed in enumerate(seeds):
                    per_seed[int(seed)][b] += [n[j], s[j]]
        if previous_update < UPDATES:
            raise ValueError(f"Run {run} has {previous_update} updates; needs >= {UPDATES}")
        for seed, result in per_seed.items():
            if seed in output:
                raise ValueError(f"Duplicate seed {seed} in {run}")
            if np.any(result[:, 0] == 0):
                empty = np.flatnonzero(result[:, 0] == 0)
                raise ValueError(f"Seed {seed} has bins with zero ended games: {empty.tolist()}")
            output[seed] = result
    return output


def paired_results(baseline, candidate, n_bootstrap=4000):
    seeds = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        raise ValueError(f"Seed mismatch: baseline {sorted(baseline)}, candidate {sorted(candidate)}")
    def summarize(source):
        curve = np.stack([source[seed][:, 1] / source[seed][:, 0] for seed in seeds])
        # AUC: equally weighted mean of 50 fixed 20M-step bin scores.
        auc = curve.mean(axis=1)
        # Endpoint: all finished game scores from the last 100M steps.
        last = np.array([source[seed][45:, 1].sum() / source[seed][45:, 0].sum() for seed in seeds])
        return curve, auc, last
    b_curve, b_auc, b_last = summarize(baseline)
    c_curve, c_auc, c_last = summarize(candidate)
    deltas = {"auc": c_auc - b_auc, "endpoint_900m_1b": c_last - b_last}
    rng = np.random.default_rng(20261111)
    idx = rng.integers(0, len(seeds), (n_bootstrap, len(seeds)))
    inference = {}
    for name, differences in deltas.items():
        boot = differences[idx].mean(axis=1)
        inference[name] = {
            "baseline_mean": float((b_auc if name == "auc" else b_last).mean()),
            "candidate_mean": float((c_auc if name == "auc" else c_last).mean()),
            "mean_paired_difference": float(differences.mean()),
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        }
    return seeds, b_curve, c_curve, inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", action="append", type=Path, required=True)
    p.add_argument("--candidate-run", action="append", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    baseline = read_runs(args.baseline_run)
    candidate = read_runs(args.candidate_run)
    seeds, b_curve, c_curve, results = paired_results(baseline, candidate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": "extrinsic score per finished game",
        "bin_steps": BIN_STEPS, "bins": N_BINS,
        "step_budget_per_seed": STEP_BUDGET,
        "seeds": seeds, "n_independent_seeds": len(seeds),
        "comparison": results,
        "baseline_run_paths": [str(p) for p in args.baseline_run],
        "candidate_run_paths": [str(p) for p in args.candidate_run],
        "statistical_unit": "independent PPO seed",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "curve_bins.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "bin", "end_agent_steps", "baseline_return_per_game",
                         "candidate_return_per_game"])
        for i, seed in enumerate(seeds):
            for b in range(N_BINS):
                writer.writerow([seed, b+1, min((b+1)*BIN_STEPS, STEP_BUDGET),
                                 b_curve[i, b], c_curve[i, b]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    x = (np.arange(N_BINS) + 1) * BIN_STEPS / 1e9
    for curve, title in ((b_curve, "Existing fast CleanRL-style JAX baseline"),
                         (c_curve, "Neural RND candidate")):
        mean = curve.mean(axis=0)
        # Seed bootstrap for the pointwise band; no frame/episode pseudoreplication.
        rng = np.random.default_rng(20261111)
        idx = rng.integers(0, len(seeds), (1500, len(seeds)))
        band = np.quantile(curve[idx].mean(axis=1), [0.025, 0.975], axis=0)
        line, = ax.plot(x, mean, label=title)
        ax.fill_between(x, band[0], band[1], color=line.get_color(), alpha=.15)
    ax.set(xlabel="Agent steps per independent seed (billions)",
           ylabel="Extrinsic score per finished episode",
           title=f"Pixel Montezuma, 1B matched-seed training curves (n={len(seeds)})")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "matched_1b_extrinsic_curves.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
