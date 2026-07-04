"""Shared bimodal per-seed reward simulation for the Optuna seed-allocation tutorial.

Mirrors the real "train run 3.1.2" sweep: 12 hyperparameter combinations ("cells"),
each with a bimodal per-seed outcome (a seed either learns to reach the goal or does not).
A "seed run" costs 1 unit of computation; in reality one seed run is ~10-19 hours.
Every (cell, seed) outcome is fixed and reproducible via a per-quantity keyed substream.
"""

import hashlib

import numpy as np

# Target true means of the 12 cells, taken from the real 3.1.2 cell means.
TRUE_MEANS = [52.8, 35.2, 29.0, 26.1, 26.0, 17.9, 17.3, 9.6, 4.9, 1.0, 0.5, 0.1]
N_CELLS = len(TRUE_MEANS)

# Bimodal calibration: success -> Normal(HIGH, SUCCESS_SD); failure -> Normal(0, FAIL_SD).
# Mixture mean = p*HIGH + (1-p)*0 = p*HIGH, so p = target_mean / HIGH reproduces each target.
HIGH = 80.0
SUCCESS_SD = 10.0
FAIL_SD = 3.0
SUCCESS_PROB = [m / HIGH for m in TRUE_MEANS]

# The user's rule constants: keep a cell only if its mean + 1.96*sd/sqrt(n) is still >= this bar.
THRESHOLD = 35.0
Z_95 = 1.96
MIN_SEEDS = 10   # do not judge a cell before this many seeds
MAX_SEEDS = 50   # cap each cell here (the full grid would run 50 seeds for every cell)


def substream(base_seed, *parts):
    # One numpy generator per named quantity, keyed by a stable string, so adding a new
    # keyed quantity never shifts the draws of existing ones.
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def draw_reward(base_seed, cell_index, seed_index):
    # Draw the fixed, reproducible reward for one (cell, seed) pair from its own substream.
    rng = substream(base_seed, "cell", cell_index, "seed", seed_index)
    # Decide success vs failure for this seed, then draw the matching Normal.
    if rng.random() < SUCCESS_PROB[cell_index]:
        return float(rng.normal(HIGH, SUCCESS_SD))
    return float(rng.normal(0.0, FAIL_SD))


def upper_ci(rewards):
    # Upper end of the 95% confidence interval of the mean: mean + 1.96 * sd / sqrt(n).
    n = len(rewards)
    mean = float(np.mean(rewards))
    # Sample standard deviation (ddof=1); undefined for n<2, so report +inf there (never prune).
    if n < 2:
        return mean, float("inf"), mean
    sd = float(np.std(rewards, ddof=1))
    return mean, sd, mean + Z_95 * sd / np.sqrt(n)


def true_survivors():
    # The cells whose TRUE mean is at or above the 35 bar (the "correct" set to keep).
    return [i for i, m in enumerate(TRUE_MEANS) if m >= THRESHOLD]


if __name__ == "__main__":
    # Sanity check: empirical mean over 200 seeds should track each target mean.
    for i in range(N_CELLS):
        rewards = [draw_reward(0, i, s) for s in range(200)]
        print(f"cell {i:2d}  target {TRUE_MEANS[i]:5.1f}  p {SUCCESS_PROB[i]:.3f}  "
              f"empirical_mean(200 seeds) {np.mean(rewards):6.2f}")
    print("true survivors (mean>=35):", true_survivors())
