"""Parallel asks with and without constant_liar on the same 30-trial history.

We seed a study with ~30 informative COMPLETE trials on the toy bump landscape, then ask a batch
of 8 trials WITHOUT reporting any result (so the 8 sit in the RUNNING state). We do this once with
TPESampler(seed=0) and once with TPESampler(seed=0, constant_liar=True), and measure the batch's
spread (mean pairwise distance) and where the points sit.
"""

import hashlib
import itertools

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.distributions import FloatDistribution

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOW = np.array([-6.0, -3.0])
HIGH = np.array([-2.0, -1.0])
PEAK = np.array([-6.0, -2.0])
RANGE = HIGH - LOW
DIST = {"log10_ridge": FloatDistribution(-6.0, -2.0), "log10_beta": FloatDistribution(-3.0, -1.0)}


def substream(base_seed, *parts):
    """Return one numpy generator per named quantity, keyed by a stable string."""
    # Hash the joined name so adding a quantity never shifts an existing draw.
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bump(point):
    """Return the noiseless bump height (max 50) peaked at PEAK, width 1 decade."""
    d2 = np.sum((point - PEAK) ** 2)
    return 50.0 * np.exp(-d2 / (2 * 1.0**2))


def seed_study(sampler, n_seed_trials, base_seed):
    """Create a maximize study and add n_seed_trials completed trials on the bump landscape."""
    # Points spread over the box; values are the bump plus small noise so the optimum is clear.
    study = optuna.create_study(direction="maximize", sampler=sampler)
    pt_rng = substream(base_seed, "seed_points")
    for i in range(n_seed_trials):
        p = LOW + pt_rng.uniform(0.0, 1.0, size=2) * RANGE
        noise = substream(base_seed, "seed_noise", i).normal(0.0, 2.0)
        study.add_trial(
            optuna.trial.create_trial(
                params={"log10_ridge": p[0], "log10_beta": p[1]},
                distributions=DIST,
                value=float(bump(p) + noise),
            )
        )
    return study


def ask_batch(constant_liar, n_batch, n_seed_trials, base_seed):
    """Ask n_batch trials without reporting results; return their (ridge, beta) points."""
    # Same seed for both settings so any spread difference is caused only by constant_liar.
    sampler = TPESampler(seed=base_seed, constant_liar=constant_liar)
    study = seed_study(sampler, n_seed_trials, base_seed)
    pts = []
    for _ in range(n_batch):
        # ask() with fixed distributions suggests+stores params and leaves the trial RUNNING.
        trial = study.ask(DIST)
        pts.append([trial.params["log10_ridge"], trial.params["log10_beta"]])
    return np.array(pts)


def mean_pairwise_distance(pts):
    """Return the mean Euclidean distance over all unordered pairs of points in the batch."""
    # Average distance across the C(n,2) pairs; a small value means the batch is clustered.
    ds = [np.sqrt(np.sum((a - b) ** 2)) for a, b in itertools.combinations(pts, 2)]
    return float(np.mean(ds))


def main():
    """Run the batch ask both ways and print the spread and point locations."""
    print("optuna version:", optuna.__version__)
    n_seed_trials, n_batch, base_seed = 30, 8, 0
    print(f"history = {n_seed_trials} completed trials; batch = {n_batch} asks with no results reported")
    print(f"peak at {tuple(PEAK)}; box ridge[-6,-2] beta[-3,-1]\n")

    for label, cl in [("constant_liar=False", False), ("constant_liar=True", True)]:
        pts = ask_batch(cl, n_batch, n_seed_trials, base_seed)
        spread = mean_pairwise_distance(pts)
        dist_to_peak = [np.sqrt(np.sum((p - PEAK) ** 2)) for p in pts]
        print(f"== {label} ==")
        print(f"mean pairwise distance in (log ridge, log beta): {spread:.4f}")
        print(f"batch centroid: ({pts[:,0].mean():.3f}, {pts[:,1].mean():.3f})   "
              f"mean dist to peak: {np.mean(dist_to_peak):.3f}")
        print("the 8 asked points (log10_ridge, log10_beta):")
        for p in pts:
            print(f"    ({p[0]:7.3f}, {p[1]:7.3f})")
        print()

    # Direct side-by-side summary of the spread.
    pts_off = ask_batch(False, n_batch, n_seed_trials, base_seed)
    pts_on = ask_batch(True, n_batch, n_seed_trials, base_seed)
    print("summary of batch spread (mean pairwise distance):")
    print(f"  constant_liar=False: {mean_pairwise_distance(pts_off):.4f}  (clustered)")
    print(f"  constant_liar=True : {mean_pairwise_distance(pts_on):.4f}  (spread out)")
    print(f"  ratio on/off       : {mean_pairwise_distance(pts_on)/mean_pairwise_distance(pts_off):.2f}x")


if __name__ == "__main__":
    main()
