"""Mini-TPE from scratch (numpy only) vs pure random vs the real optuna TPESampler.

The toy landscape is a single smooth bump peaked at (log10_ridge=-6, log10_beta=-2) over the
box [-6,-2] x [-3,-1], height 50, width ~1 decade, with additive noise (SD 10) per evaluation.
We count, out of 60 trials, how many land within half a decade of the peak.
"""

import hashlib

import numpy as np

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Search box: log10_ridge in [-6,-2], log10_beta in [-3,-1]; peak at the (-6,-2) corner/center.
LOW = np.array([-6.0, -3.0])
HIGH = np.array([-2.0, -1.0])
PEAK = np.array([-6.0, -2.0])
RANGE = HIGH - LOW
N_STARTUP = 10
N_TRIALS = 60
N_CANDIDATES = 24
TOP_FRACTION = 0.25


def substream(base_seed, *parts):
    """Return one numpy generator per named quantity, keyed by a stable string."""
    # Hash the joined name so adding a new quantity never shifts an existing draw.
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bump(point):
    """Return the noiseless bump height (max 50) at a 2-D point in log space."""
    # Isotropic Gaussian bump, width 1 decade, centered on PEAK.
    d2 = np.sum((point - PEAK) ** 2)
    return 50.0 * np.exp(-d2 / (2 * 1.0**2))


def evaluate(point, base_seed, method, trial_i):
    """Return the noisy objective (bump + N(0,10)) at a point, noise keyed per trial."""
    # One fresh noise draw per (seed, method, trial), so runs are reproducible and independent.
    noise = substream(base_seed, method, "noise", trial_i).normal(0.0, 10.0)
    return bump(point) + noise


def near_peak(point):
    """Return True if the point is within half a decade (Euclidean) of the peak."""
    return np.sqrt(np.sum((point - PEAK) ** 2)) < 0.5


def kde_logpdf(x, data, bw):
    """Return log of a 1-D Gaussian-kernel density at points x, equal-weight kernels at data."""
    # Mixture of len(data) Gaussians; use logsumexp for numerical stability.
    z = (x[:, None] - data[None, :]) / bw  # (n_x, n_data)
    log_kernels = -0.5 * z**2 - np.log(bw) - 0.5 * np.log(2 * np.pi)
    m = np.max(log_kernels, axis=1)
    return m + np.log(np.mean(np.exp(log_kernels - m[:, None]), axis=1))


def sample_from_kde(data, bw, size, rng):
    """Draw size samples from a 1-D Gaussian KDE, clipped to the parameter's bounds, per param."""
    # Pick a kernel center uniformly, then add Gaussian noise of width bw.
    idx = rng.integers(0, len(data), size=size)
    return data[idx] + rng.normal(0.0, bw, size=size)


def mini_tpe_run(base_seed):
    """Run the from-scratch mini-TPE for N_TRIALS and return the array of evaluated points."""
    points = []  # list of 2-D points, one per trial
    values = []  # objective value per trial
    # (a) 10 random startup points, uniform over the box.
    startup_rng = substream(base_seed, "mini_tpe", "startup")
    for i in range(N_STARTUP):
        p = LOW + startup_rng.uniform(0.0, 1.0, size=2) * RANGE
        points.append(p)
        values.append(evaluate(p, base_seed, "mini_tpe", i))

    # Remaining trials use the fitted good/rest densities.
    for i in range(N_STARTUP, N_TRIALS):
        pts = np.array(points)
        vals = np.array(values)
        # (b) split by top-25% objective (maximize) into good/rest.
        n = len(vals)
        n_good = max(1, int(np.ceil(TOP_FRACTION * n)))
        order = np.argsort(-vals)  # best first
        good = pts[order[:n_good]]
        rest = pts[order[n_good:]]
        cand_rng = substream(base_seed, "mini_tpe", "cand", i)
        # Build one candidate set of N_CANDIDATES joint points drawn from the good density.
        cand = np.empty((N_CANDIDATES, 2))
        score = np.zeros(N_CANDIDATES)
        for d in range(2):
            # (c) fixed bandwidth = range / sqrt(n_group), per parameter per group.
            bw_good = RANGE[d] / np.sqrt(len(good))
            bw_rest = RANGE[d] / np.sqrt(max(len(rest), 1))
            # (d) draw candidates from good density l, clip to bounds.
            c = np.clip(sample_from_kde(good[:, d], bw_good, N_CANDIDATES, cand_rng),
                        LOW[d], HIGH[d])
            cand[:, d] = c
            log_l = kde_logpdf(c, good[:, d], bw_good)
            log_g = kde_logpdf(c, rest[:, d], bw_rest)
            score += log_l - log_g  # summed over parameters
        # (d cont.) pick the candidate maximizing summed log l - log g.
        best = cand[int(np.argmax(score))]
        points.append(best)
        values.append(evaluate(best, base_seed, "mini_tpe", i))
    return np.array(points)


def random_run(base_seed):
    """Run pure random search for N_TRIALS and return the array of evaluated points."""
    # Uniform over the box; a fresh generator keyed for the random method.
    rng = substream(base_seed, "random", "points")
    points = []
    for i in range(N_TRIALS):
        p = LOW + rng.uniform(0.0, 1.0, size=2) * RANGE
        points.append(p)  # value not needed for the hit count
    return np.array(points)


def real_tpe_run(base_seed):
    """Run the installed optuna TPESampler for N_TRIALS and return the array of evaluated points."""
    # Same landscape and noise convention as mini-TPE, driven through optuna.
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=base_seed))
    points = []

    def objective(trial):
        r = trial.suggest_float("log10_ridge", LOW[0], HIGH[0])
        b = trial.suggest_float("log10_beta", LOW[1], HIGH[1])
        p = np.array([r, b])
        points.append(p)
        return evaluate(p, base_seed, "real_tpe", trial.number)

    study.optimize(objective, n_trials=N_TRIALS)
    return np.array(points)


def main():
    """Run all three methods at three base seeds and print the near-peak hit counts."""
    print("optuna version:", optuna.__version__)
    print(f"landscape: peak at {tuple(PEAK)}, box {list(zip(LOW, HIGH))}, "
          f"height 50, width 1 decade, noise SD 10")
    print(f"budget = {N_TRIALS} trials; 'hit' = within 0.5 (half a decade) of the peak\n")
    seeds = [0, 1, 2]
    print(f"{'seed':>4} {'mini_tpe hits':>14} {'random hits':>12} {'real_tpe hits':>14}")
    tot = {"mini": 0, "rand": 0, "real": 0}
    for s in seeds:
        mini_pts = mini_tpe_run(s)
        rand_pts = random_run(s)
        real_pts = real_tpe_run(s)
        mini_hits = int(sum(near_peak(p) for p in mini_pts))
        rand_hits = int(sum(near_peak(p) for p in rand_pts))
        real_hits = int(sum(near_peak(p) for p in real_pts))
        tot["mini"] += mini_hits
        tot["rand"] += rand_hits
        tot["real"] += real_hits
        print(f"{s:>4} {mini_hits:>14} {rand_hits:>12} {real_hits:>14}")
    print(f"\n{'sum':>4} {tot['mini']:>14} {tot['rand']:>12} {tot['real']:>14}")
    # Also report the closest point each method reached, averaged over seeds, for seed 0.
    print("\nseed 0 closest-approach distance to peak:")
    for name, fn, meth in [("mini_tpe", mini_tpe_run, "mini"), ("random", random_run, "rand"),
                           ("real_tpe", real_tpe_run, "real")]:
        pts = fn(0)
        dmin = min(np.sqrt(np.sum((p - PEAK) ** 2)) for p in pts)
        print(f"  {name:>9}: {dmin:.3f}")


if __name__ == "__main__":
    main()
