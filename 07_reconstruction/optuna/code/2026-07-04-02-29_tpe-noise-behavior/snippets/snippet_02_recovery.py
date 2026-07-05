# Does TPE return to the true-best region after unlucky early failures placed there? Yes, quickly.
import hashlib
import numpy as np
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
SPACE = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string (order-independent)
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def true_mean(x):
    # single-peak landscape: true mean m(x), max 50 at x=8 (the true best)
    return 50.0 * np.exp(-((np.asarray(x) - 8.0) ** 2) / 2.0)


def one_seed(rng, m):
    # bimodal per-seed reward: success near 80 w.p. m/80 else collapse near 0 (mean over seeds = m)
    return float(rng.normal(80, 10)) if rng.random() < m / 80.0 else float(rng.normal(0, 3))


def near_peak_fractions(seed, with_failures):
    # seed 20 informative random observations (+ optionally 3 forced failures at x=8), run TPE 100 trials,
    # return the fraction of proposals with |x-8|<1 in each successive 25-trial window
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    xr = substream(seed, "init_x")
    for i in range(20):
        x = float(xr.uniform(0, 10))
        study.add_trial(optuna.trial.create_trial(
            params={"x": x}, distributions=SPACE,
            value=one_seed(substream(seed, "init_r", i), float(true_mean(x)))))
    if with_failures:
        fr = substream(seed, "fail")
        for _ in range(3):
            study.add_trial(optuna.trial.create_trial(
                params={"x": 8.0}, distributions=SPACE, value=float(abs(fr.normal(0, 0.5)))))
    xs = []
    for i in range(100):
        trial = study.ask(SPACE)
        x = trial.params["x"]
        xs.append(x)
        study.tell(trial, one_seed(substream(seed, "reward", i), float(true_mean(x))))
    arr = np.asarray(xs)
    return [float(np.mean(np.abs(arr[lo:lo + 25] - 8.0) < 1.0)) for lo in (0, 25, 50, 75)]


# Average the per-window near-peak fraction over 3 base seeds, with and without the 3 forced failures at x=8.
for with_failures in (True, False):
    fr = np.mean([near_peak_fractions(s, with_failures) for s in (0, 1, 2)], axis=0)
    label = "with 3 forced failures at x=8" if with_failures else "without forced failures      "
    print(f"{label}: trials 1-25={fr[0]:.2f}  26-50={fr[1]:.2f}  51-75={fr[2]:.2f}  76-100={fr[3]:.2f}")
