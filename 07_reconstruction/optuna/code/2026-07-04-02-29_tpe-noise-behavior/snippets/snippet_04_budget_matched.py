# Spend a fixed seed-run budget three ways: single-seed best-by-value is inflated; averaging fixes it.
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
    # single-peak landscape: true mean reward m(x), max 50 at x=8
    return 50.0 * np.exp(-((np.asarray(x) - 8.0) ** 2) / 2.0)


def one_seed(rng, m):
    # bimodal per-seed reward: success near 80 w.p. m/80 else collapse near 0 (mean over seeds = m)
    return float(rng.normal(80, 10)) if rng.random() < m / 80.0 else float(rng.normal(0, 3))


def run(seed, n_trials, seeds_per_trial):
    # run TPE with each trial value a mean of `seeds_per_trial` bimodal draws; return best-by-value stats
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    for i in range(n_trials):
        trial = study.ask(SPACE)
        x = trial.params["x"]
        vals = [one_seed(substream(seed, "r", i, s), float(true_mean(x))) for s in range(seeds_per_trial)]
        study.tell(trial, float(np.mean(vals)))
    bx = study.best_trial.params["x"]
    return study.best_trial.value, float(true_mean(bx))


# 300 seed-runs three ways, averaged over 3 base seeds: report inflation and the true mean of the picked x.
print("design (300 seed-runs)  | best value | m(picked x) | inflation")
for label, n_trials, k in [("300 x 1 seed        ", 300, 1),
                           ("100 x 3-seed mean   ", 100, 3),
                           (" 30 x 10-seed mean  ", 30, 10)]:
    bv = [run(s, n_trials, k) for s in (0, 1, 2)]
    best_value = np.mean([v for v, _ in bv])
    m_picked = np.mean([m for _, m in bv])
    print(f"{label} | {best_value:>10.2f} | {m_picked:>11.2f} | {best_value - m_picked:>9.2f}")
