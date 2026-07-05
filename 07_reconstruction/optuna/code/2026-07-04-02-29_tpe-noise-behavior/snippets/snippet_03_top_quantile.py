# Single-seed TPE targets the top-quantile draw, not the mean; averaging seeds re-aligns it.
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


def reward(rng, x):
    # region A near x=3: consistent, true mean 60. region B near x=7: half succeed near 80, half
    # collapse near 0, true mean 40 (lower) -- but its ~80 successes own the top decile of single draws.
    if abs(x - 3.0) <= 0.5:
        return float(rng.normal(60.0, 5.0))
    if abs(x - 7.0) <= 0.5:
        return float(rng.normal(80.0, 10.0)) if rng.random() < 0.5 else float(rng.normal(0.0, 3.0))
    return float(rng.normal(5.0, 3.0))


def region(x):
    # label a proposed x by which region it lands in
    return "A" if abs(x - 3.0) <= 0.5 else ("B" if abs(x - 7.0) <= 0.5 else "elsewhere")


def concentration(seed, n_trials, seeds_per_trial, window, startup):
    # run TPE, each trial value the mean of `seeds_per_trial` reward draws; report which of A/B got
    # more proposals in the last `window` trials
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, n_startup_trials=startup))
    xs = []
    for i in range(n_trials):
        trial = study.ask(SPACE)
        x = trial.params["x"]
        xs.append(x)
        vals = [reward(substream(seed, seeds_per_trial, i, s), x) for s in range(seeds_per_trial)]
        study.tell(trial, float(np.mean(vals)))
    tail = [region(x) for x in xs[-window:]]
    a, b = tail.count("A"), tail.count("B")
    return "A" if a > b else ("B" if b > a else "tie")


# Tally over 8 base seeds: single-seed (150 trials) vs 10-seed means (15 trials) -- same 150 seed-runs.
single = {"A": 0, "B": 0, "tie": 0}
mean10 = {"A": 0, "B": 0, "tie": 0}
for seed in range(8):
    single[concentration(seed, 150, 1, window=50, startup=10)] += 1
    mean10[concentration(seed, 15, 10, window=12, startup=3)] += 1
print("region of late-window concentration over 8 seeds (A: true mean 60 is best; B: true mean 40):")
print(f"  single seed     : B {single['B']}  A {single['A']}  tie {single['tie']}")
print(f"  mean of 10 seeds: B {mean10['B']}  A {mean10['A']}  tie {mean10['tie']}")
