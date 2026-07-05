# How many bad draws at a good location does it take to move TPE's proposal off the peak?
import hashlib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)
SPACE = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string (order-independent)
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def add(study, x, value):
    # add one already-evaluated (x, value) trial as a COMPLETE trial
    study.add_trial(optuna.trial.create_trial(
        params={"x": float(x)}, distributions=SPACE, value=float(value)))


def build(n_bad):
    # good group = 20 observations near x=8 (values 45-55); 180 background trials (values ~10) make
    # gamma=ceil(0.1*n)=20 so the 20 good ARE the good group; then n_bad bad draws (value ~0) at x=8
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
    bx, bv = substream(0, "bg_x"), substream(0, "bg_val")
    for _ in range(180):
        add(study, bx.uniform(0, 10), bv.normal(10, 4))
    gx, gv = substream(0, "good_x"), substream(0, "good_val")
    for _ in range(20):
        add(study, np.clip(gx.normal(8.0, 0.6), 0, 10), gv.uniform(45, 55))
    br = substream(0, "bad_val")
    for _ in range(n_bad):
        add(study, 8.0, abs(br.normal(0, 0.3)))
    return study


def acquisition(study, grid):
    # drive the installed sampler internals: split good/rest by value, fit l and g, return log l - log g
    sampler = TPESampler(seed=0)
    trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
    below, above = _split_trials(study, trials, sampler._gamma(len(trials)), False)
    l = sampler._build_parzen_estimator(study, SPACE, below, handle_below=True)
    g = sampler._build_parzen_estimator(study, SPACE, above, handle_below=False)
    return sampler._compute_acquisition_func({"x": grid}, l, g)


grid = np.linspace(0, 10, 1001)
i8 = int(np.argmin(np.abs(grid - 8.0)))
print("bad draws at x=8 | argmax x | acq(x=8) - max(acq)")
for n_bad in (0, 1, 3, 6):
    acq = acquisition(build(n_bad), grid)
    print(f"{n_bad:>16} | {grid[int(np.argmax(acq))]:>8.3f} | {acq[i8] - acq.max():>18.4f}")
