# With repeated trials at one x, the acquisition there reflects how many of that x's draws were "good".
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


# 12 draws at x=2 (all high) and 12 at x=8 (4 high, 8 low). With the good-group size set to 16, the
# good group is {12 at x=2, 4 at x=8} and the rest is {8 at x=8}.
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=0))
v2 = substream(0, "x2")
for _ in range(12):
    add(study, 2.0, v2.uniform(90, 100))    # x=2: all 12 in the good group
vh = substream(0, "x8_high")
for _ in range(4):
    add(study, 8.0, vh.uniform(70, 75))     # x=8: 4 in the good group
vl = substream(0, "x8_low")
for _ in range(8):
    add(study, 8.0, vl.uniform(10, 20))     # x=8: 8 in the rest group

sampler = TPESampler(seed=0)
trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
below, above = _split_trials(study, trials, 16, False)   # good-group size 16
l = sampler._build_parzen_estimator(study, SPACE, below, handle_below=True)
g = sampler._build_parzen_estimator(study, SPACE, above, handle_below=False)
acq = sampler._compute_acquisition_func({"x": np.array([2.0, 8.0])}, l, g)

print("x   good/rest draws   acq = log l - log g")
print(f"2.0   12 / 0          {acq[0]:.4f}   (12/12 of this x's draws reached the top quantile)")
print(f"8.0    4 / 8          {acq[1]:.4f}   ( 4/12 of this x's draws reached the top quantile)")
