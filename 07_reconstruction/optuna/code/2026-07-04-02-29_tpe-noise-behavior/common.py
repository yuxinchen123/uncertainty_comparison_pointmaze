"""Shared helpers for the TPE-under-noise verification scripts (optuna 4.9.0).

Every script in this folder imports from here so the reward model, the keyed RNG, and the
routine that drives the installed TPESampler's internals are defined once and identically.
"""

import hashlib

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One float parameter x in [0, 10] used by every landscape in this folder.
SPACE = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}


def substream(base, *parts):
    """Return one numpy generator per named quantity, keyed by a stable string (order-independent)."""
    # key = "base::part0::part1::..." hashed to a 32-bit seed so adding a new quantity never shifts old draws
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal_reward(rng, m):
    """One per-seed reward for a config with true mean m in [0,80]: Normal(80,10) w.p. m/80 else Normal(0,3)."""
    # bimodal single-seed outcome: the run either learns (near 80) or does not (near 0); mean over seeds equals m
    p = m / 80.0
    if rng.random() < p:
        return float(rng.normal(80.0, 10.0))
    return float(rng.normal(0.0, 3.0))


def peak_mean(x):
    """True mean-reward landscape for points 2 and 4: a single Gaussian bump peaking at 50 at x=8."""
    # m(x) = 50 * exp(-(x-8)^2 / 2) on [0,10]; max true mean is 50 at x=8, so p_success peaks at 50/80
    return 50.0 * np.exp(-((np.asarray(x) - 8.0) ** 2) / 2.0)


def region_reward(rng, x):
    """Two-region reward for point 3: A near x=3 is a consistent good learner; B near x=7 is the
    project's own bimodal regime (half its seeds succeed near 80, half collapse near 0)."""
    # region A (|x-3|<=0.5): Normal(60,5), true mean 60, low variance -- reliably good, no upper tail
    if abs(x - 3.0) <= 0.5:
        return float(rng.normal(60.0, 5.0))
    # region B (|x-7|<=0.5): success Normal(80,10) w.p. 0.5 else collapse Normal(0,3); true mean 40 (below A),
    # but its ~80 successes own the top decile of single draws
    if abs(x - 7.0) <= 0.5:
        if rng.random() < 0.5:
            return float(rng.normal(80.0, 10.0))
        return float(rng.normal(0.0, 3.0))
    # elsewhere: Normal(5,3)
    return float(rng.normal(5.0, 3.0))


def region_reward_task_original(rng, x):
    """The task's literal point-3 numbers, kept for the documented contrast that they do NOT flip:
    A near x=3 Normal(30,2) mean 30; B near x=7 is 80 w.p. 0.15 else Normal(15,3) mean 24.75; else Normal(5,2)."""
    # region A: consistent Normal(30,2), true mean 30
    if abs(x - 3.0) <= 0.5:
        return float(rng.normal(30.0, 2.0))
    # region B: exactly 80 w.p. 0.15 else Normal(15,3), true mean 24.75
    if abs(x - 7.0) <= 0.5:
        if rng.random() < 0.15:
            return 80.0
        return float(rng.normal(15.0, 3.0))
    # elsewhere: Normal(5,2)
    return float(rng.normal(5.0, 2.0))


def add_observation(study, x, value):
    """Add one already-evaluated (x, value) trial to the study as a COMPLETE trial."""
    # create_trial with a value defaults to state COMPLETE, so TPE will use it in the split
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": float(x)},
            distributions=SPACE,
            value=float(value),
        )
    )


def acquisition_on_grid(study, sampler, grid):
    """Drive the installed sampler's internals to return log l(x) - log g(x) on a grid of x values.

    Reproduces the body of TPESampler._sample (fetch finished trials, split good/rest by value,
    fit the two Parzen estimators, evaluate the acquisition) but on a supplied grid instead of the
    24 internally-drawn candidates. For a plain (non-log) FloatDistribution the internal parameter
    representation equals the external one, so the raw grid values can be passed straight through.
    """
    # split all COMPLETE trials into below/good (top gamma by value) and above/rest, exactly as _sample does
    trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
    n_below = sampler._gamma(len(trials))
    below, above = _split_trials(study, trials, n_below, False)
    # fit l from the good group and g from the rest, then evaluate log l - log g on the grid
    mpe_below = sampler._build_parzen_estimator(study, SPACE, below, handle_below=True)
    mpe_above = sampler._build_parzen_estimator(study, SPACE, above, handle_below=False)
    acq = sampler._compute_acquisition_func({"x": np.asarray(grid)}, mpe_below, mpe_above)
    return np.asarray(acq), len(below), len(above)


def run_tpe(base_seed, n_trials, value_fn, seeded_obs=(), n_startup_trials=10):
    """Run an ask/tell TPE study for n_trials, returning the list of proposed x in order.

    value_fn(trial_index, x) returns the observed value reported for that trial (single seed or a
    mean of seeds). seeded_obs is a list of (x, value) added as COMPLETE trials before the loop.
    n_startup_trials is TPE's random-startup count (10 is the optuna default); lower it for short runs.
    """
    # fresh seeded study; add any pre-existing observations first so TPE conditions on them
    sampler = TPESampler(seed=base_seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for x, value in seeded_obs:
        add_observation(study, x, value)
    # ask/tell loop: record each proposed x, report the value the objective returns for it
    xs = []
    for i in range(n_trials):
        trial = study.ask(SPACE)
        x = trial.params["x"]
        xs.append(x)
        study.tell(trial, value_fn(i, x))
    return study, sampler, xs
