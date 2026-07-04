"""Confirm empirically that at each ask TPE draws n_ei_candidates from the good density l(x)
and returns the candidate maximizing log l(x) - log g(x).

We drive the installed TPESampler's own internal methods on a real study, so the numbers
are produced by optuna 4.9.0 code, not a re-implementation.
"""

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_study_with_trials(n_trials, seed):
    """Create a maximize study over one float and add n_trials completed trials (peak at x=8)."""
    # Objective is a smooth bump peaked at x=8 in [0,10]; deterministic so the split is clear.
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        x = rng.uniform(0.0, 10.0)
        value = float(np.exp(-((x - 8.0) ** 2) / (2 * 1.0**2)))
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": x},
                distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
                value=value,
            )
        )
    return study


def main():
    """Reproduce one TPE ask through the sampler's internals and check each documented step."""
    print("optuna version:", optuna.__version__)
    n_trials = 60
    sampler = TPESampler(seed=0)
    study = build_study_with_trials(n_trials, seed=0)
    search_space = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}

    # Reproduce the body of TPESampler._sample: fetch finished trials and split good/rest.
    trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
    n = len(trials)
    n_below = sampler._gamma(n)
    from optuna.samplers._tpe.sampler import _split_trials
    below_trials, above_trials = _split_trials(study, trials, n_below, False)
    print(f"\nn finished = {n}, gamma(n) = n_below = {n_below} "
          f"(good={len(below_trials)}, rest={len(above_trials)})")

    # Build the two Parzen estimators exactly as the sampler does.
    mpe_below = sampler._build_parzen_estimator(study, search_space, below_trials, handle_below=True)
    mpe_above = sampler._build_parzen_estimator(study, search_space, above_trials, handle_below=False)

    # STEP A: candidates are drawn from the BELOW (good) density l(x); count must be n_ei_candidates.
    samples = mpe_below.sample(sampler._rng.rng, sampler._n_ei_candidates)
    print(f"\nn_ei_candidates = {sampler._n_ei_candidates}; drawn candidate count = {samples['x'].size}")
    print("candidates drawn from l(x) (good) -> mean of candidates:",
          round(float(samples["x"].mean()), 3),
          "| mean of good observations:", round(float(np.mean([t.params['x'] for t in below_trials])), 3))

    # STEP B: acquisition = log l(x) - log g(x), recomputed independently to confirm the formula.
    acq_from_sampler = sampler._compute_acquisition_func(samples, mpe_below, mpe_above)
    log_l = mpe_below.log_pdf(samples)
    log_g = mpe_above.log_pdf(samples)
    acq_independent = log_l - log_g
    print("\nacq == (log l - log g) exactly:",
          bool(np.allclose(acq_from_sampler, acq_independent)))

    # STEP C: the returned parameter is the candidate at argmax of the acquisition.
    ret = TPESampler._compare(samples, acq_from_sampler)
    best_idx = int(np.argmax(acq_independent))
    print("\nargmax candidate index         :", best_idx)
    print("returned x                     :", round(ret["x"], 6))
    print("candidate x at argmax          :", round(float(samples["x"][best_idx]), 6))
    print("returned == argmax candidate   :", np.isclose(ret["x"], samples["x"][best_idx]))

    # Show the candidate table sorted by acquisition so the argmax pick is visible.
    order = np.argsort(-acq_independent)
    print("\ntop 5 candidates by acquisition (x, log l, log g, acq=log l - log g):")
    for i in order[:5]:
        print(f"  x={samples['x'][i]:7.4f}  log_l={log_l[i]:8.4f}  "
              f"log_g={log_g[i]:8.4f}  acq={acq_independent[i]:8.4f}")


if __name__ == "__main__":
    main()
