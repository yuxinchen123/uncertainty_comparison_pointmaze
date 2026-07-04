# Each ask draws n_ei_candidates from l(x) (good) and returns argmax of log l - log g
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 60 completed trials on a bump peaked at x=8 in [0,10]
space = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}
sampler = TPESampler(seed=0)
study = optuna.create_study(direction="maximize", sampler=sampler)
rng = np.random.default_rng(0)
for _ in range(60):
    x = rng.uniform(0, 10)
    study.add_trial(optuna.trial.create_trial(
        params={"x": x}, distributions=space,
        value=float(np.exp(-((x - 8.0) ** 2) / 2))))

# reproduce one ask through the sampler internals
trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
n_below = sampler._gamma(len(trials))                              # good count = gamma(n)
below, above = _split_trials(study, trials, n_below, False)
mpe_below = sampler._build_parzen_estimator(study, space, below, handle_below=True)
mpe_above = sampler._build_parzen_estimator(study, space, above, handle_below=False)

cands = mpe_below.sample(sampler._rng.rng, sampler._n_ei_candidates)   # drawn from l(x)
acq = sampler._compute_acquisition_func(cands, mpe_below, mpe_above)   # log l - log g
ret = TPESampler._compare(cands, acq)                                  # argmax
print("candidates drawn from l(x):", cands["x"].size)
print("acq equals log l - log g   :",
      np.allclose(acq, mpe_below.log_pdf(cands) - mpe_above.log_pdf(cands)))
print("returned x == argmax cand  :", np.isclose(ret["x"], cands["x"][np.argmax(acq)]))
print("returned x (near peak 8)   :", round(ret["x"], 3))
