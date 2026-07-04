# TPESampler defaults, the good/rest split size, and the trial-age weighting (optuna 4.9.0)
import math
import numpy as np
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import default_gamma, default_weights

s = TPESampler()                                   # all defaults
print("n_startup_trials:", s._n_startup_trials)    # random search until this many finish
print("n_ei_candidates :", s._n_ei_candidates)     # candidates drawn per ask
print("multivariate    :", s._multivariate)        # per-parameter (independent) by default
print("constant_liar   :", s._constant_liar)

# default_gamma(n) = min(ceil(0.1 n), 25): the number of "good" trials, capped at 25.
print("gamma at n=100,600,9600:", default_gamma(100), default_gamma(600), default_gamma(9600))

# default_weights(n): below 25 finished trials every trial weighs 1; at 25+ the oldest
# n-25 trials ramp from 1/n up to 1 while the most recent 25 keep weight 1.
w = default_weights(30)
print("weights(30) oldest-first, first4:", np.round(w[:4], 3), "last4:", np.round(w[-4:], 3))
