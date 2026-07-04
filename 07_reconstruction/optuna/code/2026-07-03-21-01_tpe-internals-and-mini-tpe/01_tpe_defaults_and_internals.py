"""Print TPESampler defaults and Parzen-estimator internals from installed optuna 4.9.0.

Everything here is read from the live objects/functions in the installed package, then
exercised on tiny toy observations so the printed numbers are the code's actual output.
"""

import numpy as np

import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import default_gamma, default_weights, hyperopt_default_gamma
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.distributions import FloatDistribution, CategoricalDistribution


def show_constructor_defaults():
    """Print the TPESampler constructor default values and the deprecated fall-backs."""
    # Instantiate with no arguments and read the attributes the constructor stored.
    s = TPESampler()
    print("== TPESampler() default attributes ==")
    print("n_startup_trials :", s._n_startup_trials)
    print("n_ei_candidates  :", s._n_ei_candidates)
    print("gamma is default_gamma :", s._gamma is default_gamma)
    print("multivariate     :", s._multivariate)
    print("group            :", s._group)
    print("constant_liar    :", s._constant_liar)
    # The deprecated knobs are folded into the ParzenEstimatorParameters namedtuple.
    p = s._parzen_estimator_parameters
    print("prior_weight (fallback)        :", p.prior_weight)
    print("consider_magic_clip (fallback) :", p.consider_magic_clip)
    print("consider_endpoints (fallback)  :", p.consider_endpoints)
    print("weights is default_weights     :", p.weights is default_weights)


def show_gamma():
    """Print the good/rest split size n_below = gamma(n) for the default and hyperopt gamma."""
    # default_gamma = min(ceil(0.1 n), 25); hyperopt = min(ceil(0.25 sqrt(n)), 25).
    print("\n== gamma(n): number of 'good' (below) trials ==")
    print(f"{'n':>6} {'default_gamma':>14} {'hyperopt_gamma':>15}")
    for n in [5, 10, 20, 50, 100, 250, 300, 600, 9600]:
        print(f"{n:>6} {default_gamma(n):>14} {hyperopt_default_gamma(n):>15}")


def show_weights():
    """Print default_weights(n): how older trials are down-weighted and above what count."""
    # Below 25 trials the weights are all 1; at >=25 the oldest n-25 ramp from 1/n..1.
    print("\n== default_weights(n): per-observation weight, oldest first ==")
    for n in [3, 24, 25, 30, 40]:
        w = default_weights(n)
        head = ", ".join(f"{x:.3f}" for x in w[:4])
        tail = ", ".join(f"{x:.3f}" for x in w[-4:])
        print(f"n={n:>3}  len={len(w):>3}  first4=[{head}]  last4=[{tail}]  "
              f"min={w.min():.4f} max={w.max():.4f}")


def show_numeric_mixture():
    """Fit a 1-D Parzen estimator over a few numeric observations and print the mixture."""
    # Build the ParzenEstimatorParameters with the documented defaults.
    params = _ParzenEstimatorParameters(
        prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False,
        weights=default_weights, multivariate=False, categorical_distance_func={},
    )
    # Three observed values of a float in [0, 10]; expect 3 kernels + 1 prior = 4 components.
    obs = {"x": np.array([2.0, 3.0, 8.0])}
    space = {"x": FloatDistribution(low=0.0, high=10.0)}
    mpe = _ParzenEstimator(obs, space, params)
    dist = mpe._mixture_distribution
    comp = dist.distributions[0]  # one _BatchedDistributions object for param "x"
    print("\n== numeric mixture over x in [0,10], observations [2,3,8] ==")
    print("num mixture weights:", np.round(dist.weights, 4), " (last is the prior)")
    print("component mus   :", np.round(comp.mu, 4), " (last mu = center 5.0 = prior)")
    print("component sigmas:", np.round(comp.sigma, 4), " (last sigma = range 10 = prior)")
    print("n_components =", len(comp.mu), "= n_obs(3) + 1 prior")


def show_magic_clip():
    """Show that magic clip raises the smallest Gaussian sigma off a near-duplicate pair."""
    # Two observations 0.001 apart in [0,10]: without a floor one sigma would be ~0.001.
    space = {"x": FloatDistribution(low=0.0, high=10.0)}
    obs = {"x": np.array([5.000, 5.001, 1.0])}
    params_on = _ParzenEstimatorParameters(1.0, True, False, default_weights, False, {})
    params_off = _ParzenEstimatorParameters(1.0, False, False, default_weights, False, {})
    sig_on = _ParzenEstimator(obs, space, params_on)._mixture_distribution.distributions[0].sigma
    sig_off = _ParzenEstimator(obs, space, params_off)._mixture_distribution.distributions[0].sigma
    # minsigma = (high-low)/min(100, 1+n_kernels); n_kernels = n_obs+1 = 4 -> range/5 = 2.0.
    print("\n== magic clip on near-duplicate pair [5.000, 5.001, 1.0] in [0,10] ==")
    print("sigmas WITH magic clip   :", np.round(sig_on, 5))
    print("sigmas WITHOUT magic clip:", np.round(sig_off, 5))
    print("minsigma floor = range/min(100,1+n_kernels) = 10/5 =", 10.0 / min(100.0, 1 + 4))


def show_categorical():
    """Fit categorical weights and show the additive prior smoothing over choices."""
    # Choices A,B,C; observe A three times, B once, never C.
    params = _ParzenEstimatorParameters(1.0, True, False, default_weights, False, {})
    space = {"c": CategoricalDistribution(choices=["A", "B", "C"])}
    obs = {"c": np.array([0, 0, 0, 1])}  # internal repr: A=0, B=1, C=2
    mpe = _ParzenEstimator(obs, space, params)
    comp = mpe._mixture_distribution.distributions[0]
    print("\n== categorical over [A,B,C], observed A,A,A,B (C never seen) ==")
    print("per-kernel category weights (rows=kernels incl. prior, cols=A,B,C):")
    print(np.round(comp.weights, 4))
    # Mixture-weighted marginal probability of each category (what actually gets sampled).
    marg = mpe._mixture_distribution.weights @ comp.weights
    print("mixture-marginal P(category):", np.round(marg, 4),
          " -> C keeps nonzero probability from the prior")


if __name__ == "__main__":
    print("optuna version:", optuna.__version__)
    show_constructor_defaults()
    show_gamma()
    show_weights()
    show_numeric_mixture()
    show_magic_clip()
    show_categorical()
