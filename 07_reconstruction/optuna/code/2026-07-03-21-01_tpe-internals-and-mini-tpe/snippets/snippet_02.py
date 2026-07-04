# The numeric Parzen mixture: one truncated-normal per observation + one prior; magic clip
import numpy as np
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.distributions import FloatDistribution

# defaults folded from the (now-deprecated) knobs: prior_weight=1, magic_clip=True, endpoints=False
params = _ParzenEstimatorParameters(
    prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False,
    weights=lambda n: np.ones(n), multivariate=False, categorical_distance_func={},
)
space = {"x": FloatDistribution(low=0.0, high=10.0)}

# three observations -> 3 kernels + 1 prior kernel (mean at range center 5, sigma = full range 10)
mpe = _ParzenEstimator({"x": np.array([2.0, 3.0, 8.0])}, space, params)
comp = mpe._mixture_distribution.distributions[0]
print("mixture weights:", np.round(mpe._mixture_distribution.weights, 3), "(last = prior)")
print("component mus  :", np.round(comp.mu, 3), "(last = prior center 5.0)")
print("component sigma:", np.round(comp.sigma, 3), "(last = prior sigma = range 10)")

# magic clip floors the smallest sigma at range/min(100, 1+n_kernels); here 10/5 = 2.0
obs = {"x": np.array([5.000, 5.001, 1.0])}   # a near-duplicate pair would give sigma ~ 0.001
sig = _ParzenEstimator(obs, space, params)._mixture_distribution.distributions[0].sigma
print("sigmas with magic clip:", np.round(sig, 3), "-> smallest floored to 2.0, not 0.001")
