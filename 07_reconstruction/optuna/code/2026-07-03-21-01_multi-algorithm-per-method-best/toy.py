"""Shared toy reward simulation used by all verification scripts.

Models the train-run-3.2.1 objective: final training-episode reward as a function
of optimizer method, bonus readout, log10(beta), and (for sgd1t only) eta0 and t0.
The three methods have different peak heights and peak locations so that one method
(sgd1t) is clearly best; this is what drives the allocation experiment in script 02.
Noise is additive Gaussian with standard deviation 6, drawn from a per-call keyed
generator so every call is reproducible.
"""

import hashlib

import numpy as np

# True mean-reward structure per method. center is in log10(beta); amplitude is the
# peak mean reward; width is the Gaussian bump standard deviation in decades of beta.
METHOD_PEAK = {
    "adam": {"center": 2.0, "amplitude": 20.0, "width": 0.65},     # peak 20 at beta=1e2
    "adagrad": {"center": 1.0, "amplitude": 27.0, "width": 0.65},  # peak 27 at beta=1e1
    "sgd1t": {"center": 2.0, "amplitude": 45.0, "width": 0.65},    # peak 45 at beta=1e2
}

# The methods and their true best mean reward (used by scripts to compare found-vs-truth).
# adam's best cell is readout=l2 (adds +3); sgd1t's best cell is eta0=1e-2, t0=1e4.
TRUE_PEAK = {"adam": 23.0, "adagrad": 27.0, "sgd1t": 45.0}


def substream(base_seed, *parts):
    """Return one numpy generator keyed by a stable string, so each named quantity is independent."""
    # build a stable key string from the base seed and all naming parts
    key = "::".join(str(p) for p in (base_seed, *parts))
    # derive a 32-bit seed by hashing the key, so unrelated quantities never share a stream
    digest = int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF
    return np.random.default_rng(digest)


def reward_mean(method, readout, log10_beta, eta0=None, t0=None):
    """Return the noise-free mean reward for one configuration (the true landscape)."""
    # look up this method's Gaussian bump in log10(beta)
    peak = METHOD_PEAK[method]
    # bump value: amplitude at the center beta, decaying with a Gaussian shape away from it
    bump = peak["amplitude"] * np.exp(-0.5 * ((log10_beta - peak["center"]) / peak["width"]) ** 2)
    mean = bump
    # adam gains +3 when the bonus readout is l2 (only adam has a readout effect)
    if method == "adam" and readout == "l2":
        mean += 3.0
    # sgd1t has a mild extra dependence on eta0 and t0, peaking at eta0=1e-2 and t0=1e4
    if method == "sgd1t":
        # eta0 penalty grows with the squared log-distance from 1e-2 (0 at the optimum)
        eta_pen = 4.0 * (np.log10(eta0) - (-2.0)) ** 2
        # t0 penalty grows with the log-distance from 1e4 (0 at the optimum, ~2 at 1e3)
        t0_pen = 2.0 * abs(np.log10(t0) - 4.0)
        mean += -eta_pen - t0_pen
    return float(mean)


def reward(method, readout, log10_beta, eta0=None, t0=None, base_seed=0):
    """Return one noisy reward draw for a configuration; noise SD 6, reproducible by config."""
    # compute the noise-free mean for this configuration
    mean = reward_mean(method, readout, log10_beta, eta0, t0)
    # draw one additive-noise sample from a generator keyed by the full configuration
    gen = substream(base_seed, "reward_noise", method, readout, round(log10_beta, 6), eta0, t0)
    noise = gen.normal(0.0, 6.0)
    return mean + noise
