"""Noisy toy reward landscape that mirrors the project's SAC exploration sweep.

Two searchable knobs on a log scale: ridge lambda in [1e-6, 1e-2] and beta in [1e-3, 1e-1].
A single smooth peak sits at ridge=1e-6, beta=1e-2 with height ~50 and width ~1 decade in
each axis, plus additive per-evaluation noise of SD ~10 drawn via the keyed-substream pattern
so a (noise_seed, ridge, beta) triple always yields the same noisy reward.
"""

import hashlib

import numpy as np

# True peak location, stated in log10 units (ridge=1e-6 -> -6, beta=1e-2 -> -2).
PEAK_LOG10_RIDGE = -6.0
PEAK_LOG10_BETA = -2.0
# Smooth-bump height and per-axis width (in decades).
PEAK_HEIGHT = 50.0
BUMP_WIDTH_DECADES = 1.0
# Additive noise standard deviation, mirroring the sweep's large per-seed spread.
NOISE_SD = 10.0


# Build one independent RNG per named quantity, keyed by a stable string.
def substream(base_seed, *parts):
    # join the base seed and the naming parts into one stable key
    key = "::".join(str(p) for p in (base_seed, *parts))
    # hash the key to a 32-bit seed so each named quantity gets its own generator
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


# Noise-free reward: a Gaussian bump in log10(ridge)-log10(beta) space.
def clean_reward(log10_ridge, log10_beta):
    # squared distance from the peak in each axis, scaled by the bump width
    d_ridge = (log10_ridge - PEAK_LOG10_RIDGE) / BUMP_WIDTH_DECADES
    d_beta = (log10_beta - PEAK_LOG10_BETA) / BUMP_WIDTH_DECADES
    # Gaussian bump peaking at PEAK_HEIGHT when both distances are zero
    return PEAK_HEIGHT * np.exp(-0.5 * (d_ridge ** 2 + d_beta ** 2))


# Noisy reward: clean bump plus one reproducible additive noise draw for this exact point.
def noisy_reward(ridge, beta, noise_seed):
    # move both knobs to log10 units where the bump is defined
    log10_ridge = np.log10(ridge)
    log10_beta = np.log10(beta)
    # one generator keyed by the noise seed and the rounded point, so repeats match
    rng = substream(noise_seed, "reward", round(log10_ridge, 6), round(log10_beta, 6))
    # single Gaussian noise draw added to the clean bump
    return float(clean_reward(log10_ridge, log10_beta) + rng.normal(0.0, NOISE_SD))


# True if a point sits within half a decade of the peak in BOTH axes.
def near_peak(ridge, beta, half_decade=0.5):
    # log10 distance from the peak in each axis
    d_ridge = abs(np.log10(ridge) - PEAK_LOG10_RIDGE)
    d_beta = abs(np.log10(beta) - PEAK_LOG10_BETA)
    # both axes must be within the half-decade box
    return (d_ridge <= half_decade) and (d_beta <= half_decade)
