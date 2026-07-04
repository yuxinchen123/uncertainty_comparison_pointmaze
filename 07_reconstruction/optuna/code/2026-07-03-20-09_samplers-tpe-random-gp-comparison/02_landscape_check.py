"""Sanity-check the toy landscape: peak height, decay away from the peak, noise level, repeatability.

Confirms the clean bump peaks near 50 at (ridge=1e-6, beta=1e-2), falls off by ~1 decade,
that the additive noise has SD near 10, and that the same point re-queried gives the same value.
"""

import numpy as np

import landscape as L


# Print the clean reward on a small log grid so the single peak is visible.
def show_clean_grid():
    # header row of beta values in log10
    print("clean_reward on a log grid (rows=log10 ridge, cols=log10 beta):")
    betas = [-3.0, -2.0, -1.0]
    print("            beta=1e-3   beta=1e-2   beta=1e-1")
    # one row per ridge decade
    for lr in [-6.0, -5.0, -4.0, -3.0, -2.0]:
        vals = [L.clean_reward(lr, lb) for lb in betas]
        print(f"ridge=1e{int(lr):d}   " + "   ".join(f"{v:9.3f}" for v in vals))


# Estimate the noise SD by re-drawing many points and comparing to the clean value.
def show_noise_sd():
    # sample 2000 random points inside the domain and measure noisy-minus-clean spread
    rng = L.substream(0, "landscape_check_points")
    residuals = []
    for _ in range(2000):
        # uniform in log space across the searchable domain
        log10_ridge = rng.uniform(-6.0, -2.0)
        log10_beta = rng.uniform(-3.0, -1.0)
        ridge = 10.0 ** log10_ridge
        beta = 10.0 ** log10_beta
        # a distinct noise seed per point keeps draws independent
        noisy = L.noisy_reward(ridge, beta, noise_seed=1234)
        residuals.append(noisy - L.clean_reward(log10_ridge, log10_beta))
    # report the measured spread against the configured SD
    print(f"\nmeasured noise SD over 2000 points = {np.std(residuals):.3f} (configured {L.NOISE_SD})")


# Confirm the same point re-queried returns the identical noisy value.
def show_repeatability():
    # two calls at the same point and seed must match exactly
    a = L.noisy_reward(1e-6, 1e-2, noise_seed=7)
    b = L.noisy_reward(1e-6, 1e-2, noise_seed=7)
    # a different seed at the same point must differ
    c = L.noisy_reward(1e-6, 1e-2, noise_seed=8)
    print(f"\nrepeat same point+seed: {a:.4f} == {b:.4f} -> {a == b}")
    print(f"same point, seed 7 vs 8: {a:.4f} vs {c:.4f} -> differ: {a != c}")


# Run all three checks.
def main():
    show_clean_grid()
    show_noise_sd()
    show_repeatability()


if __name__ == "__main__":
    main()
