"""Floored power-law fitting for bonus decay curves.

Ported from the convergence run 1 analysis (fit_convergence.py) so the fitted model and its
conventions match the prior work exactly: y(n) = c + a*(n + n0)^(-alpha), nonlinear least
squares with residuals on log y (a floored fit, never a straight-line log regression), fixed
burn-in n >= BURN_IN, bounded parameters, multiple starts. FIXED harness module (program.md).
"""
import numpy as np
from scipy.optimize import least_squares

BURN_IN = 2  # fixed fit window start: steps 2 .. n_steps (step 0 is the normalization baseline)


def fit_power_floor(steps: np.ndarray, y: np.ndarray) -> dict:
    """Fit y(n) = c + a*(n+n0)^(-alpha) on the window n >= BURN_IN (log-y residuals, bounded,
    multi-start); returns the best fit's parameters and cost, with slope = -alpha."""
    # restrict to the in-window checkpoints and precompute the log targets once
    m = steps >= BURN_IN
    n, yy = steps[m].astype(float), np.asarray(y, dtype=float)[m]
    logy = np.log(np.maximum(yy, 1e-300))

    def resid(x):
        c, a, n0, alpha = x
        return np.log(c + a * (n + n0) ** (-alpha) + 1e-300) - logy

    # multi-start over the decay exponent and the floor (the two parameters that trap local
    # minima); a0 chosen so the model matches the first in-window value at each start
    y_end = float(yy[-1])
    best = None
    for alpha0 in (0.3, 0.7, 1.6):
        for c0 in (1e-8, max(y_end * 0.8, 1e-8)):
            a0 = max(float(yy[0]) - c0, 1e-6) * (n[0]) ** alpha0
            try:
                r = least_squares(resid, x0=[c0, a0, 0.0, alpha0],
                                  bounds=([0.0, 1e-10, 0.0, 1e-3], [1.0, 1e3, 1e4, 5.0]),
                                  max_nfev=400)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        raise RuntimeError("fit_power_floor: every start failed (non-finite curve?)")
    c, a, n0, alpha = best.x
    return {"c": float(c), "a": float(a), "n0": float(n0), "alpha": float(alpha),
            "slope": float(-alpha), "cost": float(best.cost)}


def local_exponent(steps: np.ndarray, y: np.ndarray, c: float):
    """alpha_eff(n) = -dlog(y-c)/dlog(n) between successive in-window checkpoints (masked where
    y <= c); returns (geometric-midpoint steps, alpha_eff) lists for the phase-structure plot."""
    m = steps >= BURN_IN
    n, yy = steps[m].astype(float), np.asarray(y, dtype=float)[m] - c
    mids, vals = [], []
    for i in range(len(n) - 1):
        if yy[i] > 0 and yy[i + 1] > 0 and n[i + 1] > n[i]:
            mids.append(float(np.sqrt(n[i] * n[i + 1])))
            vals.append(float(-(np.log(yy[i + 1]) - np.log(yy[i]))
                              / (np.log(n[i + 1]) - np.log(n[i]))))
    return mids, vals
