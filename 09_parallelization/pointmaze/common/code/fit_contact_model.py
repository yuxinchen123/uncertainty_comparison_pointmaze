"""Verify the candidate closed-form MuJoCo contact law against the probe data.

Reads fixtures/contact_probe.npz (rows: gap, vy, ay, ncon, y_next, vy_next; the probed wall
is the top border, face y=3.5, normal -y, dist = 3.4 - y = gap). Implements the full law
(impedance from solimp, aref from solref, regularized force, implicit-damping Euler) and
prints the max prediction error over all rows, for the two candidate impedance arguments
(r = dist - margin vs dist).
"""
from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent.parent / "fixtures"

# model constants (probe-verified)
H, M, D, G = 0.01, 4.1887902047863905, 1.0, 100.0
MARGIN, TAU, ZETA = 0.002, 0.02, 1.0
D0, DMAX, WIDTH, MID, POWER = 0.9, 0.95, 0.001, 0.5, 2.0


def impedance(x_abs):
    """solimp sigmoid: fraction y(x) in [0,1] for x = |violation|/width, then d0..dmax.

    before: x_abs = 0.0005/0.001 = 0.5 (half the width)  after: d = 0.9 + 0.05*y(0.5)
    """
    x = np.clip(x_abs / WIDTH, 0.0, 1.0)
    y = np.where(x < MID, (x ** POWER) / (MID ** (POWER - 1)),
                 1.0 - ((1.0 - x) ** POWER) / ((1.0 - MID) ** (POWER - 1)))
    return D0 + (DMAX - D0) * y


def predict(gap, vy, ay, imp_arg):
    """One-step (y_next_delta, vy_next) prediction for the top-wall contact case.

    gap = dist (surface distance, negative = penetration); normal n = -y.
    imp_arg: 'r' -> impedance of |dist - margin|, 'dist' -> impedance of |dist|.
    """
    vy_c = np.clip(vy, -5.0, 5.0)                 # PointEnv clips velocity before stepping
    F = G * np.clip(ay, -1.0, 1.0)
    a_unc_y = (F - D * vy_c) / M                  # continuous-time unconstrained accel
    active = gap <= MARGIN
    r = gap - MARGIN
    x_abs = np.abs(r) if imp_arg == "r" else np.abs(gap)
    d = impedance(x_abs)
    b = 2.0 / (DMAX * TAU)
    k = d / (DMAX ** 2 * TAU ** 2 * ZETA ** 2)
    v_n = -vy_c                                   # separation velocity (normal -y)
    aref = -b * v_n - k * r
    A = 1.0 / M
    R = (1.0 - d) / d * A
    a_unc_n = -a_unc_y
    f = np.maximum(0.0, (aref - a_unc_n) / (A + R)) * active
    Fy = F - f                                    # contact force acts along -y
    vy2 = (M * vy_c + H * Fy) / (M + H * D)       # implicit-damping Euler
    y2_delta = H * vy2
    return y2_delta, vy2


def main():
    data = np.load(FIX / "contact_probe.npz")
    rows = data["rows"]
    gap, vy, ay, ncon, y_next, vy_next = rows.T
    y0 = 3.4 - gap
    for imp_arg in ("r", "dist"):
        dy_pred, vy_pred = predict(gap, vy, ay, imp_arg)
        y_err = np.abs((y0 + dy_pred) - y_next)
        v_err = np.abs(vy_pred - vy_next)
        print(f"impedance arg = {imp_arg:4s}: max|y|err {y_err.max():.3e}  "
              f"max|vy|err {v_err.max():.3e}  (worst row: gap={gap[v_err.argmax()]:+.4f} "
              f"vy={vy[v_err.argmax()]:+.2f} ay={ay[v_err.argmax()]:+.0f})")


if __name__ == "__main__":
    main()
