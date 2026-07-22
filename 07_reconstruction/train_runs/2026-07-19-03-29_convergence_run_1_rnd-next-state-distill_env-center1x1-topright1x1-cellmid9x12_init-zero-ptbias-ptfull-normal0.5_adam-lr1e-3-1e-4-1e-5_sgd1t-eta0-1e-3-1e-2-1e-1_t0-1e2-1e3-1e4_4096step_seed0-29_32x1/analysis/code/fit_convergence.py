#!/usr/bin/env python
"""Convergence run 1 fitting: power-with-floor fits of the normalized per-point l2 curves.

Model (main.tex sec:convergence-fit-method): y(n) = c + a*(n + n0)^(-alpha), fitted by nonlinear
least squares with residuals on log y (a floored fit, NOT a straight-line log regression), fixed
burn-in n >= 2, bounded parameters, multiple starts. Two granularities per (config, env column),
reported separately: 30 per-seed fits (slope mean +- SE over seeds) and one seed-average-curve
fit. Diverged runs are excluded from every fit and counted. Output: analysis/data/fits.json.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python fit_convergence.py
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
RUN = os.path.dirname(ANALYSIS)
SWEEP = "2026-07-19-03-32_convergence-run1"
LOCAL = os.path.join(RUN, "data", SWEEP, "local")

BURN_IN = 2          # fixed fit window: steps 2 .. 4096 (user decision; step 0 = baseline)
TARGET_SLOPE = -0.5  # the reference slope; ranking key = closeness to it

ENVS = ["center_square", "top_right_cell", "cell_midpoints"]
ENV_COLS = ENVS + ["aggregate"]

INIT_LABELS = {  # (rnd_weight_init, rnd_bias_init) -> fixed label, in I1..I4 order
    ("orthogonal", "zero"): "I1-zero",
    ("orthogonal", "pytorch_default"): "I2-ptbias",
    ("pytorch_default", "pytorch_default"): "I3-ptfull",
    ("orthogonal", "normal_0.5"): "I4-normal0.5",
}
INIT_ORDER = list(INIT_LABELS.values())


def opt_label(r: dict) -> str:
    """Fixed optimizer-config label from a record (matches build_queue.py's ordering scheme)."""
    if r["rnd_optimizer"] == "adam":
        return "adam-lr%g" % r["rnd_lr"]
    return "sgd1t-eta%g-t%g" % (r["rnd_sgd_eta0"], r["rnd_sgd_t0"])


OPT_ORDER = (["adam-lr1e-05", "adam-lr0.0001", "adam-lr0.001"]
             + ["sgd1t-eta%g-t%g" % (e, t) for e in (1e-3, 1e-2, 1e-1) for t in (1e2, 1e3, 1e4)])


def load_curves():
    """Load every record into per-seed normalized point matrices.

    Returns curves[(init, opt, env, seed)] = (steps (T,), y (T, P) normalized) for non-diverged
    runs, and div[(init, opt, env)] = diverged-seed count."""
    curves, div = {}, {}
    for p in glob.glob(os.path.join(LOCAL, "*.json")):
        r = json.load(open(p))
        init = INIT_LABELS[(r["rnd_weight_init"], r["rnd_bias_init"])]
        key3 = (init, opt_label(r), r["point_set"])
        if r["diverged"]:
            div[key3] = div.get(key3, 0) + 1
            continue
        # normalize point-wise by the step-0 baseline row: b (T, P) -> y = b / b[0]
        # before: b[0] = [7.4, 12.1, ...] (raw l2 at init); after: y[0] = [1.0, 1.0, ...]
        steps = np.array([c["step"] for c in r["checkpoints"]])
        b = np.array([c["b_l2"] for c in r["checkpoints"]])
        curves[key3 + (r["a_seed"],)] = (steps, b / b[0])
    return curves, div


def fit_power_floor(steps: np.ndarray, y: np.ndarray) -> dict:
    """Fit y(n) = c + a*(n+n0)^(-alpha) on the window n >= BURN_IN (log-y residuals, bounded,
    multi-start); returns the best fit's parameters and cost."""
    m = steps >= BURN_IN
    n, yy = steps[m].astype(float), y[m]
    logy = np.log(yy)

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
    c, a, n0, alpha = best.x
    return {"c": float(c), "a": float(a), "n0": float(n0), "alpha": float(alpha),
            "cost": float(best.cost)}


def local_exponent(steps: np.ndarray, y: np.ndarray, c: float):
    """alpha_eff(n) = -dlog(y-c)/dlog(n) between successive in-window checkpoints (masked where
    y <= c); returns (midpoint steps, alpha_eff) lists."""
    m = steps >= BURN_IN
    n, yy = steps[m].astype(float), y[m] - c
    mids, vals = [], []
    for i in range(len(n) - 1):
        if yy[i] > 0 and yy[i + 1] > 0 and n[i + 1] > n[i]:
            mids.append(float(np.sqrt(n[i] * n[i + 1])))
            vals.append(float(-(np.log(yy[i + 1]) - np.log(yy[i]))
                              / (np.log(n[i + 1]) - np.log(n[i]))))
    return mids, vals


def _fit_one(args):
    """Pool worker: fit one curve (steps, y) -> fit dict."""
    steps, y = args
    return fit_power_floor(steps, y)


def main() -> None:
    """Fit every (config, env column) at both granularities and write fits.json."""
    import multiprocessing as mp
    curves, div = load_curves()
    seeds = sorted({k[3] for k in curves})
    steps_ref = None
    results = []
    for init in INIT_ORDER:
        for opt in OPT_ORDER:
            for env_col in ENV_COLS:
                # collect the per-seed env-mean curves for this cell (aggregate = all 308 points
                # of the three envs concatenated; a seed enters only with all three non-diverged)
                per_seed = []
                used = []
                for s in seeds:
                    if env_col == "aggregate":
                        parts = [curves.get((init, opt, e, s)) for e in ENVS]
                        if any(p is None for p in parts):
                            continue
                        steps = parts[0][0]
                        y = np.concatenate([p[1] for p in parts], axis=1).mean(axis=1)
                    else:
                        part = curves.get((init, opt, env_col, s))
                        if part is None:
                            continue
                        steps, y = part[0], part[1].mean(axis=1)
                    per_seed.append(y)
                    used.append(s)
                    steps_ref = steps
                row = {"init": init, "opt": opt, "env": env_col, "n_seeds": len(used),
                       "seeds": used,
                       "diverged": (sum(div.get((init, opt, e), 0) for e in ENVS)
                                    if env_col == "aggregate" else div.get((init, opt, env_col), 0))}
                if per_seed:
                    row["_ys"] = np.stack(per_seed)              # (n_seeds, T); fitted below
                    row["_steps"] = steps_ref
                results.append(row)
    # fit every curve (per-seed curves + each cell's mean curve) in one process pool
    tasks, owners = [], []
    for row in results:
        if "_ys" not in row:
            continue
        for y in row["_ys"]:
            tasks.append((row["_steps"], y))
            owners.append((row, "seed"))
        tasks.append((row["_steps"], row["_ys"].mean(axis=0)))
        owners.append((row, "avg"))
    with mp.Pool(processes=8) as pool:
        fits = pool.map(_fit_one, tasks, chunksize=16)
    per_row_fits = {}
    for (row, kind), fit in zip(owners, fits):
        if kind == "seed":
            per_row_fits.setdefault(id(row), []).append(fit)
        else:
            row["avg_fit"] = fit
    for row in results:
        if "_ys" not in row:
            continue
        ys, steps = row.pop("_ys"), row.pop("_steps")
        seed_fits = per_row_fits[id(row)]
        # per-seed curves + floors are stored so the figures can draw the 30 faint sublines
        # (bonus panel) and their per-seed local exponents (alpha panel)
        row["per_seed_curves"] = ys.tolist()
        row["per_seed_cs"] = [f["c"] for f in seed_fits]
        slopes = -np.array([f["alpha"] for f in seed_fits])
        row["per_seed_slopes"] = slopes.tolist()
        row["slope_mean"] = float(slopes.mean())
        row["slope_se"] = (float(slopes.std(ddof=1) / np.sqrt(len(slopes)))
                           if len(slopes) > 1 else 0.0)
        row["avg_slope"] = -row["avg_fit"]["alpha"]
        mean_curve = ys.mean(axis=0)
        row["steps"] = steps.tolist()
        row["curve_mean"] = mean_curve.tolist()
        row["curve_se"] = ((ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])).tolist()
                           if ys.shape[0] > 1 else [0.0] * len(mean_curve))
        mids, vals = local_exponent(steps, mean_curve, row["avg_fit"]["c"])
        row["alpha_eff_steps"] = mids
        row["alpha_eff"] = vals
    out = {"sweep_id": SWEEP, "burn_in": BURN_IN, "target_slope": TARGET_SLOPE,
           "env_cols": ENV_COLS, "init_order": INIT_ORDER, "opt_order": OPT_ORDER,
           "results": results}
    os.makedirs(os.path.join(ANALYSIS, "data"), exist_ok=True)
    path = os.path.join(ANALYSIS, "data", "fits.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    n_fit = sum(1 for r in results if "slope_mean" in r)
    print(f"wrote {path}: {len(results)} cells, {n_fit} with fits "
          f"({len(results) - n_fit} empty/diverged-out)")


if __name__ == "__main__":
    main()
