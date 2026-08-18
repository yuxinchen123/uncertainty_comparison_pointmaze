#!/usr/bin/env python
"""Mechanism check: simulate the LINEARIZED residual dynamics of the real network under the
SGD-1/t schedule and compare predicted per-position slopes with exp 002's measured ones.

The linearized update is e <- e - eta_t (1/P) J J^T e with J the network Jacobian at
initialization. J J^T e is computed matrix-free: v = J^T e by reverse mode, then J v by a
second reverse pass over a dummy dual objective. Curves are recorded on the campaign's
checkpoint grid, fitted with the campaign's floored power fit, and scattered against the
measured slopes of exp 002 (same seed, same schedule).

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python linearization_check.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DOCDIR = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.dirname(DOCDIR)
sys.path.insert(0, os.path.join(PROJ, "code"))
from decay_harness.fitting import fit_power_floor          # noqa: E402
from decay_harness.points import checkpoint_steps, point_set  # noqa: E402
from method import make_mlp                                 # noqa: E402  (shared helpers)

CAMPAIGN = os.path.join(PROJ, "experiments",
                        "2026-08-17-22-05_autoresearch_uniform-fullbatch_"
                        "cellmid108-center100_4096step_seed0-9")
SEED = 0
ETA0, T0 = 1e-2, 1e2
N_STEPS = 4096


def jjt_apply(pred, x, e):
    """Return (1/P) J J^T e without forming J: one reverse pass for u = J^T e, one more for
    J u via the double-backward identity d/de' <J^T e', u> = J u."""
    params = [p for p in pred.parameters() if p.requires_grad]
    out = pred(x)
    # u = J^T e (reverse mode), graph kept so a second differentiation can run
    u = torch.autograd.grad(out, params, grad_outputs=e, create_graph=True)
    # J u: differentiate <u_detached, J^T w> with respect to w at w = 0 — equivalently, take
    # the gradient of sum(u * u_detached) with respect to the DUMMY w through out
    w = torch.zeros_like(out, requires_grad=True)
    uw = torch.autograd.grad(out, params, grad_outputs=w, create_graph=True)
    s = sum((a * b.detach()).sum() for a, b in zip(uw, u))
    (jv,) = torch.autograd.grad(s, w)
    return jv / x.shape[0]


def simulate():
    """Linearized SGD-1/t distillation on the cell midpoints; per-position norm curves."""
    torch.manual_seed(SEED)
    target = make_mlp(4, 256, 128, SEED, "target")
    pred = make_mlp(4, 256, 128, SEED, "predictor")
    for p in target.parameters():
        p.requires_grad_(False)
    x = torch.as_tensor(point_set("cell_midpoints"))
    with torch.no_grad():
        e = (pred(x) - target(x)).detach()
    cps = set(checkpoint_steps(N_STEPS))
    steps, curves = [0], [e.norm(dim=1).numpy().copy()]
    for t in range(1, N_STEPS + 1):
        eta = ETA0 / (1.0 + (t - 1) / T0)
        e = e - eta * jjt_apply(pred, x, e).detach()
        if t in cps:
            steps.append(t)
            curves.append(e.norm(dim=1).numpy().copy())
    return np.array(steps), np.array(curves)   # (T,), (T, P)


def measured_slopes():
    """exp 002's per-position fitted slopes on the cell midpoints, seed-mean curves."""
    recs = []
    for p in sorted(glob.glob(os.path.join(CAMPAIGN, "exp_002_sgd1t_baseline",
                                           "cell_midpoints_seed*.json"))):
        with open(p) as fh:
            recs.append(json.load(fh))
    steps = np.asarray(recs[0]["checkpoint_steps"])
    b = np.mean([np.asarray(r["bonus"], dtype=float) for r in recs], axis=0)
    return steps, np.array([fit_power_floor(steps, b[:, i])["slope"]
                            for i in range(b.shape[1])])


def main():
    """Fit the simulated curves, scatter predicted against measured slopes."""
    steps, curves = simulate()
    pred_slopes = np.array([fit_power_floor(steps, curves[:, i])["slope"]
                            for i in range(curves.shape[1])])
    _, meas = measured_slopes()
    r = np.corrcoef(pred_slopes, meas)[0, 1]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(pred_slopes, meas, "o", ms=4, alpha=0.7)
    lo = min(pred_slopes.min(), meas.min()) - 0.05
    hi = max(pred_slopes.max(), meas.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1.5)
    ax.set_xlabel("slope predicted by the linearized dynamics")
    ax.set_ylabel("measured slope (exp 002, seed-mean)")
    ax.set_title(f"cell midpoints, SGD-1/t; correlation {r:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCDIR, "figures", "linearization_check.pdf"))
    print(f"correlation predicted vs measured: {r:.4f}")
    print(f"predicted slope range [{pred_slopes.min():.3f}, {pred_slopes.max():.3f}]; "
          f"measured [{meas.min():.3f}, {meas.max():.3f}]")


if __name__ == "__main__":
    main()
