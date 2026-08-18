#!/usr/bin/env python
"""Off-support bonus fields: train each champion for 256 uniform full-batch steps on the 108
cell midpoints, then evaluate the bonus on a fine quarter-cell grid. Shows what an exploring
agent would read BETWEEN the trained positions — the quantity the campaign metric does not
score. Methods are imported from the experiment folders' submit-time snapshots, so the fields
are exactly the validated methods' fields.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python offsupport_fields.py
"""
import importlib.util
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
from decay_harness.points import point_set  # noqa: E402

CAMPAIGN = os.path.join(PROJ, "experiments",
                        "2026-08-17-22-05_autoresearch_uniform-fullbatch_"
                        "cellmid108-center100_4096step_seed0-9")
N_STEPS = 256
PANELS = [  # (experiment folder holding the method snapshot, panel title)
    ("exp_010_linhead_rls_residual_count", "residual-encoded shrink (global ReLU features)"),
    ("exp_019_coinflip_adaptive_uniform", "coin flips (adaptive dictionary)"),
    ("val_112_elliptical_sigma035_uniform30", "elliptical readout ($\\sigma \\leq 0.35$)"),
]


def load_method(exp: str):
    """Import the Method class from an experiment folder's submit-time snapshot."""
    path = os.path.join(CAMPAIGN, exp, "method.py")
    spec = importlib.util.spec_from_file_location(f"method_{exp}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Method


def fine_grid() -> np.ndarray:
    """Quarter-cell evaluation grid over the maze extent: 48 x 36 = 1728 points."""
    xs = np.linspace(-5.875, 5.875, 48)
    ys = np.linspace(-4.375, 4.375, 36)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = np.zeros(gx.size)
    return np.stack([gx.ravel(), gy.ravel(), z, z], axis=1).astype(np.float32), xs, ys


def main():
    """Train each champion briefly on the cell midpoints and render its fine-grid field."""
    train = torch.as_tensor(point_set("cell_midpoints"))
    grid, xs, ys = fine_grid()
    xg = torch.as_tensor(grid)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.6 * len(PANELS), 4.4))
    for ax, (exp, title) in zip(np.atleast_1d(axes), PANELS):
        Method = load_method(exp)
        m = Method(0)
        for _ in range(N_STEPS):
            m.update(train)
        field = m.bonus(xg).reshape(len(xs), len(ys))
        # per-panel auto scale: the fields sit near the visited level 0.06, so a fixed
        # [0, 1] scale would render every panel uniformly dark
        pc = ax.pcolormesh(xs, ys, field.T, shading="nearest", cmap="viridis",
                           vmin=0.0, vmax=float(field.max()))
        ax.plot(train[:, 0], train[:, 1], ".", color="white", ms=2)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nvisited-cell level $\\approx {(N_STEPS + 1) ** -0.5:.3f}$",
                     fontsize=12)
        plt.colorbar(pc, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCDIR, "figures", "offsupport_fields.pdf"))
    print("wrote offsupport_fields.pdf")
    # print the field statistics the caption cites
    for exp, title in PANELS:
        Method = load_method(exp)
        m = Method(0)
        for _ in range(N_STEPS):
            m.update(train)
        f = m.bonus(xg)
        on = m.bonus(train)
        print(f"{exp}: on-support mean {on.mean():.4f}, off-support mean {f.mean():.4f}, "
              f"off-support max {f.max():.4f}")


if __name__ == "__main__":
    main()
