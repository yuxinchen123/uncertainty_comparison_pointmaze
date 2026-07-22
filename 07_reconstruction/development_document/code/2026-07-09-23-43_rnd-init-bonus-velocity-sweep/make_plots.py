#!/usr/bin/env python
"""Initial (UNTRAINED) RND bonus vs velocity, at three maze positions — line plots.

Question this figure answers (run-3.2.2 follow-up): does the untrained RND network react differently
to the x velocity than to the y velocity, and how strongly does velocity move the initial bonus at
all? The RND input is the full observation [x, y, vx, vy], so half the input dimensions are velocity;
if the random network's bonus varies a lot along velocity, the position field (the heatmap figure) is
only part of the initial-bonus story.

Setup (identical untrained network to the surface/heatmap figure — same seed, same construction):
- Network: the step-0 RND of the run-3.2.2 benchmark's best seed (C2 = adam / mse / beta=100,
  a_seed=111), rebuilt by mirroring train.py run() (see the sibling folder's script for the details;
  both scripts print the same probe checksum to prove they evaluate the same network).
- Velocity range: the env clips velocity to +-5 (gymnasium_robotics point.py), so each sweep runs
  from -5 to +5 in 100 points. One sweep varies vx with vy = 0, the other varies vy with vx = 0.
- Three positions (world coordinates; maze cells are 1x1, row 0 = top):
    start        = (-4.5, -3.0)  -- the start cell (7, 1) center ("left bottom corner");
    left middle  = (-4.5,  0.0)  -- the open left-edge middle cell (4, 1) center; NOTE the cell
                                    directly above the start, (6, 1), is a WALL, so the left-edge
                                    middle open cell is the one meant by "the grid above the start";
    middle point = ( 0.0,  0.0)  -- the exact middle of the maze extent (inside open cell (4, 6)).
- Values are the RAW mse-readout bonus B = 0.5*||e||^2 (per the user: no normalization here).

Outputs (this folder): velocity_sweep_all.{pdf,png} (all 3 positions x 2 axes in one plot, styled
after reference_files/list_price_distribution_musical_instruments.pdf: smooth lines + soft fill),
velocity_sweep_{start,leftmid,middle}.{pdf,png} (per-position two-curve plots in the same style),
velocity_values.json.
Run:  conda run -n exploration python make_plots.py
"""
import json
import os
import random
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
HERE = os.path.dirname(os.path.abspath(__file__))

import gymnasium as gym            # noqa: E402
import gymnasium_robotics          # noqa: E402
from train import Config, make_base_env, select_cells, build_env_stack   # noqa: E402
from rnd_exploration.methods import EnvContext, build_intrinsic_model    # noqa: E402
from rnd_exploration.envs.point_maze_utils import get_maze_map           # noqa: E402

V_MAX = 5.0                # env velocity clip (+-5)
N_POINTS = 100             # sweep resolution
BEST_SEED = 111            # best C2 (adam/mse/beta=100) seed of run 3.2.2

POSITIONS = [              # (label, x, y) — see module docstring for the cell reasoning
    ("start (-4.5, -3.0)", -4.5, -3.0),
    ("left-edge middle (-4.5, 0.0)", -4.5, 0.0),
    ("middle point (0.0, 0.0)", 0.0, 0.0),
]
POS_TAGS = ["start", "leftmid", "middle"]
POS_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]   # blue / red / green (reference uses blue+red)


def build_untrained_rnd(a_seed: int):
    """Rebuild the exact step-0 RND of a run (same recipe as the sibling surface/heatmap script)."""
    cfg = Config(algorithm="rnd_next_state", beta=100.0, a_seed=a_seed, device="cpu",
                 goal_position="top_right", env_max_episode=400, use_wandb=False)
    # train.py run() seeding order, then env stack, then the registry factory
    random.seed(a_seed)
    np.random.seed(a_seed)
    torch.manual_seed(a_seed)
    gym.register_envs(gymnasium_robotics)
    base_env = make_base_env(cfg)
    goal_cell, start_cell = select_cells(cfg, base_env, a_seed)
    train_flat, train_pos, train_posvel = build_env_stack(cfg, base_env, start_cell, goal_cell)
    train_flat.observation_space.seed(a_seed)
    train_flat.action_space.seed(a_seed)
    obs_dim = int(np.prod(train_flat.observation_space.shape))
    act_dim = int(np.prod(train_flat.action_space.shape))
    ctx = EnvContext(obs_shape=(obs_dim,), action_dim=act_dim,
                     observation_space=train_flat.observation_space,
                     action_space=train_flat.action_space,
                     position_wrapper=train_pos, position_velocity_wrapper=train_posvel)
    model = build_intrinsic_model(cfg.algorithm, cfg, ctx)
    maze_map = np.array(get_maze_map(train_flat))
    return model, maze_map


def sweep(model, x, y, axis: str):
    """Raw bonus along one velocity axis at a fixed position, the other velocity component held at 0.

    Before: x=-4.5, y=-3.0, axis='vx'. After: (vs (100,), bonus (100,)) with vs from -5 to 5."""
    vs = np.linspace(-V_MAX, V_MAX, N_POINTS)
    vx = vs if axis == "vx" else np.zeros_like(vs)
    vy = vs if axis == "vy" else np.zeros_like(vs)
    obs = np.stack([np.full_like(vs, x), np.full_like(vs, y), vx, vy], axis=1).astype(np.float32)
    with torch.no_grad():
        out = model.compute({"next_observations": obs})
    return vs, np.asarray(out, dtype=float)


def style_axes(ax):
    """Reference style (list_price_distribution): open top/right, light look, roomy labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)


def main() -> None:
    """Build the network, run the 6 sweeps, render 1 combined + 3 per-position figures."""
    model, _ = build_untrained_rnd(BEST_SEED)
    with torch.no_grad():
        probe = float(model.compute({"next_observations": np.zeros((1, 4), dtype=np.float32)})[0])
    print(f"[vel] probe bonus at obs [0,0,0,0]: {probe:.6f}  (cross-script model checksum)")

    # the 6 sweeps: 3 positions x (vx sweep | vy sweep), raw bonus values
    curves = {}
    for (label, x, y), tag in zip(POSITIONS, POS_TAGS):
        vs, bx = sweep(model, x, y, "vx")
        _, by = sweep(model, x, y, "vy")
        curves[tag] = {"label": label, "vs": vs, "vx_sweep": bx, "vy_sweep": by}
        print(f"[vel] {label}: vx-sweep range {bx.min():.2f}..{bx.max():.2f}   "
              f"vy-sweep range {by.min():.2f}..{by.max():.2f}")

    # combined figure: color = position, solid = vx sweep, dashed = vy sweep, soft fill like the ref
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    # lines only in the combined plot: six stacked translucent fills read as mud (verifier finding);
    # the soft-fill reference look is kept in the per-position two-curve plots below
    for (tag, c) in zip(POS_TAGS, POS_COLORS):
        d = curves[tag]
        ax.plot(d["vs"], d["vx_sweep"], color=c, lw=2.6, label=f"{d['label']} — sweep $v_x$")
        ax.plot(d["vs"], d["vy_sweep"], color=c, lw=2.6, ls="--", label=f"{d['label']} — sweep $v_y$")
    ax.set_xlabel("velocity (the swept component; the other is 0)", fontsize=14)
    ax.set_ylabel("raw RND bonus  $\\frac{1}{2}\\Vert e\\Vert^2$", fontsize=14)
    ax.set_title("Untrained RND bonus vs velocity (seed 111, C2 benchmark network)", fontsize=15)
    ax.legend(fontsize=10.5, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "velocity_sweep_all.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "velocity_sweep_all.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-position two-curve figures in the exact reference look (blue vx, red vy, filled)
    for tag in POS_TAGS:
        d = curves[tag]
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        ax.plot(d["vs"], d["vx_sweep"], color="#1f77b4", lw=2.8, label="sweep $v_x$ ($v_y=0$)")
        ax.fill_between(d["vs"], d["vx_sweep"], alpha=0.18, color="#1f77b4")
        ax.plot(d["vs"], d["vy_sweep"], color="#d62728", lw=2.8, label="sweep $v_y$ ($v_x=0$)")
        ax.fill_between(d["vs"], d["vy_sweep"], alpha=0.18, color="#d62728")
        ax.set_xlabel("velocity", fontsize=14)
        ax.set_ylabel("raw RND bonus", fontsize=14)
        ax.set_title(f"Untrained RND bonus vs velocity — {d['label']}", fontsize=14)
        # upper center is the empty zone of these U-shaped curves (verifier: bottom-right occluded)
        ax.legend(fontsize=12, frameon=False, loc="upper center")
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"velocity_sweep_{tag}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(HERE, f"velocity_sweep_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # dump the numbers
    dump = {"seed": BEST_SEED, "config": "adam/mse/beta=100 (C2 benchmark), untrained",
            "v_range": [-V_MAX, V_MAX], "n_points": N_POINTS,
            "positions": {tag: {"label": curves[tag]["label"],
                                "vx_sweep": curves[tag]["vx_sweep"].round(4).tolist(),
                                "vy_sweep": curves[tag]["vy_sweep"].round(4).tolist()}
                          for tag in POS_TAGS},
            "vs": curves["start"]["vs"].round(4).tolist()}
    with open(os.path.join(HERE, "velocity_values.json"), "w") as fh:
        json.dump(dump, fh, indent=1)
    print(f"[vel] wrote 4 figures (pdf+png) + velocity_values.json under {HERE}")


if __name__ == "__main__":
    main()
