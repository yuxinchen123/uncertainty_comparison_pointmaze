#!/usr/bin/env python
"""Initial (UNTRAINED) RND bonus field over the PointMaze_Large-v3 maze — 3D surface + annotated heatmap.

Question this figure answers (run-3.2.2 follow-up): the count oracle's bonus min(1, 1/sqrt(n)) starts
UNIFORM (numerator 1 everywhere); is the untrained RND network's initial bonus field also even across
the maze, or does the random network already prefer some states before any training?

What is computed:
- The exact initial network of the run-3.2.2 RND benchmark's best seed (C2 = adam / mse readout /
  beta=100; best of its 100 fresh seeds is a_seed=111, final reward 109.87). The construction mirrors
  train.py run() step by step (same RNG seeding order, same env stack, same registry factory), so the
  torch-initialized predictor/target weights are byte-identical to that run's step-0 networks.
- The observation-normalization warmup (200 samples of the observation space, methods._warmup_obs_rms)
  is re-seeded deterministically from a_seed: the ORIGINAL run's warmup draws are unrecoverable (the
  gymnasium Space RNG is seeded from OS entropy, not from a_seed). The script MEASURES how much this
  matters by redoing the warmup with a different space seed and reporting the field difference.
- The bonus is the mse readout B(x) = 0.5*||e(x)||^2 exactly as train.py computes it (obs-normalized,
  clamp +-5). beta only scales the reward handed to SAC and is irrelevant to the field's shape; the
  plotted values are normalized so the largest value is 1.

Grids (obs = [x, y, vx, vy]; world extent X in [-6, 6], Y in [-4.5, 4.5]; row 0 of the maze map is the
TOP row, x right / y up — the Figure 1 world convention of the writeup):
- Heatmap: the user's 10 x 10 grid over the full extent, evaluated at each sub-grid midpoint.
- Surface: a dense 100 x 100 evaluation (the network is continuous, so the surface is evaluated
  densely rather than interpolated from the 10 x 10 values).
- Velocity: the env clips velocity to +-5 (gymnasium_robotics point.py: np.clip(qvel, -5, 5)), so the
  three velocity levels are vx = vy = 0, 2.5, 5.0 (zero, half max, max).
- Normalization: heatmaps share ONE max (the largest of the 300 midpoint values across all three
  velocity levels -> exactly one annotated cell reads 1.00); surfaces share the dense global max.
  Shared normalization keeps the velocity levels comparable (per-panel normalization would hide the
  velocity effect this figure set exists to show).

Outputs (this folder): surface_v{0,2p5,5}.{pdf,png}, heatmap_v{0,2p5,5}.{pdf,png}, field_values.json.
Walls are drawn as Rectangle/Poly3DCollection patches, never imshow (vector-PDF wall-rendering rule).
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
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
HERE = os.path.dirname(os.path.abspath(__file__))

import gymnasium as gym            # noqa: E402
import gymnasium_robotics          # noqa: E402
from train import Config, make_base_env, select_cells, build_env_stack   # noqa: E402
from rnd_exploration.methods import EnvContext, build_intrinsic_model, _warmup_obs_rms  # noqa: E402
from rnd_exploration.envs.point_maze_utils import get_maze_map           # noqa: E402
from gymnasium.wrappers.utils import RunningMeanStd                      # noqa: E402

# --- geometry / evaluation constants (env-verified: full_observation_list.py + point.py clip) ---
LEFT, RIGHT, BOTTOM, TOP = -6.0, 6.0, -4.5, 4.5
V_MAX = 5.0                       # env clips qvel to +-5 -> max velocity level
VEL_LEVELS = [0.0, V_MAX / 2, V_MAX]
VEL_TAGS = ["0", "2p5", "5"]
N_COARSE = 10                     # the user's 10x10 midpoint grid (heatmap + annotated numbers)
N_DENSE = 100                     # dense evaluation for the smooth 3D surface
BEST_SEED = 111                   # best C2 (adam/mse/beta=100) seed of run 3.2.2 (final reward 109.87)

# --- 3D surface style (matched to reference_files/distribution.png via the style study: the original
# is MATLAB surf + shading interp + jet, low camera, white panes with dotted dark grid, wide-flat box;
# azimuth chosen so the maze reads like Figure 1: x to the right, y receding upward, start lower-left,
# goal upper-right) ---
STYLE = {
    "cmap": "jet",
    "elev": 20, "azim": -60,
    "box_aspect": (1.33, 1.0, 0.31),   # x span 12 : y span 9 : flat z, like the reference
    "figsize": (9.4, 5.2),
}


def build_untrained_rnd(a_seed: int):
    """Rebuild the exact step-0 RND of a run: mirror train.py run() up to build_intrinsic_model.

    Returns (model, maze_map, goal_cell, start_cell, ctx). Weights reproduce the run byte-identically
    (same seeding order + construction order); the obs-RMS warmup is re-seeded from a_seed (see module
    docstring)."""
    # the C2 benchmark config: rnd_next_state, adam/mse (Config defaults), run-3.2.2 env fields
    cfg = Config(algorithm="rnd_next_state", beta=100.0, a_seed=a_seed, device="cpu",
                 goal_position="top_right", env_max_episode=400, use_wandb=False)
    # run() lines 466-471: seed every RNG together, in this exact order
    random.seed(a_seed)
    np.random.seed(a_seed)
    torch.manual_seed(a_seed)
    gym.register_envs(gymnasium_robotics)
    # run() lines 475-490: base env, fixed cells, train stack
    base_env = make_base_env(cfg)
    goal_cell, start_cell = select_cells(cfg, base_env, a_seed)
    train_flat, train_pos, train_posvel = build_env_stack(cfg, base_env, start_cell, goal_cell)
    # deterministic warmup: gymnasium Spaces are otherwise seeded from OS entropy at first sample
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
    return model, maze_map, goal_cell, start_cell, ctx


def bonus_at(model, xs, ys, v):
    """Evaluate the RND bonus at every (x, y) pair with vx = vy = v.

    Before: xs, ys flat arrays of equal length L (e.g. the 100 midpoints of the 10x10 grid), v=2.5.
    After:  float array (L,) of raw B_mse values, e.g. [31.2, 28.9, ...]."""
    obs = np.stack([xs, ys, np.full_like(xs, v), np.full_like(xs, v)], axis=1).astype(np.float32)
    with torch.no_grad():
        out = model.compute({"next_observations": obs})
    return np.asarray(out, dtype=float)


def grid_eval(model, n, v):
    """Bonus over an n x n midpoint grid of the maze extent at velocity level v.

    Before: n=10, v=0. After: (x_centers (n,), y_centers (n,), V (n_y, n_x)) with y ascending."""
    x_centers = LEFT + (np.arange(n) + 0.5) * (RIGHT - LEFT) / n
    y_centers = BOTTOM + (np.arange(n) + 0.5) * (TOP - BOTTOM) / n
    XX, YY = np.meshgrid(x_centers, y_centers)          # shape (n_y, n_x)
    vals = bonus_at(model, XX.ravel(), YY.ravel(), v).reshape(n, n)
    return x_centers, y_centers, vals


def wall_rects(maze_map):
    """World-coordinate (x_left, y_bottom, w, h) of every wall cell (row 0 = TOP, Figure 1 convention).
    Before: 9x12 int map. After: list like [(-6.0, 3.5, 1, 1), ...] for each cell == 1."""
    rows, cols = maze_map.shape
    cw, ch = (RIGHT - LEFT) / cols, (TOP - BOTTOM) / rows
    return [(LEFT + c * cw, TOP - (r + 1) * ch, cw, ch)
            for r in range(rows) for c in range(cols) if int(maze_map[r, c]) == 1]


def cell_center(row, col, maze_map):
    """World (x, y) center of maze cell (row, col); row 0 is the TOP row."""
    rows, cols = maze_map.shape
    cw, ch = (RIGHT - LEFT) / cols, (TOP - BOTTOM) / rows
    return LEFT + (col + 0.5) * cw, TOP - (row + 0.5) * ch


def draw_surface(xd, yd, Vn, maze_map, goal_cell, start_cell, v, raw_max, path):
    """One 3D-surface image (reference_files/distribution.png style): dense normalized field as a jet
    surface, maze walls as dark tiles on the floor plane, start/goal markers, dotted panes."""
    XX, YY = np.meshgrid(xd, yd)
    fig = plt.figure(figsize=STYLE["figsize"])
    ax = fig.add_subplot(111, projection="3d")
    # the surface itself: dense, interpolated-shading look (high rcount/ccount, no mesh lines)
    ax.plot_surface(XX, YY, Vn, cmap=STYLE["cmap"], rcount=N_DENSE, ccount=N_DENSE,
                    linewidth=0, antialiased=True, alpha=0.92, vmin=0.0, vmax=1.0)
    # maze geometry: wall cells as dark tiles just above the floor plane (vector patches, not imshow)
    tiles = [[(x, y, 0.001), (x + w, y, 0.001), (x + w, y + h, 0.001), (x, y + h, 0.001)]
             for (x, y, w, h) in wall_rects(maze_map)]
    ax.add_collection3d(Poly3DCollection(tiles, facecolor="#4a4a4a", edgecolor="none", alpha=0.9))
    # start / goal floor markers so the orientation is checkable against Figure 1
    for cell, txt, color in [(start_cell, "S", "black"), (goal_cell, "G", "darkred")]:
        cx, cy = cell_center(cell[0], cell[1], maze_map)
        ax.text(cx, cy, 0.02, txt, color=color, fontsize=13, fontweight="bold", ha="center")
    # axes: world orientation (x right, y up in the floor plane), z = normalized bonus with 0/0.5/1
    ax.set_xlim(LEFT, RIGHT); ax.set_ylim(BOTTOM, TOP); ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1]); ax.set_xlabel("x"); ax.set_ylabel("y")
    # z-axis name as a 2D annotation: mplot3d's zlabel is clipped by the tight bbox at this camera,
    # while a text2D at axes coordinates is always included in the saved bounding box
    ax.text2D(1.11, 0.55, "normalized RND bonus", transform=ax.transAxes, rotation=90,
              ha="left", va="center", fontsize=11)
    ax.set_box_aspect(STYLE["box_aspect"])
    ax.view_init(elev=STYLE["elev"], azim=STYLE["azim"])
    # dotted pane grid like the reference
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"linestyle": ":", "color": (0.45, 0.45, 0.45, 1), "linewidth": 0.7})
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title(f"Untrained RND bonus, velocity $v_x=v_y={v:g}$   (raw max {raw_max:.1f})")
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def draw_heatmap(xc, yc, Vn, maze_map, goal_cell, start_cell, v, raw_max, path):
    """One classic annotated heatmap: the 10x10 normalized midpoint values (pcolormesh + a number per
    cell) with the true maze walls overlaid as gray patches; x right, y up (Figure 1 convention)."""
    n = len(xc)
    x_edges = np.linspace(LEFT, RIGHT, n + 1)
    y_edges = np.linspace(BOTTOM, TOP, n + 1)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    pm = ax.pcolormesh(x_edges, y_edges, Vn, cmap=STYLE["cmap"], vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidth=0.6)
    # one number per sub-grid cell (the user's 10x10 midpoint values, normalized: global max = 1);
    # text black on bright jet colors (cyan/yellow), white on dark ones (deep blue/red)
    cmap_obj = plt.get_cmap(STYLE["cmap"])
    for iy in range(n):
        for ix in range(n):
            r, g, b, _ = cmap_obj(Vn[iy, ix])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(xc[ix], yc[iy], f"{Vn[iy, ix]:.2f}", ha="center", va="center", fontsize=7.5,
                    color=("black" if lum > 0.55 else "white"))
    # true maze geometry on top: translucent gray wall patches + outline (vector, not imshow)
    for (x, y, w, h) in wall_rects(maze_map):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#3c3c3c", edgecolor="none", alpha=0.55))
    # S/G offset upward off the cell's printed number, with a dark stroke for contrast on any color
    import matplotlib.patheffects as pe
    for cell, txt, color in [(start_cell, "S", "white"), (goal_cell, "G", "yellow")]:
        cx, cy = cell_center(cell[0], cell[1], maze_map)
        ax.text(cx, cy + 0.28, txt, color=color, fontsize=14, fontweight="bold", ha="center",
                va="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
    ax.set_xlim(LEFT, RIGHT); ax.set_ylim(BOTTOM, TOP)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(pm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("normalized RND bonus (global max = 1)")
    ax.set_title(f"Untrained RND bonus, $10\\times10$ midpoints, $v_x=v_y={v:g}$   (raw max {raw_max:.1f})")
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_surface_log(xd, yd, Vself, maze_map, goal_cell, start_cell, raw_max, path):
    """v=0 detail view, both requested transforms at once: the field SELF-normalized (its own max = 1)
    on a log10 z axis — exposes the bowl's relief that the shared linear scale compresses.

    Before: Vself dense 100x100 self-normalized values in (0, 1], e.g. min ~0.004, max 1.0.
    After:  a surface of z = log10(Vself), z ticks at the decades (1, 0.1, 0.01, ...)."""
    Z = np.log10(np.clip(Vself, 1e-3, None))          # (0,1] -> [-3, 0]
    zmin = float(np.floor(Z.min()))
    XX, YY = np.meshgrid(xd, yd)
    fig = plt.figure(figsize=STYLE["figsize"])
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Z, cmap=STYLE["cmap"], rcount=N_DENSE, ccount=N_DENSE,
                    linewidth=0, antialiased=True, alpha=0.92, vmin=zmin, vmax=0.0)
    # walls on the log floor
    tiles = [[(x, y, zmin + 0.01), (x + w, y, zmin + 0.01), (x + w, y + h, zmin + 0.01), (x, y + h, zmin + 0.01)]
             for (x, y, w, h) in wall_rects(maze_map)]
    ax.add_collection3d(Poly3DCollection(tiles, facecolor="#4a4a4a", edgecolor="none", alpha=0.9))
    for cell, txt, color in [(start_cell, "S", "black"), (goal_cell, "G", "darkred")]:
        cx, cy = cell_center(cell[0], cell[1], maze_map)
        ax.text(cx, cy, zmin + 0.05, txt, color=color, fontsize=13, fontweight="bold", ha="center")
    ax.set_xlim(LEFT, RIGHT); ax.set_ylim(BOTTOM, TOP); ax.set_zlim(zmin, 0)
    ticks = list(range(int(zmin), 1))
    ax.set_zticks(ticks)
    ax.set_zticklabels(["$1$" if t == 0 else f"$10^{{{t}}}$" for t in ticks])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.text2D(1.15, 0.55, "self-normalized RND bonus (log scale)", transform=ax.transAxes,
              rotation=90, ha="left", va="center", fontsize=10)
    ax.set_box_aspect(STYLE["box_aspect"])
    ax.view_init(elev=STYLE["elev"], azim=STYLE["azim"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"linestyle": ":", "color": (0.45, 0.45, 0.45, 1), "linewidth": 0.7})
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_title(f"Untrained RND bonus, $v_x=v_y=0$ — self-normalized, log scale   (raw max {raw_max:.1f})")
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def draw_heatmap_log(xc, yc, Vself, maze_map, goal_cell, start_cell, raw_max, path):
    """v=0 detail heatmap, both transforms at once: SELF-normalized values on a log color scale,
    annotated per cell (2 significant digits below 0.1, else 2 decimals)."""
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe
    n = len(xc)
    vmin = max(float(Vself.min()), 1e-3)
    norm = LogNorm(vmin=vmin, vmax=1.0)
    x_edges = np.linspace(LEFT, RIGHT, n + 1)
    y_edges = np.linspace(BOTTOM, TOP, n + 1)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    pm = ax.pcolormesh(x_edges, y_edges, Vself, cmap=STYLE["cmap"], norm=norm,
                       edgecolors="white", linewidth=0.6)
    cmap_obj = plt.get_cmap(STYLE["cmap"])
    for iy in range(n):
        for ix in range(n):
            v = Vself[iy, ix]
            r, g, b, _ = cmap_obj(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            label = f"{v:.2f}" if v >= 0.095 else f"{v:.3f}"
            ax.text(xc[ix], yc[iy], label, ha="center", va="center", fontsize=7.0,
                    color=("black" if lum > 0.55 else "white"))
    for (x, y, w, h) in wall_rects(maze_map):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#3c3c3c", edgecolor="none", alpha=0.55))
    for cell, txt, color in [(start_cell, "S", "white"), (goal_cell, "G", "yellow")]:
        cx, cy = cell_center(cell[0], cell[1], maze_map)
        ax.text(cx, cy + 0.28, txt, color=color, fontsize=14, fontweight="bold", ha="center",
                va="center", path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
    ax.set_xlim(LEFT, RIGHT); ax.set_ylim(BOTTOM, TOP)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(pm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("self-normalized RND bonus (log color scale)")
    ax.set_title(f"Untrained RND bonus, $10\\times10$ midpoints, $v_x=v_y=0$ — self-normalized, log scale"
                 f"   (raw max {raw_max:.1f})")
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the untrained network, evaluate both grids at the three velocity levels, render 6 images."""
    model, maze_map, goal_cell, start_cell, ctx = build_untrained_rnd(BEST_SEED)
    probe = bonus_at(model, np.array([0.0]), np.array([0.0]), 0.0)[0]
    print(f"[field] probe bonus at obs [0,0,0,0]: {probe:.6f}  (cross-script model checksum)")
    print(f"[field] goal_cell={goal_cell} start_cell={start_cell} (expect (1,10) / (7,1))")

    # evaluate: coarse 10x10 (heatmap numbers) + dense 100x100 (surface), three velocity levels
    coarse, dense = {}, {}
    for v in VEL_LEVELS:
        coarse[v] = grid_eval(model, N_COARSE, v)
        dense[v] = grid_eval(model, N_DENSE, v)
    coarse_max = max(coarse[v][2].max() for v in VEL_LEVELS)   # shared normalizer: largest value -> 1
    dense_max = max(dense[v][2].max() for v in VEL_LEVELS)
    print(f"[field] raw maxima: coarse(10x10)={coarse_max:.2f}  dense(100x100)={dense_max:.2f}")

    # warmup sensitivity: redo ONLY the obs-RMS warmup with a different space seed, same weights,
    # and measure how much the normalized coarse field moves (reported, not plotted)
    base_v0 = coarse[0.0][2] / coarse_max
    ctx.observation_space.seed(BEST_SEED + 1)
    ctx.action_space.seed(BEST_SEED + 1)
    model.obs_rms = RunningMeanStd(shape=model._rnd_obs_shape)
    _warmup_obs_rms(model, ctx)
    alt = grid_eval(model, N_COARSE, 0.0)[2]
    alt_v0 = alt / alt.max()
    corr = float(np.corrcoef(base_v0.ravel(), alt_v0.ravel())[0, 1])
    print(f"[field] warmup-seed sensitivity (v=0, normalized field): max abs diff "
          f"{np.abs(base_v0 - alt_v0).max():.4f}, Pearson r = {corr:.3f}")
    # restore the canonical warmup for the plotted values
    ctx.observation_space.seed(BEST_SEED)
    ctx.action_space.seed(BEST_SEED)
    model.obs_rms = RunningMeanStd(shape=model._rnd_obs_shape)
    _warmup_obs_rms(model, ctx)

    # render the six images + dump the numbers
    values_dump = {"seed": BEST_SEED, "config": "adam/mse/beta=100 (C2 benchmark), untrained",
                   "coarse_global_max": float(coarse_max), "dense_global_max": float(dense_max),
                   "velocity_levels": VEL_LEVELS, "per_level": {}}
    for v, tag in zip(VEL_LEVELS, VEL_TAGS):
        xc, yc, Vc = coarse[v]
        xd, yd, Vd = dense[v]
        draw_surface(xd, yd, Vd / dense_max, maze_map, goal_cell, start_cell, v,
                     float(Vd.max()), os.path.join(HERE, f"surface_v{tag}"))
        draw_heatmap(xc, yc, Vc / coarse_max, maze_map, goal_cell, start_cell, v,
                     float(Vc.max()), os.path.join(HERE, f"heatmap_v{tag}"))
        values_dump["per_level"][str(v)] = {
            "raw_max_coarse": float(Vc.max()), "raw_min_coarse": float(Vc.min()),
            "raw_mean_coarse": float(Vc.mean()),
            "normalized_coarse_10x10_y_ascending": (Vc / coarse_max).round(4).tolist(),
        }
        print(f"[field] v={v:g}: coarse raw min/mean/max = {Vc.min():.2f}/{Vc.mean():.2f}/{Vc.max():.2f}")
    # the v=0 detail pair: self-normalized + log scale (both transforms at once), surface + heatmap
    xc0, yc0, Vc0 = coarse[0.0]
    xd0, yd0, Vd0 = dense[0.0]
    draw_surface_log(xd0, yd0, Vd0 / Vd0.max(), maze_map, goal_cell, start_cell,
                     float(Vd0.max()), os.path.join(HERE, "surface_v0_selfnorm_log"))
    draw_heatmap_log(xc0, yc0, Vc0 / Vc0.max(), maze_map, goal_cell, start_cell,
                     float(Vc0.max()), os.path.join(HERE, "heatmap_v0_selfnorm_log"))
    values_dump["v0_selfnorm_10x10_y_ascending"] = (Vc0 / Vc0.max()).round(4).tolist()
    with open(os.path.join(HERE, "field_values.json"), "w") as fh:
        json.dump(values_dump, fh, indent=1)
    print(f"[field] wrote 8 images (pdf+png) + field_values.json under {HERE}")


if __name__ == "__main__":
    main()
