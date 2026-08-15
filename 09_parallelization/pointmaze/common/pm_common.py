"""Shared ground truth for every PointMaze GPU implementation (numpy only, no torch/jax).

Holds the verified physics constants, the four maze maps, the env configuration dataclass,
and the precomputed wall-clamp geometry that the torch / CUDA / JAX steppers all consume.
Physics provenance: physics_spec.md (probe-verified against Gymnasium-Robotics 1.3.1).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---- physics constants (probe-verified exact; see physics_spec.md) ----
H = 0.01                       # integrator timestep [s]; one env step = one integrator step
M = 4.1887902047863905         # ball mass [kg] = 4/3 * pi * 0.1^3 * 1000
D = 1.0                        # joint damping
G = 100.0                      # actuator gear (force = G * action)
R = 0.1                        # ball radius [m]
VEL_CLIP = 5.0                 # per-axis |velocity| clip applied BEFORE each integrator step
ACT_CLIP = 1.0                 # per-axis |action| clip
# derived integrator coefficients: v' = A_COEF * clip(v) + B_COEF * clip(a)  (free space)
A_COEF = M / (M + H * D)       # 0.99762...
B_COEF = H * G / (M + H * D)   # 0.23815...
INV_MHD = 1.0 / (M + H * D)    # implicit-damping Euler divisor

# ---- contact model constants (probe-verified exact to 8e-16; see fit_contact_model.py) ----
MARGIN = 0.002                 # contact active when surface distance <= margin
TAU = 0.02                     # solref time constant
ZETA = 1.0                     # solref damping ratio
SOL_D0, SOL_DMAX = 0.9, 0.95   # solimp impedance range
SOL_WIDTH, SOL_MID = 0.001, 0.5  # solimp width and midpoint (power 2 baked into the formula)
B_REF = 2.0 / (SOL_DMAX * TAU)                       # aref velocity gain, 105.263...
K_OVER_D = 1.0 / (SOL_DMAX ** 2 * TAU ** 2 * ZETA ** 2)  # aref stiffness per unit impedance

# ---- maze maps (1 = wall; the reference letters r/g/c are ordinary open cells here
# because our configs fix start and goal cells explicitly). Copied verbatim from
# gymnasium_robotics.envs.maze.maps with letters mapped to 0. ----
MAPS = {
    "umaze": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    "open": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "medium": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "large": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}


@dataclass(frozen=True)
class EnvConfig:
    """All env-side knobs. Defaults = the run-6 setup `initial_single_large_pointmaze_max_400`."""
    map_name: str = "large"
    start_cell: Tuple[int, int] = (7, 1)     # (row, col), row 0 at top
    goal_cell: Tuple[int, int] = (1, 10)
    position_noise: float = 0.25             # uniform(-x, +x) added per axis at every reset (start AND goal)
    max_episode_steps: int = 400
    continuing_task: bool = True             # True: never terminated (only truncated at the cap)
    reward_shift: float = 0.0                # added to every step's reward (-1 = ExPLORe convention)
    goal_radius: float = 0.45                # sparse reward and termination threshold


def cell_center(cell: Tuple[int, int], rows: int, cols: int) -> Tuple[float, float]:
    """World (x, y) of a cell center. Example: large map (7,1) -> (-4.5, -3.0)."""
    i, j = cell
    return (j + 0.5) - cols / 2.0, rows / 2.0 - (i + 0.5)


NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def build_geometry(map_name: str) -> dict:
    """Precompute the per-cell neighbor-wall-box tables the steppers gather from at runtime.

    Contacts are computed per neighboring wall BOX (exactly like MuJoCo generates one contact
    per geom pair): nearest point on the box rectangle -> distance and normal. The 8 neighbor
    cells of the ball's cell are the only boxes that can ever be within contact range (max
    per-step travel 0.052 m, penetration a few mm, cells 1 m).

    Input: map name. Output dict of numpy arrays (float64; steppers cast as needed):
      wall     [rows, cols] uint8  1 = wall cell
      nb_wall  [rows, cols, 8]     neighbor k is a wall box (k indexes NEIGHBOR_OFFSETS)
      nb_rect  [rows, cols, 8, 4]  neighbor box rectangle (xl, xr, yb, yt); for non-wall
                                    neighbors a far-away dummy rectangle so dist is huge
      rows, cols  ints

    Example, large map cell (1, 8) (open, wall above at (0, 8)): nb_wall[1, 8, 1] = True and
    nb_rect[1, 8, 1] = (2.0, 3.0, 3.5, 4.5).
    """
    wall = np.array(MAPS[map_name], dtype=np.uint8)
    rows, cols = wall.shape

    nb_wall = np.zeros((rows, cols, 8), dtype=bool)
    nb_rect = np.full((rows, cols, 8, 4), 1e6)
    nb_mask = np.zeros((rows, cols), dtype=np.uint8)

    # cell (i, j) spans x in [j - cols/2, j+1 - cols/2], y in [rows/2 - (i+1), rows/2 - i]
    for i in range(rows):
        for j in range(cols):
            if wall[i, j]:
                continue
            for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
                ii, jj = i + di, j + dj
                if 0 <= ii < rows and 0 <= jj < cols and wall[ii, jj]:
                    nb_wall[i, j, k] = True
                    nb_mask[i, j] |= 1 << k
                    nb_rect[i, j, k] = (jj - cols / 2.0, (jj + 1) - cols / 2.0,
                                       rows / 2.0 - (ii + 1), rows / 2.0 - ii)

    return dict(wall=wall, nb_wall=nb_wall, nb_rect=nb_rect, nb_mask=nb_mask,
                rows=rows, cols=cols)
