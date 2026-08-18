"""Shared ground truth for the AntMaze environment family (numpy + xml only, no jax).

Holds the maze presets (umaze / medium / large), the environment configuration dataclass, and
the MJCF builder that turns the vendored Gymnasium ant model plus a maze map into one model the
MJX stepper loads. Reference semantics: Gymnasium-Robotics 1.3.1 `AntMaze_<Map>-v5`
(`gymnasium_robotics/envs/maze/ant_maze_v5.py` + `maze_v4.py`), whose numbers are restated in
`spec.md` beside this file. The maze maps are the same three grids the PointMaze family uses
(`gymnasium_robotics.envs.maze.maps.U_MAZE / MEDIUM_MAZE / LARGE_MAZE`), scaled by 4, so they
are imported from `pm_common` rather than copied.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from os import path
from typing import Tuple

from ..pointmaze.pm_common import MAPS

# ---- reference constants (AntMaze-v5; see spec.md for provenance) ----
SCALING = 4.0        # maze_size_scaling: one maze cell is a 4 m x 4 m square
HEIGHT = 0.5         # maze_height: wall boxes are HEIGHT * SCALING = 2 m tall
GOAL_RADIUS = 0.45   # sparse reward paid when |torso xy - goal xy| <= this, in metres
FRAME_SKIP = 5       # one environment step = 5 physics steps of 0.01 s (20 Hz control)

ANT_XML = path.join(path.dirname(__file__), "assets", "ant.xml")


@dataclass(frozen=True)
class AntMazeConfig:
    """All env-side knobs. Build one with `preset(map_name)` for the standard three mazes.

    The reference resets the ant with zero joint noise (`reset_noise_scale=0.0` in
    `ant_maze_v5.py`) and this platform sets the cell-position noise to zero as well (the
    PointMaze convention since 2026-08-16), so every reset is fully deterministic: the only
    randomness in an episode is the policy's own sampling.
    """
    map_name: str = "umaze"
    start_cell: Tuple[int, int] = (3, 1)     # (row, col), row 0 at top
    goal_cell: Tuple[int, int] = (1, 1)
    max_episode_steps: int = 700
    continuing_task: bool = True             # True: never terminated (only truncated at the cap)
    reward_shift: float = 0.0                # added to every step's reward
    goal_radius: float = GOAL_RADIUS
    frame_skip: int = FRAME_SKIP
    # ---- solver settings the fused model runs with (deviation from the reference; spec.md) ----
    # The reference ant.xml integrates with RK4 at MuJoCo's default solver iterations. Under MJX
    # that compiles to a program about 400x slower than the settings below, so the fused model
    # runs implicitfast with a fixed Newton iteration budget. iterations=4 / ls_iterations=8 was
    # chosen because the passive settling height of the C-MuJoCo model at these settings equals
    # the RK4 full-solver height to 1e-4 (probe 3, 2026-08-18), while iterations=1 — the MJX
    # example-model setting — is unstable for this ant (it NaNs in float32).
    integrator: str = "implicitfast"
    solver_iterations: int = 4
    ls_iterations: int = 8


def preset(map_name: str) -> AntMazeConfig:
    """The standard configuration of one of the three mazes.

    Start and goal follow the PointMaze platform convention — start at the bottom-left open
    cell, goal at the far end — with the episode caps of the reference registrations
    (700 for umaze, 1000 for medium and large).
    """
    presets = {
        "umaze": AntMazeConfig("umaze", (3, 1), (1, 1), 700),
        "medium": AntMazeConfig("medium", (6, 1), (1, 6), 1000),
        "large": AntMazeConfig("large", (7, 1), (1, 10), 1000),
    }
    if map_name not in presets:
        raise ValueError(f"unknown antmaze map {map_name!r}; presets exist for {sorted(presets)}")
    return presets[map_name]


def cell_center(cell: Tuple[int, int], rows: int, cols: int) -> Tuple[float, float]:
    """World (x, y) of a cell centre, in metres. Example: large map (7,1) -> (-18.0, -12.0)."""
    i, j = cell
    return ((j + 0.5) - cols / 2.0) * SCALING, (rows / 2.0 - (i + 0.5)) * SCALING


def merged_wall_boxes(map_name: str):
    """Horizontal runs of wall cells merged into single boxes, as (x, y, half_x, half_y) metres.

    before: umaze row 0 = [1,1,1,1,1] -> five separate 4 m boxes (the reference emits one geom
            per wall cell)
    after:  one box centred on the run, 20 m wide
    The union of boxes is identical, so the collision geometry is the reference's; merging only
    cuts the number of geoms MJX pairs against the ant (45 -> 20 on the large map).
    """
    maze_map = MAPS[map_name]
    rows, cols = len(maze_map), len(maze_map[0])
    xc, yc = cols / 2.0 * SCALING, rows / 2.0 * SCALING
    boxes = []
    for i in range(rows):
        j = 0
        while j < cols:
            if maze_map[i][j] == 1:
                j0 = j
                while j < cols and maze_map[i][j] == 1:
                    j += 1
                boxes.append((((j0 + j) / 2.0) * SCALING - xc, yc - (i + 0.5) * SCALING,
                              (j - j0) / 2.0 * SCALING, 0.5 * SCALING))
            else:
                j += 1
    return boxes


def build_antmaze_xml(cfg: AntMazeConfig) -> str:
    """The vendored ant.xml with the maze walls and the fused solver settings, as an XML string.

    Wall boxes carry the reference's contact attributes (`contype=1 conaffinity=1`, box type,
    z centred at HEIGHT/2 * SCALING = 1 m), placed exactly where `maze_v4.Maze.make_maze` puts
    them; only the per-cell-to-per-run merge and the option line differ (spec.md, deviations).
    """
    tree = ET.parse(ANT_XML)
    root = tree.getroot()
    worldbody = tree.find(".//worldbody")
    for k, (x, y, hx, hy) in enumerate(merged_wall_boxes(cfg.map_name)):
        ET.SubElement(worldbody, "geom", name=f"block_{k}",
                      pos=f"{x} {y} {HEIGHT / 2.0 * SCALING}",
                      size=f"{hx} {hy} {HEIGHT / 2.0 * SCALING}",
                      type="box", contype="1", conaffinity="1", rgba="0.7 0.5 0.3 1.0")
    opt = tree.find(".//option")
    opt.set("integrator", cfg.integrator)
    opt.set("iterations", str(cfg.solver_iterations))
    opt.set("ls_iterations", str(cfg.ls_iterations))
    return ET.tostring(root, encoding="unicode")
