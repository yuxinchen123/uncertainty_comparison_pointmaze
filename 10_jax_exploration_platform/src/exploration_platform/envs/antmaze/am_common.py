"""Shared ground truth for the AntMaze environment family (numpy + xml only, no jax).

Holds the maze presets (umaze / medium / large), the environment configuration dataclass, and
the MJCF builder that turns the vendored Gymnasium ant model plus a maze map and a ceiling on the
wall tops into one model the MJX stepper loads. Reference semantics: Gymnasium-Robotics 1.3.1 `AntMaze_<Map>-v5`
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
PHYSICS_TIME_STEP = 0.01   # seconds per physics step: the `timestep` of the reference ant.xml

# ---- the ceiling (a deviation from the reference, decided in the session of 2026-11-12 and reported to the user, who may reverse it; spec.md, deviation 8) ----
# Under strong torques an ant can leave the floor and rise above the 2 m walls, and then cross
# them. One plane on the wall tops, with its normal pointing down, stops the torso there.
CEILING_HEIGHT = HEIGHT * SCALING   # metres: the plane lies on the wall tops, at 2 m
TORSO_RADIUS = 0.25                 # metres: the torso sphere of ant.xml
CONTACT_MARGIN = 0.01               # metres: the geom margin of ant.xml's default class
# torso-centre height at which the torso sphere comes within the margin of the ceiling: 1.74 m
CEILING_CONTACT_HEIGHT = CEILING_HEIGHT - TORSO_RADIUS - CONTACT_MARGIN
# The ceiling zone that the fault record counts: torso centre above 1.5 m, which is 0.5 m under
# the plane and 0.24 m under the height of first contact.
CEILING_ZONE_DEPTH = 0.5
CEILING_ZONE_HEIGHT = CEILING_HEIGHT - CEILING_ZONE_DEPTH
# Collision bits. ant.xml gives every geom contype 1; the floor and the walls have conaffinity 1
# and the ant's geoms conaffinity 0, so each ant geom meets the floor and the walls and nothing
# else. The ceiling answers only to bit 2 (conaffinity 2, contype 0), and only the torso sphere
# also carries that bit (contype 3), so the model gains exactly one pair, torso against ceiling.
CEILING_BIT = 2
TORSO_GEOM = "torso_geom"

ANT_XML = path.join(path.dirname(__file__), "assets", "ant.xml")
# the implementations of the physics step a configuration may name in its `physics` field
PHYSICS = ("jax", "warp")


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
    # MJX contact budget: after broadphase, at most max_geom_pairs pairs per collision-function
    # group and max_contact_points contact slots per condim. Without a cap MJX pads every
    # possible pair (225 slots on the umaze, more on the large map) and every Newton iteration
    # drags the whole padded contact state through memory — the large map then stops scaling
    # with the copy count. The ant can genuinely touch the floor and at most a wall or two at
    # once, so 8 pairs / 32 points is generous. These are MJX-only numerics; the C-MuJoCo
    # reference ignores them, so the parity gates compare against the uncapped reference.
    max_geom_pairs: int = 8
    max_contact_points: int = 32
    # ---- how the MJX stepper runs that model (speed only; `mjx_antmaze` reads these two) ----
    # solver_loop: "while_loop" calls mjx.step as shipped, whose Newton passes run in a
    # jax.lax.while_loop; "fixed_count" runs the same passes in a loop of exactly
    # solver_iterations passes in which an environment whose stop test is met keeps its values,
    # each pass its own copy in the compiled program; "fixed_count_scan" is that loop as one
    # jax.lax.scan.
    # carried_state: "whole_data" carries the whole MJX data from one step to the next;
    # "next_step_inputs" carries only what the next step reads (positions, velocities, the
    # solver's start values, the clock) and rebuilds the rest inside the step.
    solver_loop: str = "while_loop"
    carried_state: str = "whole_data"
    # ---- which implementation runs the physics step (the composer reads `physics`) ----
    # "jax": MJX's JAX implementation (`mjx_antmaze.JaxAntMaze`), the stepper of every run before
    # 2026-11-12. "warp": MuJoCo Warp called from JAX (`warp_antmaze.WarpAntMaze`), about 8 times
    # as fast on an H100 NVL; it needs an interpreter with warp-lang
    # (/p/rlprojects/RND/.venvs/platform_jax_warp). The model, the reward, the termination and the
    # reset are the same; the contact numerics differ within the gates of spec.md.
    # The two sizes below are read by `warp_antmaze` only. All environments of a process share one
    # pool of contact slots (and one of candidate geom pairs) of
    # warp_contact_slots_per_environment x (policies x environments per policy) slots; one
    # environment holds at most warp_constraint_rows_per_environment constraint rows. MuJoCo Warp
    # drops what does not fit, and the trainer then stops the unit.
    physics: str = "jax"
    warp_contact_slots_per_environment: int = 48
    # 512 since 2026-11-12: with 160 (sized from random torques, which needed 49) the first training unit
    # stopped at 161 rows after 2.4 million steps per policy; one ant pressed into a corner under the
    # ceiling can hold several dozen contacts of 4 rows each
    warp_constraint_rows_per_environment: int = 512
    # ---- the physics fault guard (`faults.py`; spec.md, deviation 9) ----
    # Both at 0.0: off; only a non-finite state is respawned, as in every run before 2026-11-12.
    # With either above zero the guard is on: an environment whose torso moved faster than
    # fault_torso_speed (metres per second) or turned faster than fault_torso_angular_speed
    # (radians per second) after any physics step, or whose torso centre left the maze's volume
    # (outer walls, floor, ceiling), is respawned like a non-finite one: no reward, no arrival, a
    # truncation that shows the spawn. A bound left at 0 is not tested.
    fault_torso_speed: float = 0.0
    fault_torso_angular_speed: float = 0.0
    # ---- the physics time step (spec.md, deviation 10) ----
    # Seconds per physics step; one environment step lasts frame_skip x physics_time_step seconds.
    # 0.01 with frame_skip 5 is the reference's 0.05 s and the physics of every run before
    # 2026-11-12. `implicitfast` at 0.01 s adds rotational energy to a torso that turns fast in the
    # air, and trained policies drove ants to torso speeds of 1e6 m/s and more that way; at 0.005 s
    # with frame_skip 10 (the same 0.05 s per environment step) no ant ran away in 127 million
    # environment steps of the same policies. Any positive value is accepted with any frame_skip;
    # the unit's start record holds both, so a run whose environment step is not 0.05 s shows it.
    physics_time_step: float = PHYSICS_TIME_STEP


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


def maze_half_extent(map_name: str) -> Tuple[float, float]:
    """Half the maze's width and half its depth in metres: the outer faces of its outer walls.

    before: "umaze" (5 x 5 cells of 4 m)   after: (10.0, 10.0)
    before: "large" (9 rows x 12 columns)   after: (24.0, 18.0)
    """
    maze_map = MAPS[map_name]
    return len(maze_map[0]) / 2.0 * SCALING, len(maze_map) / 2.0 * SCALING


def merged_wall_boxes(map_name: str):
    """Horizontal runs of wall cells merged into single boxes, as (x, y, half_x, half_y) metres.

    before: umaze row 0 = [1,1,1,1,1] -> five separate 4 m boxes (the reference emits one geom
            per wall cell)
    after:  one box centred on the run, 20 m wide
    The union of boxes is identical, so the collision geometry is the reference's; merging only
    cuts the number of geoms MJX pairs against the ant (62 -> 31 on the large map, 38 -> 20 on the
    medium map, 18 -> 8 on the umaze; spec.md, deviation 4).
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
    """The vendored ant.xml with the maze walls, the ceiling and the fused solver settings, as
    an XML string.

    Wall boxes carry the reference's contact attributes (`contype=1 conaffinity=1`, box type,
    z centred at HEIGHT/2 * SCALING = 1 m), placed exactly where `maze_v4.Maze.make_maze` puts
    them; the per-cell-to-per-run merge, the ceiling and the option line differ (spec.md,
    deviations).
    """
    tree = ET.parse(ANT_XML)
    root = tree.getroot()
    worldbody = tree.find(".//worldbody")
    for k, (x, y, hx, hy) in enumerate(merged_wall_boxes(cfg.map_name)):
        ET.SubElement(worldbody, "geom", name=f"block_{k}",
                      pos=f"{x} {y} {HEIGHT / 2.0 * SCALING}",
                      size=f"{hx} {hy} {HEIGHT / 2.0 * SCALING}",
                      type="box", contype="1", conaffinity="1", rgba="0.7 0.5 0.3 1.0")

    # GOAL: the ceiling, which only the torso sphere meets. The plane is the floor's (same size,
    # the default class's margin, friction and contact dimension) turned over and lifted to the
    # wall tops; it is not drawn (alpha 0).
    # before: torso_geom contype 1 (ant.xml's default)   -> pairs: torso-floor, torso-walls
    # after:  torso_geom contype 1 | CEILING_BIT = 3      -> the same pairs and torso-ceiling
    torso = root.find(f".//geom[@name='{TORSO_GEOM}']")
    torso.set("contype", str(1 | CEILING_BIT))
    ET.SubElement(worldbody, "geom", name="ceiling", type="plane", pos=f"0 0 {CEILING_HEIGHT}",
                  zaxis="0 0 -1", size="40 40 40", contype="0", conaffinity=str(CEILING_BIT),
                  rgba="0 0 0 0")
    # GOAL: the option line. ant.xml's own is RK4 at 0.01 s with MuJoCo's default iteration counts.
    # before: <option integrator="RK4" timestep="0.01"/>
    # after (the default configuration): <option integrator="implicitfast" timestep="0.01"
    #        iterations="4" ls_iterations="8"/>; with physics_time_step=0.005: timestep="0.005"
    if not cfg.physics_time_step > 0.0 or cfg.frame_skip < 1:
        raise ValueError(f"physics_time_step must be above zero and frame_skip at least 1; got "
                         f"{cfg.physics_time_step!r} and {cfg.frame_skip!r}")
    opt = tree.find(".//option")
    opt.set("timestep", str(cfg.physics_time_step))
    opt.set("integrator", cfg.integrator)
    opt.set("iterations", str(cfg.solver_iterations))
    opt.set("ls_iterations", str(cfg.ls_iterations))

    # The contact-budget numerics MUST be the model's only custom numerics, in this order.
    # MJX 3.11 reads them as `numeric_data[id]` — the data array indexed by the numeric's ID
    # rather than its address — so with ant.xml's 15-element `init_qpos` numeric in front, a
    # cap numeric with id 1 would be read from init_qpos's data and come out 0, silently
    # deleting every contact (the ant then falls through the floor). ant.xml's `init_qpos`
    # numeric is unused here (qpos0 comes from the body positions), so it is dropped and the
    # two size-1 caps take ids 0 and 1, whose ids equal their addresses.
    custom = root.find("custom")
    for numeric in list(custom):
        custom.remove(numeric)
    ET.SubElement(custom, "numeric", name="max_geom_pairs", data=str(cfg.max_geom_pairs))
    ET.SubElement(custom, "numeric", name="max_contact_points",
                  data=str(cfg.max_contact_points))
    return ET.tostring(root, encoding="unicode")
