"""Named environment setups: one frozen dataclass per configured environment.

Every train run passes its environment hyperparameters through ONE named EnvSetup (train.py
--env_setup <name>), so each env configuration is a durable, reproducible record. Individual CLI
flags still override single fields (train.py applies the setup only to flags the command line did
not set explicitly).

Registry contents:
- ``initial_single_large_pointmaze_max_400`` — the previous project setup (runs <= train run 5),
  verbatim: continuing PointMaze_Large-v3, top-right goal, 400-step cap, +-0.25 m reset noise.
- 16 section-8 configs (2026-07): 8 base envs (PointMaze v3 / AntMaze v5, UMaze/Open/Medium/Large)
  x 2 start/goal sets, named by the start corner (..._start_bottom_left / ..._start_top_left).
  All use exact-cell-center resets (no noise), continuing_task=False (terminated=True at the
  goal), the ExPLORe reward shift (-1 per step, 0 at the goal), the registered default episode
  limit (max_episode_steps=None), and discount 0.99.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class EnvSetup:
    """One named environment configuration (all env-side knobs of a train run)."""
    name: str
    env_id: str
    start_cell: Tuple[int, int]          # (row, col), 0-based, row 0 at the top
    goal_cell: Tuple[int, int]
    continuing_task: bool                # False => terminated=True at distance <= 0.45 m
    reset_target: bool
    position_noise_range: float          # 0.0 => exact cell centers (set on the built env)
    max_episode_steps: Optional[int]     # None => the env id's registered default
    reward_shift: float                  # added to every step's extrinsic reward (-1 = ExPLORe)
    attach_xy_to_state: bool             # AntMaze: re-attach achieved_goal (x, y) to the state
    include_contact_forces: bool         # AntMaze: keep the 78-d contact forces in the state
    discount_factor: float               # SAC discount used with this env


def _section8_setup(env_id: str, start_corner: str, start_cell, goal_cell) -> EnvSetup:
    """Build one section-8 config: shared knobs fixed, per-env id/cells/start-corner name filled in."""
    return EnvSetup(
        name=f"{env_id}_start_{start_corner}",
        env_id=env_id,
        start_cell=tuple(start_cell),
        goal_cell=tuple(goal_cell),
        continuing_task=False,
        reset_target=False,
        position_noise_range=0.0,
        max_episode_steps=None,
        reward_shift=-1.0,
        attach_xy_to_state=env_id.startswith("AntMaze"),
        include_contact_forces=False,
        discount_factor=0.99,
    )


# Corner cells per map (0-based (row, col), row 0 at the top), verified open in the registered
# gymnasium-robotics 1.3.1 map arrays (PointMaze and AntMaze share identical maps per size class).
# UMaze uses the two ends of the U corridor (top-left / bottom-left); the other maps use the two
# corner-to-corner diagonals.
#   set (1) start_bottom_left: start = bottom-left, goal = top-left (UMaze) / top-right (others)
#   set (2) start_top_left:    start = top-left,    goal = bottom-left (UMaze) / bottom-right (others)
_CORNERS = {
    # size class: (bottom_left, top_left, top_right, bottom_right)
    "UMaze": ((3, 1), (1, 1), None, None),
    "Open": ((3, 1), (1, 1), (1, 5), (3, 5)),
    "Medium": ((6, 1), (1, 1), (1, 6), (6, 6)),
    "Large": ((7, 1), (1, 1), (1, 10), (7, 10)),
}


def _section8_pair(env_id: str, size: str):
    """The two start/goal sets for one env id: UMaze pairs the U's two ends, others go corner-to-corner."""
    bottom_left, top_left, top_right, bottom_right = _CORNERS[size]
    if size == "UMaze":
        return [
            _section8_setup(env_id, "bottom_left", bottom_left, top_left),
            _section8_setup(env_id, "top_left", top_left, bottom_left),
        ]
    return [
        _section8_setup(env_id, "bottom_left", bottom_left, top_right),
        _section8_setup(env_id, "top_left", top_left, bottom_right),
    ]


_SECTION8_ENV_IDS = [
    ("PointMaze_UMaze-v3", "UMaze"),
    ("PointMaze_Open-v3", "Open"),
    ("PointMaze_Medium-v3", "Medium"),
    ("PointMaze_Large-v3", "Large"),
    ("AntMaze_UMaze-v5", "UMaze"),
    ("AntMaze_Open-v5", "Open"),
    ("AntMaze_Medium-v5", "Medium"),
    ("AntMaze_Large-v5", "Large"),
]

ENV_SETUPS: "dict[str, EnvSetup]" = {
    setup.name: setup
    for env_id, size in _SECTION8_ENV_IDS
    for setup in _section8_pair(env_id, size)
}

# The previous project setup (every run up to train run 5), recorded verbatim: continuing task on
# the Large map with the top-right goal, 400-step cap, the default +-0.25 m reset noise, raw sparse
# reward (no shift), 4-d state (PointMaze already carries x, y), discount 0.999.
ENV_SETUPS["initial_single_large_pointmaze_max_400"] = EnvSetup(
    name="initial_single_large_pointmaze_max_400",
    env_id="PointMaze_Large-v3",
    start_cell=(7, 1),
    goal_cell=(1, 10),
    continuing_task=True,
    reset_target=False,
    position_noise_range=0.25,
    max_episode_steps=400,
    reward_shift=0.0,
    attach_xy_to_state=False,
    include_contact_forces=False,
    discount_factor=0.999,
)

ENV_SETUP_NAMES = list(ENV_SETUPS)
