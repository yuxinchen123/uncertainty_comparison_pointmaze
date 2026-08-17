"""The platform's visit-count table must be 07_reconstruction's table.

A fixed trajectory is handed to both: to the fused implementation here, and — through a
subprocess, because the two live in different interpreters — to `PositionVelocityVisitCountWrapper`
and `VisitCount` in `07_reconstruction/src/rnd_exploration/`, running their own code. The count
table must be identical entry for entry, and the bonus of every step must agree.

The trajectory is deliberately awkward: it walks the whole world including positions outside the
maze, rests on wall cells, drives velocities past the +-5 the environment clips to, and revisits
the same discrete state many times, so the discretisation is tested at its edges rather than only
in its middle.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_visit_count_against_07.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import make_bonus  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import MAPS, EnvConfig  # noqa: E402

SEVEN_PYTHON = "/p/rlprojects/RND/.venvs/exploration/bin/python"
DUMPER = HERE / "dump_07_visit_counts.py"
ENV = EnvConfig()
WALL = np.asarray(MAPS[ENV.map_name])
ROWS, COLS = WALL.shape
PRESETS = {"gt_position_velocity_sqrt": -0.5, "gt_position_velocity_linear": -1.0}


def fixed_trajectory() -> np.ndarray:
    """A [M, 4] float32 trajectory chosen to reach the edges of the discretisation.

    before: nothing; after: 260 rows — a sweep across the whole world (including outside it), a
    stationary stretch on one open cell, and a set of velocities at and beyond the clip
    """
    rows = []
    # a sweep over the world and past its edges, so both the interior cells and the clipping are hit
    for i in range(120):
        t = i / 119.0
        rows.append([-7.0 + 14.0 * t, 5.5 - 11.0 * t, -6.0 + 12.0 * t, 6.0 - 12.0 * t])
    # forty steps resting on the start cell, so one entry reaches a count of forty
    rows += [[-4.5, -3.0, 0.0, 0.0]] * 40
    # a diagonal through the open corridors at a few fixed velocities
    for i in range(60):
        t = i / 59.0
        rows.append([-4.5 + 9.0 * t, -3.0 + 6.0 * t, 2.5, -2.5])
    # the velocity edges, on one open cell, in both directions and past the clip
    for v in (-9.0, -5.0, -4.999, -0.001, 0.0, 0.001, 4.999, 5.0, 9.0):
        for w in (-9.0, -5.0, 0.0, 5.0, 9.0):
            rows.append([-4.5, -3.0, v, w])
    return np.asarray(rows, dtype=np.float32)


def platform_result(preset: str, trajectory: np.ndarray):
    """The platform's table and per-row bonus after counting the trajectory as one rollout."""
    cfg = PPOConfig(n_copies=1, n_envs=1, num_steps=len(trajectory))
    bonus = make_bonus(preset)(cfg, ENV, 1, cfg.base_seed, [0])
    _params, state = bonus.init()
    state, reward, _extra = bonus.post_rollout({}, state, jnp.asarray(trajectory)[None], None)
    return np.asarray(state["counts"])[0], np.asarray(reward)[0]


def seven_result(preset: str, trajectory: np.ndarray, work: Path):
    """07_reconstruction's table and per-row bonus, produced by its own code in its own env."""
    trajectory_file = work / "trajectory.npy"
    output_file = work / f"07_{preset}.npz"
    np.save(trajectory_file, trajectory)
    finished = subprocess.run(
        [SEVEN_PYTHON, str(DUMPER), str(trajectory_file), str(PRESETS[preset]),
         str(output_file)],
        capture_output=True, text=True, env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"})
    if finished.returncode != 0:
        raise RuntimeError(f"07_reconstruction's dumper failed:\n{finished.stdout}\n"
                           f"{finished.stderr}")
    print(f"  {finished.stdout.strip()}")
    loaded = np.load(output_file)
    # 07 keeps the table as [rows, cols, 10, 10]; flattening it in C order gives exactly the
    # platform's index (row*cols + col)*100 + velocity_bin_x*10 + velocity_bin_y
    return loaded["table"].reshape(-1), loaded["bonus"], loaded["counts"]


def test_table_and_bonus_match_07():
    """Entry for entry on the table, and step by step on the bonus."""
    trajectory = fixed_trajectory()
    print(f"trajectory of {len(trajectory)} steps")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for preset, decay in PRESETS.items():
            print(f"{preset} (decay {decay}):")
            table_here, bonus_here = platform_result(preset, trajectory)
            table_07, bonus_07, counts_07 = seven_result(preset, trajectory, work)

            assert table_here.shape == table_07.shape, (
                f"table shapes differ: {table_here.shape} against {table_07.shape}")
            differing = int((table_here != table_07).sum())
            print(f"  table: {int(table_here.sum())} visits here, {int(table_07.sum())} there; "
                  f"{differing} of {table_here.size} entries differ; "
                  f"largest count {int(table_here.max())}")
            assert differing == 0, "the two count tables are not the same table"

            worst = float(np.abs(bonus_here.astype(np.float64) - bonus_07).max())
            print(f"  bonus: worst difference over {len(trajectory)} steps {worst:.2e}; "
                  f"range here [{bonus_here.min():.4f}, {bonus_here.max():.4f}], "
                  f"there [{bonus_07.min():.4f}, {bonus_07.max():.4f}]")
            assert worst <= 1e-6, "the two bonuses disagree beyond single-precision rounding"

            # the comparison is only worth anything if the trajectory actually revisited states
            assert int(counts_07.max()) >= 40, "the trajectory did not revisit any state"
            assert bonus_here.min() < 0.2, "no state was visited often enough to score low"
    print("ok test_table_and_bonus_match_07")


if __name__ == "__main__":
    test_table_and_bonus_match_07()
