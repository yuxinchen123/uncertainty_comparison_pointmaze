"""Run 07_reconstruction's own visit-count wrapper over a supplied trajectory and dump the result.

This file runs under 07_reconstruction's interpreter, not the platform's: its wrapper is a
gymnasium wrapper and the platform environment holds no gymnasium. The platform's parity test
invokes it as a subprocess and compares what it wrote.

The environment is a stub. The wrapper only ever asks its environment for the maze map, the cell
size, and the next observation, so a stub that hands back the supplied trajectory exercises
exactly the code under test — the discretisation and the counting — with nothing else in the way.

Usage:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python dump_07_visit_counts.py \
      <trajectory.npy> <decay> <output.npz>
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

SEVEN = Path("/p/rlprojects/RND/07_reconstruction")
sys.path.insert(0, str(SEVEN / "src"))
from rnd_exploration.envs.point_maze_wrappers import (  # noqa: E402
    PositionVelocityVisitCountWrapper)
from rnd_exploration.methods.visit_count import VisitCount  # noqa: E402

# the large maze, copied from the platform's own constants so the two sides count the same world
LARGE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


class Maze:
    """What the wrapper reads off `env.unwrapped.maze`."""

    def __init__(self):
        """The map and the size of one cell, in metres."""
        self.maze_map = LARGE
        self.maze_size_scaling = 1.0


class StubEnv(gym.Env):
    """An environment that only replays a supplied trajectory, one observation per step.

    It subclasses gymnasium's Env because the wrapper's constructor insists on that; nothing of
    the base class is used.
    """

    def __init__(self, trajectory):
        """Hold the trajectory and a cursor into it."""
        # gymnasium's Env already defines `unwrapped` as a property returning self, which is
        # what the wrapper reads the maze off
        self.maze = Maze()
        self.trajectory = trajectory
        self.cursor = 0

    def step(self, action):
        """Return the next observation of the trajectory, with no reward and no termination."""
        observation = self.trajectory[self.cursor]
        self.cursor += 1
        return observation, 0.0, False, False, {}


def main():
    """Count the trajectory, then score every one of its rows from the finished table."""
    trajectory_file, decay, output_file = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    trajectory = np.load(trajectory_file)

    wrapper = PositionVelocityVisitCountWrapper(StubEnv(trajectory))
    for _ in range(len(trajectory)):
        wrapper.step(None)

    # the bonus is read after collection, from the table that already holds this trajectory
    model = VisitCount(wrapper, intrinsic_decay_rate=decay)
    counts = np.asarray([wrapper.observation_to_count(row) for row in trajectory], dtype=np.int64)
    bonus = np.asarray([model._count_to_bonus(int(n)) for n in counts], dtype=np.float64)

    np.savez(output_file, table=wrapper.get_visit_counts(), counts=counts, bonus=bonus)
    print(f"07_reconstruction counted {len(trajectory)} steps, "
          f"{int(wrapper.get_visit_counts().sum())} of them onto open cells")


if __name__ == "__main__":
    main()
