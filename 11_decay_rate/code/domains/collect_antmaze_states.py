#!/usr/bin/env python
"""Collect the fixed `antmaze_states` evaluation domain: real AntMaze_Large-v5 observation
vectors, spatially stratified over the maze cells, frozen as one .npy artifact.

State convention follows the parent project's AntMaze network input: 29 dimensions =
[x, y] (achieved_goal) + the 27 contact-force-free core dimensions of `observation`
(gymnasium-robotics appends 78 contact-force dims we drop). Collection is a fixed-seed random
policy from scattered reset positions; stratification keeps at most STATES_PER_CELL states
per maze cell so the set covers the maze rather than the neighborhood of the start.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python collect_antmaze_states.py
Writes ../../domains_data/antmaze_states.npy (P, 29) float32 + a .meta.json sidecar.
"""
import json
import os

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers AntMaze)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "domains_data")

SEED = 0
N_EPISODES = 60          # random-policy episodes (fresh reset each; resets scatter start cells)
STEPS_PER_EPISODE = 300
STATES_PER_CELL = 6      # stratification cap per maze cell
CORE_DIMS = 27           # observation dims kept (contact forces dropped)


def cell_of(env, xy):
    """Map world (x, y) to the maze's (row, col) cell index."""
    # gymnasium-robotics Maze: cell_rowcol_to_xy exists; invert via the maze's own converter
    return tuple(int(v) for v in env.unwrapped.maze.cell_xy_to_rowcol(np.asarray(xy)))


def main():
    """Collect, stratify, freeze."""
    env = gym.make("AntMaze_Large-v5")
    rng = np.random.default_rng(SEED)
    by_cell = {}
    # random play: each episode resets (start cell varies with the env's own reset sampling)
    # and steps a uniform-random policy; every step contributes one candidate state
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        for _ in range(STEPS_PER_EPISODE):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            xy = obs["achieved_goal"]
            state = np.concatenate([xy, obs["observation"][:CORE_DIMS]]).astype(np.float32)
            by_cell.setdefault(cell_of(env, xy), []).append(state)
            if term or trunc:
                break
    env.close()
    # stratify: at most STATES_PER_CELL per visited cell, chosen by the seeded rng
    # before: by_cell = {(1,1): [array x 260], (1,2): [array x 31], ...}
    # after: states = 6 per cell, concatenated in sorted-cell order (deterministic)
    states = []
    for cell in sorted(by_cell):
        pool = by_cell[cell]
        idx = rng.permutation(len(pool))[:STATES_PER_CELL]
        states.extend(pool[i] for i in sorted(idx))
    arr = np.stack(states)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "antmaze_states.npy"), arr)
    meta = {"seed": SEED, "n_episodes": N_EPISODES, "steps_per_episode": STEPS_PER_EPISODE,
            "states_per_cell": STATES_PER_CELL, "core_dims": CORE_DIMS,
            "n_cells_visited": len(by_cell), "n_states": int(arr.shape[0]),
            "state_dim": int(arr.shape[1]),
            "layout": "[x, y, observation[:27]] per the parent project's AntMaze input"}
    with open(os.path.join(OUT_DIR, "antmaze_states.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
