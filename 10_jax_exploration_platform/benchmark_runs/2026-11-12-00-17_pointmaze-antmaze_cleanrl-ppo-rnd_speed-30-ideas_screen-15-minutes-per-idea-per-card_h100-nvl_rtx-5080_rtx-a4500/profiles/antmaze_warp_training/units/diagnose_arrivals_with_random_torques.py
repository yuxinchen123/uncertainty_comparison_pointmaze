"""Do ants reach the goal of the U-maze under random torques, and how do they get there?

The start cell and the goal cell of the U-maze are 8 m apart with a 4 m wall cell between them; the
way around the wall is about 16 m. An untrained policy does not walk that in 700 steps. Yet the
training-iteration measurement of 2026-11-12 counted 3 episodes that ended at the goal among 32,771
with the Warp physics (1,024 policies x 16 environments, 15 iterations), and the earlier measurement
1 among 2,049. This script looks at such episodes.

It steps the environments with random torques (standard normal, clipped to [-1, 1], fresh at every
step: what an untrained policy sends), the episode ending at the goal or after 700 steps, and keeps
the torso's position and velocity of every step. For every episode that ended at the goal it
writes: the step of the episode, the torso's height and speed at the arrival, the largest height,
speed and one-step move of the episode, the maze cells the torso went through, and the torso's
position at chosen steps before the arrival. For all environments together it writes how many
steps had the torso above the walls' top (2 m), faster than 10 m/s, or inside a wall cell.

Since the model has its ceiling (2026-11-12) the record also holds: the steps with the torso in the
ceiling zone (above 1.5 m) and at the ceiling (above 1.74 m, where the torso sphere meets it), the
episodes with at least one such step, the environment's own fault record summed over all
environments, the steps faster than 2 and 5 m/s, percentiles of the torso's height and speed, and
for every episode that ended at the goal whether it went around the wall (through the cell at the
far end of the wall) and its steps in the ceiling zone.

Run on a card, never on a login node:
  <platform_jax_warp python> diagnose_arrivals_with_random_torques.py --physics warp \
      --environments 16384 --steps 1400 --out <file.json>
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
WALL_TOP = 2.0          # metres: maze_height 0.5 x maze_size_scaling 4
FAST = 10.0             # metres per second: far above what the ant's legs give
LOOK_BACK = (1, 2, 5, 10, 20, 50, 100)


def keyed(name, *coords):
    """A JAX random key for one named random quantity (rng-seeding rule: a keyed hash, base seed 0)."""
    import jax
    # before: name = "arrival torques", coords = (3,); after: PRNGKey(low 32 bits of sha256("0::arrival torques::3"))
    text = "::".join(str(part) for part in (0, name, *coords))
    return jax.random.PRNGKey(int(hashlib.sha256(text.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def cell_of(x, y, rows, cols, scaling=4.0):
    """The maze cell (row, column) of world positions; row 0 on top. Example: (-4, -4) in a 5 x 5 map -> (3, 1)."""
    import numpy as np
    return (np.floor(rows / 2.0 - y / scaling).astype(int), np.floor(x / scaling + cols / 2.0).astype(int))


def main():
    """Step the environments with random torques and write what the episodes that reached the goal did."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--physics", required=True, choices=["warp", "jax"])
    parser.add_argument("--environments", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--chunk", type=int, default=100, help="environment steps per compiled call")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal-on-the-start-cell", action="store_true",
                        help="to check this script: the goal is put on the start cell, so every step arrives")
    args = parser.parse_args()
    started = time.time()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import exploration_platform  # noqa: F401
    from exploration_platform.envs.antmaze.am_common import (CEILING_CONTACT_HEIGHT, CEILING_ZONE_HEIGHT,
                                                             preset)
    from exploration_platform.envs.pointmaze.pm_common import MAPS
    say = lambda text: print(f"[{time.time() - started:6.1f} s] {text}", flush=True)

    # GOAL: the environment of one physics, every environment its own copy (C = environments, N = 1).
    cfg = replace(preset("umaze"), continuing_task=False, physics=args.physics)
    if args.goal_on_the_start_cell:
        cfg = replace(cfg, goal_cell=cfg.start_cell)
    if args.physics == "warp":
        from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze as environment_class
    else:
        from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze as environment_class
    env = environment_class(cfg, args.environments, 1)
    n = args.environments

    def rollout(state, key):
        """`chunk` environment steps with fresh random torques; per step the ends and the torso."""
        def body(st, k):
            """One step: was the episode ended at the goal, ended at all, and the torso before the reset."""
            torque = jnp.clip(jax.random.normal(k, (n, 1, env.act_dim), jnp.float32), -1.0, 1.0)
            st, _, _, terminated, truncated, final_obs = env.step(st, torque)
            # before: final_obs [n, 1, 29] = qpos (15) then qvel (14); after: torso xyz and its linear velocity
            return st, (terminated[:, 0], truncated[:, 0], final_obs[:, 0, 0:3], final_obs[:, 0, 15:18])
        return jax.lax.scan(body, state, jax.random.split(key, args.chunk))

    rollout = jax.jit(rollout, donate_argnums=(0,))
    state = env.reset()
    parts = []
    for chunk in range(args.steps // args.chunk):
        state, outputs = rollout(state, keyed("arrival torques", chunk))
        parts.append([np.asarray(x) for x in outputs])
        if chunk < 2:
            say(f"chunk {chunk + 1} of {args.steps // args.chunk} done")
    terminated, truncated, position, velocity = (np.concatenate([p[i] for p in parts]) for i in range(4))
    done = terminated | truncated
    say(f"{position.shape[0]} steps of {n} environments done")

    # GOAL: what all environments did: above the walls, fast, inside a wall cell.
    maze = np.asarray(MAPS[cfg.map_name])
    rows, cols = maze.shape
    row, col = cell_of(position[..., 0], position[..., 1], rows, cols)
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    in_wall = np.zeros(row.shape, bool)
    in_wall[inside] = maze[row[inside], col[inside]] == 1
    speed = np.linalg.norm(velocity, axis=-1)
    in_zone = position[..., 2] > CEILING_ZONE_HEIGHT
    # GOAL: the episode each step belongs to, to count the episodes that entered the ceiling zone.
    # before: done [steps, n], True at an episode's last step
    # after:  episode [steps, n], the number of episodes of the environment that ended before the step
    episode = np.cumsum(done, axis=0) - done
    episodes_in_zone = sum(len(np.unique(episode[in_zone[:, b], b])) for b in np.nonzero(in_zone.any(0))[0])
    percentiles = (50, 90, 99, 99.9, 99.99)
    record = {
        "physics": args.physics, "environments": n, "steps": int(position.shape[0]),
        "goal_cell": list(cfg.goal_cell), "start_cell": list(cfg.start_cell),
        "environment_steps": int(position.shape[0] * n), "episodes_ended": int(done.sum()),
        "episodes_ended_at_the_goal": int(terminated.sum()),
        "non_finite_resets": int(np.asarray(state.nan_count).sum()),
        "steps_with_the_torso_above_the_walls_top": int((position[..., 2] > WALL_TOP).sum()),
        "steps_faster_than_10_metres_per_second": int((speed > FAST).sum()),
        "steps_with_the_torso_inside_a_wall_cell": int(in_wall.sum()),
        "steps_with_the_torso_outside_the_maze": int((~inside).sum()),
        "largest_torso_height": float(position[..., 2].max()), "largest_torso_speed": float(speed.max()),
        "ceiling_zone_height": CEILING_ZONE_HEIGHT, "ceiling_contact_height": CEILING_CONTACT_HEIGHT,
        "steps_with_the_torso_in_the_ceiling_zone": int(in_zone.sum()),
        "steps_with_the_torso_at_the_ceiling": int((position[..., 2] > CEILING_CONTACT_HEIGHT).sum()),
        "episodes_with_the_torso_in_the_ceiling_zone": int(episodes_in_zone),
        # the environment's own fault record (it looks at every physics step, this script at the
        # last physics step of each environment step, so the environment's count is the larger)
        "environment_fault_record": {
            "ceiling_zone_steps": int(np.asarray(state.ceiling_zone_steps, np.int64).sum()),
            "largest_torso_speed": float(np.asarray(state.largest_torso_speed).max())},
        "steps_faster_than_2_metres_per_second": int((speed > 2.0).sum()),
        "steps_faster_than_5_metres_per_second": int((speed > 5.0).sum()),
        "torso_height_percentiles": {str(q): float(np.percentile(position[..., 2], q)) for q in percentiles},
        "torso_speed_percentiles": {str(q): float(np.percentile(speed, q)) for q in percentiles},
        "pool": env.overflow_report(state) if args.physics == "warp" else None, "arrivals": []}

    # GOAL: every episode that ended at the goal, from its first step to the arrival.
    for t, b in zip(*np.nonzero(terminated)):
        earlier = np.nonzero(done[:t, b])[0]
        first = int(earlier[-1]) + 1 if len(earlier) else 0
        path, fast = position[first:t + 1, b], speed[first:t + 1, b]
        cells = [(int(r), int(c)) for r, c in zip(*cell_of(path[:, 0], path[:, 1], rows, cols))]
        # before: one cell per step, e.g. (3,1) (3,1) (2,1) (1,1); after: each cell once per visit, in order
        visited = [cell for k, cell in enumerate(cells) if k == 0 or cell != cells[k - 1]]
        moves = np.linalg.norm(np.diff(path[:, 0:2], axis=0), axis=-1) if len(path) > 1 else np.zeros(1)
        record["arrivals"].append({
            "environment": int(b), "step_of_the_episode": int(t - first + 1),
            "torso_at_the_arrival": [round(float(x), 3) for x in path[-1]],
            "speed_at_the_arrival": round(float(fast[-1]), 2),
            "largest_height_of_the_episode": round(float(path[:, 2].max()), 2),
            "largest_speed_of_the_episode": round(float(fast.max()), 2),
            "largest_move_in_one_step": round(float(moves.max()), 2),
            "cells_in_order": visited[:40],
            # around the U-maze's wall: through the open cell (2, 3) at the wall's far end, and
            # never inside a wall cell
            "went_around_the_wall": bool((2, 3) in visited and all(maze[r, c] == 0 for r, c in visited
                                                                   if 0 <= r < rows and 0 <= c < cols)),
            "steps_in_the_ceiling_zone_of_the_episode": int((path[:, 2] > CEILING_ZONE_HEIGHT).sum()),
            "torso_before_the_arrival": {str(back): [round(float(x), 2) for x in position[t - back, b]]
                                         for back in LOOK_BACK if t - back >= first}})
    record["wall_seconds"] = time.time() - started
    args.out.write_text(json.dumps(record, indent=1))
    say(f"{record['episodes_ended']:,} episodes ended, {record['episodes_ended_at_the_goal']} at the goal; "
        f"torso above the walls' top in {record['steps_with_the_torso_above_the_walls_top']:,} steps, faster than "
        f"10 m/s in {record['steps_faster_than_10_metres_per_second']:,}, inside a wall cell in "
        f"{record['steps_with_the_torso_inside_a_wall_cell']:,}; wrote {args.out}")
    for arrival in record["arrivals"][:20]:
        print("  ", json.dumps(arrival), flush=True)


if __name__ == "__main__":
    main()
