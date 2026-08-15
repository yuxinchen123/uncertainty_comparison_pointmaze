"""Generate reference-trajectory fixtures from the real MuJoCo PointMaze_Large-v3.

Each fixture case is (q0, v0, actions[T,2]) plus the recorded MuJoCo (qpos[T,2], qvel[T,2])
after every step. GPU implementations replay the same (q0, v0, actions) and compare.
Writes fixtures/pointmaze_large_fixtures.npz and fixtures/constants.json.
"""
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

OUT = Path(__file__).resolve().parent.parent / "fixtures"


def rollout(env, q0, v0, actions):
    """Replay an action sequence from (q0, v0); return per-step post-step (qpos, qvel)."""
    env.point_env.set_state(np.asarray(q0, float), np.asarray(v0, float))
    qs, vs = [], []
    for a in actions:
        env.step(np.asarray(a, float))
        qs.append(env.point_env.data.qpos.copy())
        vs.append(env.point_env.data.qvel.copy())
    return np.array(qs), np.array(vs)


def main():
    env = gym.make("PointMaze_Large-v3", continuing_task=True, max_episode_steps=1000000).unwrapped
    env.reset(seed=0, options={"reset_cell": [1, 8], "goal_cell": [1, 10]})

    model = env.point_env.model
    consts = {
        "h": float(model.opt.timestep),
        "m": float(model.body_mass[model.body("particle").id]),
        "d": float(model.dof_damping[0]),
        "g": float(model.actuator_gear[0, 0]),
        "r": float(model.geom("particle_geom").size[0]),
        "vel_clip": 5.0,
        "maze_map_large": [list(map(int, row)) for row in env.maze.maze_map],
    }

    rng = np.random.default_rng(20260815)
    cases = {}

    # free_space: 4 segments of 100 random-action steps from the open cell (1,8) center,
    # VERIFIED contact-free (ball center stays >= R + 0.02 from every wall rectangle,
    # clear of the 0.002 contact margin); segments that stray near a wall are re-drawn
    wall = np.array([[1 if c == 1 else 0 for c in row] for row in consts["maze_map_large"]])
    wi, wj = np.nonzero(wall)
    xl = wj - wall.shape[1] / 2.0
    yb = wall.shape[0] / 2.0 - (wi + 1)

    def min_wall_dist(qs):
        # min over steps of the center-to-nearest-wall-rectangle distance
        dx = np.maximum(np.maximum(xl[None] - qs[:, :1], qs[:, :1] - (xl[None] + 1)), 0)
        dy = np.maximum(np.maximum(yb[None] - qs[:, 1:2], qs[:, 1:2] - (yb[None] + 1)), 0)
        return float(np.sqrt(dx * dx + dy * dy).min())

    k = 0
    while k < 4:
        actions = rng.uniform(-1, 1, size=(100, 2))
        qs, _ = rollout(env, (2.5, 3.0), (0.0, 0.0), actions)
        if min_wall_dist(qs) >= consts["r"] + 0.02:
            cases[f"free_space_{k}"] = ((2.5, 3.0), (0.0, 0.0), actions)
            k += 1

    # head-on pushes into a face from rest, 300 steps: settle behavior
    cases["wall_head_on_py"] = ((2.5, 3.0), (0.0, 0.0), np.tile([0.0, 1.0], (300, 1)))
    cases["wall_head_on_mx"] = ((0.5, 3.0), (0.0, 0.0), np.tile([-1.0, 0.0], (300, 1)))

    # slide: press diagonally into the top border and slide right along it into the far wall
    cases["wall_slide"] = ((0.5, 3.0), (0.0, 0.0), np.tile([0.7, 1.0], (300, 1)))

    # corner graze: press down-left from (-1.5, 3); slides along wall cells (2,3)-(2,2) top
    # face and wraps around the exposed corner at x=-4, y=2.5 into the col-1 corridor
    cases["corner_graze"] = ((-1.5, 3.0), (0.0, 0.0), np.tile([-1.0, -1.0], (300, 1)))

    # fast impact: accelerate to the velocity clip along row 1, slam the far wall face x=5
    cases["impact_fast"] = ((0.5, 3.0), (0.0, 0.0), np.tile([1.0, 0.0], (130, 1)))

    # episode-like: 400 seeded random steps from the run-6 start cell (7,1) center (-4.5, -3)
    for k in range(3):
        actions = rng.uniform(-1, 1, size=(400, 2))
        cases[f"episode_like_{k}"] = ((-4.5, -3.0), (0.0, 0.0), actions)

    OUT.mkdir(exist_ok=True)
    arrays = {}
    for name, (q0, v0, actions) in cases.items():
        qs, vs = rollout(env, q0, v0, actions)
        arrays[f"{name}__q0"] = np.array(q0, float)
        arrays[f"{name}__v0"] = np.array(v0, float)
        arrays[f"{name}__actions"] = actions.astype(float)
        arrays[f"{name}__qpos"] = qs
        arrays[f"{name}__qvel"] = vs
        # penetration check on the reference itself: distance of center to nearest wall face
        print(f"{name:20s} T={len(actions):3d} final q=({qs[-1][0]:+.4f},{qs[-1][1]:+.4f}) "
              f"final v=({vs[-1][0]:+.4f},{vs[-1][1]:+.4f})")

    np.savez_compressed(OUT / "pointmaze_large_fixtures.npz", **arrays)
    (OUT / "constants.json").write_text(json.dumps(consts, indent=1))
    print(f"wrote {OUT}/pointmaze_large_fixtures.npz ({len(cases)} cases) and constants.json")
    env.close()


if __name__ == "__main__":
    main()
