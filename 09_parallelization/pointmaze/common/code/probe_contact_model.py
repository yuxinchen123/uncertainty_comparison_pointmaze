"""Extract MuJoCo's exact contact-force law for the PointMaze ball-wall contact.

Places the ball at controlled gaps/velocities near the top border wall (face y=3.5, so
contact when y > 3.4 - margin-ish), reads the constraint-solver internals (efc_* arrays)
via mj_forward, then steps and records the outcome. Dumps everything to
fixtures/contact_probe.npz for offline fitting, and prints the solver constants.
"""
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco

gym.register_envs(gymnasium_robotics)

OUT = Path(__file__).resolve().parent.parent / "fixtures"


def main():
    env = gym.make("PointMaze_Large-v3", continuing_task=True, max_episode_steps=1000000).unwrapped
    env.reset(seed=0, options={"reset_cell": [1, 8], "goal_cell": [1, 10]})
    model, data = env.point_env.model, env.point_env.data

    # solver constants for the ball and wall geoms
    ball = model.geom("particle_geom")
    wall = model.geom("block_0_8")
    print("ball solref", ball.solref, "solimp", ball.solimp, "margin", ball.margin)
    print("wall solref", wall.solref, "solimp", wall.solimp, "margin", wall.margin)
    print("opt: timestep", model.opt.timestep, "iterations", model.opt.iterations,
          "solver", model.opt.solver, "tolerance", model.opt.tolerance,
          "impratio", model.opt.impratio, "cone", model.opt.cone)

    # grid of pre-step states: gap = 3.4 - y (positive = clear), vy, ay
    gaps = np.array([-0.004, -0.002, -0.001, -0.0005, -0.0002, 0.0, 0.0002, 0.0005,
                     0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005, 0.01])
    vys = np.array([-2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    ays = np.array([-1.0, 0.0, 1.0])

    rows = []
    efc_samples = []
    for gap in gaps:
        for vy in vys:
            for ay in ays:
                y = 3.4 - gap
                env.point_env.set_state(np.array([2.5, y]), np.array([0.0, vy]))
                # read solver internals at this state (before stepping)
                data.ctrl[:] = [0.0, ay]
                mujoco.mj_forward(model, data)
                ncon = int(data.ncon)
                efc = (data.efc_pos.copy(), data.efc_margin.copy(), data.efc_R.copy(),
                       data.efc_aref.copy(), data.efc_force.copy()) if ncon else None
                if efc is not None and len(efc_samples) < 8:
                    efc_samples.append((gap, vy, ay, ncon, [e.tolist() for e in efc],
                                        float(data.qacc[1])))
                env.step(np.array([0.0, ay]))
                rows.append([gap, vy, ay, ncon,
                             float(data.qpos[1]), float(data.qvel[1])])

    rows = np.array(rows)
    np.savez(OUT / "contact_probe.npz", rows=rows,
             columns=np.array(["gap", "vy", "ay", "ncon", "y_next", "vy_next"]))
    print(f"wrote {len(rows)} probe rows to fixtures/contact_probe.npz")
    print("\nsample efc internals (gap, vy, ay, ncon, [pos, margin, R, aref, force], qacc_y):")
    for s in efc_samples:
        print(" ", s)
    env.close()


if __name__ == "__main__":
    main()
