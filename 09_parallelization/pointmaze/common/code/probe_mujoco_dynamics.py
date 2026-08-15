"""Probe the real Gymnasium-Robotics PointMaze MuJoCo dynamics and check a closed-form model.

Goal: establish the exact per-step update rule of PointMaze_Large-v3 so the GPU
reimplementations (torch / CUDA / JAX) can match it. Prints model constants, then compares
MuJoCo's step against the semi-implicit Euler prediction with implicit joint damping:

    v' = (m * v + h * g * a) / (m + h * d)      # M dv = h (f - d v') solved for v'
    q' = q + h * v'

where m = ball mass, h = timestep, g = actuator gear, d = joint damping, a = clipped action.
"""
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def main():
    # build the raw env (no wrappers) with a fixed start far from walls so probes stay free-space
    env = gym.make("PointMaze_Large-v3", continuing_task=True, max_episode_steps=10000).unwrapped
    env.reset(seed=0, options={"reset_cell": [1, 8], "goal_cell": [1, 10]})
    env.position_noise_range = 0.0

    model = env.point_env.model
    data = env.point_env.data

    # dump every constant the reimplementation needs
    print(f"timestep        {model.opt.timestep}")
    print(f"integrator      {model.opt.integrator}  (0=Euler)")
    print(f"gravity         {model.opt.gravity}")
    print(f"body masses     {model.body_mass}")
    print(f"dof damping     {model.dof_damping}")
    print(f"actuator gear   {model.actuator_gear[:, 0]}")
    print(f"geom sizes      ball={model.geom('particle_geom').size}")
    print(f"frame_skip      {env.point_env.frame_skip}")
    print(f"maze scaling    {env.maze.maze_size_scaling}, height {env.maze.maze_height}")
    print(f"map shape       {len(env.maze.maze_map)} rows x {len(env.maze.maze_map[0])} cols")
    print(f"x_map_center    {env.maze.x_map_center}, y_map_center {env.maze.y_map_center}")

    m = float(model.body_mass[model.body('particle').id])
    h = float(model.opt.timestep)
    d = float(model.dof_damping[0])
    g = float(model.actuator_gear[0, 0])

    # free-space check: random actions, compare every step's (q', v') to the closed form
    rng = np.random.default_rng(0)
    home = np.array([2.5, 3.0])
    env.point_env.set_state(home, np.array([0.0, 0.0]))
    q = data.qpos.copy()
    v = data.qvel.copy()
    max_qerr = max_verr = 0.0
    for t in range(200):
        if t % 25 == 0:
            env.point_env.set_state(home, np.array([0.0, 0.0]))
            q = data.qpos.copy(); v = data.qvel.copy()
        a = rng.uniform(-1, 1, size=2)
        obs, _, _, _, _ = env.step(a)
        v_clip = np.clip(v, -5.0, 5.0)
        v_pred = (m * v_clip + h * g * np.clip(a, -1, 1)) / (m + h * d)
        q_pred = q + h * v_pred
        max_qerr = max(max_qerr, np.abs(q_pred - data.qpos).max())
        max_verr = max(max_verr, np.abs(v_pred - data.qvel).max())
        q = data.qpos.copy()
        v = data.qvel.copy()
    print(f"free-space closed-form errors over 200 steps: max|q| {max_qerr:.3e}  max|v| {max_verr:.3e}")

    # velocity-clip check: accelerate +x along the open row-1 corridor (x from 0.5 toward the
    # far wall face at x=5), short enough to stay >1 m clear of the wall
    env.point_env.set_state(np.array([0.5, 3.0]), np.array([0.0, 0.0]))
    q = data.qpos.copy(); v = data.qvel.copy()
    max_qerr2 = max_verr2 = 0.0
    for t in range(60):
        a = np.array([1.0, 0.0])
        env.step(a)
        v_clip = np.clip(v, -5.0, 5.0)
        v_pred = (m * v_clip + h * g * a) / (m + h * d)
        q_pred = q + h * v_pred
        max_qerr2 = max(max_qerr2, np.abs(q_pred - data.qpos).max())
        max_verr2 = max(max_verr2, np.abs(v_pred - data.qvel).max())
        q = data.qpos.copy(); v = data.qvel.copy()
    print(f"vel-clip closed-form errors over 60 steps:  max|q| {max_qerr2:.3e}  max|v| {max_verr2:.3e}")
    print(f"terminal speed reached: {data.qvel[0]:.6f} (clip 5.0, one post-clip step -> {(m*5.0 + h*g)/(m + h*d):.6f})")

    # wall-contact observation: drive into the wall left of cell (4,4)... map row 4 col 3 area
    env.point_env.set_state(np.array([2.5, 3.0]), np.array([0.0, 0.0]))
    for t in range(300):
        env.step(np.array([0.0, 1.0]))  # push +y (up) into the border wall (face at y=3.5)
    print(f"after 300 steps pushing +y: pos {data.qpos}, vel {data.qvel}")
    print(f"maze row of y: {env.maze.cell_xy_to_rowcol(data.qpos)}")

    env.close()


if __name__ == "__main__":
    main()
