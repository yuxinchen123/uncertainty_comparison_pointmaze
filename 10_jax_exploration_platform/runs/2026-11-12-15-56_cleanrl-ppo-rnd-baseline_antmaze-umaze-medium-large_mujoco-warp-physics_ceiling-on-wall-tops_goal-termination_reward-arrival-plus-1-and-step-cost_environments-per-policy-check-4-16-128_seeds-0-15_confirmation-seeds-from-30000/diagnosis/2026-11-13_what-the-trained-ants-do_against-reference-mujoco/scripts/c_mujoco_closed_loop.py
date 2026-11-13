"""Run trained AntMaze policies closed loop in C MuJoCo (float64) and record what the ants do.

The policies are the ones `roll_out_and_record.py` saved (`<name>_actors.npz`). Each episode starts at the
environment's spawn (`<name>_trajectories.npz`, `spawn_obs`), the action is the policy's mean plus its own
noise, clipped to [-1, 1], and the episode ends on arrival (torso within 0.45 m of the goal after an
environment step) or after 700 environment steps: the environment's own rules, without its fault guard
(the guard's bounds are only counted). The model is the environment's own (`build_antmaze_xml`: the ant,
the U-maze's walls, the ceiling on the wall tops). Several physics settings are run:

  ours                             implicitfast, 4 solver and 8 line-search iterations, 0.005 s x 10
  reference                        RK4, 100 and 50 (MuJoCo's defaults), 0.01 s x 5: the reference ant.xml
  rk4_at_our_step_and_solver       RK4, 4 and 8, 0.005 s x 10 (only the integrator differs from ours)
  implicit_at_our_step_and_solver  implicit (with the turning-rate derivatives), 4 and 8, 0.005 s x 10

Per episode it keeps: arrival and length, the largest height, speed and turning speed (after any physics
step), environment steps above 1.0 and 1.5 m, steps with no contact at all, steps with the torso centre
in a wall cell, how often the guard's bounds were passed, MuJoCo's own "unstable" resets, and the
mechanical energy that appeared without the motors supplying it while the ant had no contact.

Energy bookkeeping, per environment step t: define E(t) as the kinetic plus potential energy of the state
before step t (MuJoCo's own energy functions, joint armature included) and W(t) as the work of the motors
during step t (the sum over physics steps and joints of MuJoCo's actuator torque on the joint x the mean of the
joint's speed before and after the physics step x the physics time step).
Damping and friction only remove energy, so E(t+1) - E(t) - W(t) cannot be positive in exact physics while
nothing touches the ant. The record sums its positive values over the steps that begin and end with no
contact ("energy created in the air"), and the motors' work over the whole episode for scale.

Run on a compute node's processors, never on a login node:
  <platform_jax_warp python> c_mujoco_closed_loop.py --name <records/warp_rollout_...> --out <file> [--episodes 16]
"""
import argparse
import json
import os
import socket
import sys
import time
from dataclasses import replace
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))

# name: (integrator, solver iterations, line-search iterations, physics time step, physics steps per environment step)
SETTINGS = {"ours": ("implicitfast", 4, 8, 0.005, 10),
            "reference": ("RK4", 100, 50, 0.01, 5),
            "rk4_at_our_step_and_solver": ("RK4", 4, 8, 0.005, 10),
            "implicit_at_our_step_and_solver": ("implicit", 4, 8, 0.005, 10)}
COLUMNS = ("policy", "episode", "length", "arrived", "largest_height", "largest_speed", "largest_turning",
           "steps_above_1_0_m", "steps_above_1_5_m", "steps_without_contact", "steps_in_wall_cell",
           "physics_steps_beyond_guard", "unstable_resets", "energy_created_in_the_air", "motor_work_absolute",
           "largest_kinetic_energy", "mean_speed", "mean_height")
MAZE = ((1, 1, 1, 1, 1), (1, 0, 0, 0, 1), (1, 1, 1, 0, 1), (1, 0, 0, 0, 1), (1, 1, 1, 1, 1))
GOAL, GOAL_RADIUS, EPISODE_STEPS = (-4.0, 4.0), 0.45, 700
CACHE = {}


def model_of(setting):
    """The environment's C MuJoCo model with one of the physics settings, built once per process."""
    import mujoco
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset
    if setting not in CACHE:
        integrator, iterations, ls_iterations, timestep, substeps = SETTINGS[setting]
        cfg = replace(preset("umaze"), continuing_task=False, integrator=integrator, solver_iterations=iterations,
                      ls_iterations=ls_iterations, physics_time_step=timestep, frame_skip=substeps)
        model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        CACHE[setting] = (model, substeps)
    return CACHE[setting]


def in_wall(x, y):
    """Whether a point of the U-maze's floor plan lies in a wall cell (cells of 4 m, row 0 on top).

    before: x -4.0, y -4.0 (the spawn)   after: False;   before: x -4.0, y 0.0   after: True
    """
    column = min(max(int((x / 4.0 + 2.5) // 1), 0), 4)
    row = min(max(int((2.5 - y / 4.0) // 1), 0), 4)
    return bool(MAZE[row][column])


def energy(model, scratch, qpos, qvel):
    """Kinetic plus potential energy of a state, and its kinetic part, by MuJoCo's own functions."""
    import mujoco
    scratch.qpos[:], scratch.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, scratch)
    return float(scratch.energy[0] + scratch.energy[1]), float(scratch.energy[1])


def episodes_of(task):
    """All episodes of one policy with one physics setting; returns (setting, rows [episodes, len(COLUMNS)])."""
    import mujoco
    import numpy as np
    setting, policy, episodes, seed, actors_file, spawn = task
    model, substeps = model_of(setting)
    actors = np.load(actors_file)
    W0, b0, W1, b1, W2, b2 = (actors[name][policy].astype(np.float64) for name in ("W0", "b0", "W1", "b1", "W2", "b2"))
    spread = np.exp(actors["logstd"][policy].astype(np.float64))
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    dt = model.opt.timestep
    rows = []
    for episode in range(episodes):
        # one noise stream per (seed, policy, episode), whatever the setting: the same noise in every physics
        rng = np.random.default_rng([seed, policy, episode])
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = spawn[:15], spawn[15:]
        unstable_before = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        row = dict.fromkeys(COLUMNS, 0.0)
        row["policy"], row["episode"] = policy, episode
        speed_sum = height_sum = 0.0
        total, kinetic = energy(model, scratch, data.qpos, data.qvel)
        for step in range(EPISODE_STEPS):
            obs = np.concatenate([data.qpos, data.qvel]).astype(np.float32).astype(np.float64)
            hidden = np.tanh(np.tanh(obs @ W0 + b0) @ W1 + b1)
            data.ctrl[:] = np.clip(hidden @ W2 + b2 + spread * rng.standard_normal(8), -1.0, 1.0)
            # GOAL: one environment step; the motors' work, and whether anything touched the ant in it
            # (a physics step's contact count is that of the state the step starts from).
            work, touched = 0.0, False
            for _ in range(substeps):
                joint_speed_before = data.qvel[6:14].copy()
                mujoco.mj_step(model, data)
                touched = touched or data.ncon > 0
                work += float(data.qfrc_actuator[6:14] @ (joint_speed_before + data.qvel[6:14])) * 0.5 * dt
                speed = float(np.linalg.norm(data.qvel[0:3]))
                turning = float(np.linalg.norm(data.qvel[3:6]))
                row["largest_speed"] = max(row["largest_speed"], speed)
                row["largest_turning"] = max(row["largest_turning"], turning)
                row["physics_steps_beyond_guard"] += speed > 30.0 or turning > 100.0
            if int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > unstable_before:
                # MuJoCo found the state unusable and put the model back to its default state: the episode ends here
                row["length"] += 1
                break
            total_after, kinetic = energy(model, scratch, data.qpos, data.qvel)
            if not touched and scratch.ncon == 0:
                row["energy_created_in_the_air"] += max(total_after - total - work, 0.0)
            row["motor_work_absolute"] += abs(work)
            row["largest_kinetic_energy"] = max(row["largest_kinetic_energy"], kinetic)
            total = total_after
            height = float(data.qpos[2])
            row["length"] += 1
            row["largest_height"] = max(row["largest_height"], height)
            row["steps_above_1_0_m"] += height > 1.0
            row["steps_above_1_5_m"] += height > 1.5
            row["steps_without_contact"] += scratch.ncon == 0
            row["steps_in_wall_cell"] += in_wall(float(data.qpos[0]), float(data.qpos[1]))
            speed_sum += float(np.linalg.norm(data.qvel[0:3]))
            height_sum += height
            if (data.qpos[0] - GOAL[0]) ** 2 + (data.qpos[1] - GOAL[1]) ** 2 <= GOAL_RADIUS ** 2:
                row["arrived"] = 1.0
                break
        row["unstable_resets"] = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - unstable_before
        row["mean_speed"], row["mean_height"] = speed_sum / row["length"], height_sum / row["length"]
        rows.append([float(row[name]) for name in COLUMNS])
    return setting, np.asarray(rows)


def main():
    """Every policy with every setting, in parallel; one record file and a printed table per configuration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=Path, required=True, help="path of the rollout's files without the suffix")
    parser.add_argument("--episodes", type=int, default=16, help="per policy and setting")
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--policies", default="all", help="'all' or a list like 64-79,0-15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    started = time.time()
    actors_file = str(args.name) + "_actors.npz"
    actors = np.load(actors_file)
    spawn = np.load(str(args.name) + "_trajectories.npz")["spawn_obs"].astype(np.float64)
    count, per_cell = actors["W0"].shape[0], int(actors["policies_per_configuration"])
    configurations = actors["configurations"].tolist()
    if args.policies == "all":
        policies = list(range(count))
    else:
        # before: "64-79,0-3"   after: [64, ..., 79, 0, 1, 2, 3]
        policies = [p for part in args.policies.split(",") for p in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)]
    settings = args.settings.split(",")
    tasks = [(setting, policy, args.episodes, args.seed, actors_file, spawn) for setting in settings for policy in policies]
    tables = {setting: [] for setting in settings}
    with Pool(args.processes) as pool:
        for setting, rows in pool.imap_unordered(episodes_of, tasks, chunksize=1):
            tables[setting].append(rows)
    record = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
              "measured_iso": datetime.now().astimezone().isoformat(), "rollout": args.name.name,
              "settings": {name: dict(zip(("integrator", "solver_iterations", "line_search_iterations", "physics_time_step",
                                           "physics_steps_per_environment_step"), SETTINGS[name])) for name in settings},
              "episodes_per_policy": args.episodes, "policies": policies, "policies_per_configuration": per_cell,
              "configuration_fields": ["learning_rate", "ent_coef", "int_coef", "clip_coef"], "configurations": configurations,
              "columns": list(COLUMNS), "per_configuration": {}, "seconds": None}
    arrays = {}
    col = {name: index for index, name in enumerate(COLUMNS)}
    for setting in settings:
        table = np.concatenate(tables[setting])
        table = table[np.lexsort((table[:, col["episode"]], table[:, col["policy"]]))]
        arrays[setting] = table.astype(np.float32)
        record["per_configuration"][setting] = []
        print(f"== {setting}: {SETTINGS[setting]}")
        for cell, configuration in enumerate(configurations):
            rows = table[(table[:, col["policy"]] // per_cell) == cell]
            if not len(rows):
                continue
            steps = rows[:, col["length"]].sum()
            arrived = rows[:, col["arrived"]] > 0
            entry = {"configuration": configuration, "episodes": int(len(rows)), "arrival_rate": float(arrived.mean()),
                     "mean_length_of_arriving_episodes": float(rows[arrived, col["length"]].mean()) if arrived.any() else None,
                     "share_of_steps_above_1_0_m": float(rows[:, col["steps_above_1_0_m"]].sum() / steps),
                     "share_of_steps_above_1_5_m": float(rows[:, col["steps_above_1_5_m"]].sum() / steps),
                     "share_of_steps_without_contact": float(rows[:, col["steps_without_contact"]].sum() / steps),
                     "steps_in_wall_cell": float(rows[:, col["steps_in_wall_cell"]].sum()),
                     "mean_speed": float((rows[:, col["mean_speed"]] * rows[:, col["length"]]).sum() / steps),
                     "mean_height": float((rows[:, col["mean_height"]] * rows[:, col["length"]]).sum() / steps),
                     "largest_speed_median_over_episodes": float(np.median(rows[:, col["largest_speed"]])),
                     "largest_speed": float(rows[:, col["largest_speed"]].max()),
                     "largest_turning_median_over_episodes": float(np.median(rows[:, col["largest_turning"]])),
                     "largest_turning": float(rows[:, col["largest_turning"]].max()),
                     "episodes_beyond_guard": int((rows[:, col["physics_steps_beyond_guard"]] > 0).sum()),
                     "unstable_resets": int(rows[:, col["unstable_resets"]].sum()),
                     "energy_created_in_the_air_per_episode": float(rows[:, col["energy_created_in_the_air"]].mean()),
                     "motor_work_absolute_per_episode": float(rows[:, col["motor_work_absolute"]].mean()),
                     "largest_kinetic_energy_median": float(np.median(rows[:, col["largest_kinetic_energy"]]))}
            record["per_configuration"][setting].append(entry)
            print(f"   learning rate {configuration[0]:g}, weight {configuration[2]:g}: arriving {entry['arrival_rate']:.3f}, "
                  f"above 1.5 m {entry['share_of_steps_above_1_5_m']:.3f}, above 1.0 m {entry['share_of_steps_above_1_0_m']:.3f}, "
                  f"no contact {entry['share_of_steps_without_contact']:.3f}, mean speed {entry['mean_speed']:.2f}, largest speed "
                  f"{entry['largest_speed']:.1f}, largest turning {entry['largest_turning']:.0f}, beyond guard "
                  f"{entry['episodes_beyond_guard']} of {entry['episodes']}, unstable {entry['unstable_resets']}, energy created in "
                  f"the air {entry['energy_created_in_the_air_per_episode']:.1f} J per episode (motor work "
                  f"{entry['motor_work_absolute_per_episode']:.0f} J)", flush=True)
    record["seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(record, indent=1))
    np.savez(str(args.out) + "_episodes.npz", **arrays)
    print(f"wrote {args.out}.json and {args.out}_episodes.npz after {record['seconds']:.0f} s")


if __name__ == "__main__":
    main()
