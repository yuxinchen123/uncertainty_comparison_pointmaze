"""Read the trajectories recorded in the Warp AntMaze (`roll_out_and_record.py`) with C MuJoCo's own tools.

Three measurements, all on processors:

1. **What touches the ant.** For every recorded state, C MuJoCo's collision detection on the recorded
   positions: no contact at all, the floor, a wall, the ceiling; per configuration.
2. **What precedes a jump of the torso's speed.** A jump is a rise of the torso speed by more than
   `--jump` metres per second within one environment step (0.05 s) inside an episode. For each jump: what
   touched the ant in the state before it, the torso's height and turning speed and the largest joint speed
   there.
3. **Open loop.** The recorded actions of each recorded environment's first episode are replayed from the
   spawn in C MuJoCo with our settings, with our settings from a start moved by one micrometre, and with
   the reference settings, and with our settings and the state rounded to single precision after every
   environment step (the precision Warp computes in). The record says after how many environment steps the
   torso is more than 0.1 m away from where it was in Warp, and after how many steps C MuJoCo is that far
   from its own copies (two runs of a chaotic system part anyway; only a much earlier parting than that
   says the physics differ).

Run on a compute node's processors, never on a login node:
  <platform_jax_warp python> c_mujoco_from_warp_records.py --name <records/warp_rollout_...> --out <file>
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

OURS = ("implicitfast", 4, 8, 0.005, 10)
REFERENCE = ("RK4", 100, 50, 0.01, 5)
CACHE = {}


def model_of(setting):
    """The environment's C MuJoCo model with (integrator, solver, line search, time step, physics steps)."""
    import mujoco
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset
    if setting not in CACHE:
        integrator, iterations, ls_iterations, timestep, substeps = setting
        cfg = replace(preset("umaze"), continuing_task=False, integrator=integrator, solver_iterations=iterations,
                      ls_iterations=ls_iterations, physics_time_step=timestep, frame_skip=substeps)
        model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        CACHE[setting] = model
    return CACHE[setting]


def touches(task):
    """Per recorded state of some environments: [steps, environments, 5] = (contacts, floor, wall, ceiling,
    kinetic energy), from C MuJoCo's forward pass on the recorded positions and velocities."""
    import mujoco
    import numpy as np
    states = task
    model = model_of(OURS)
    data = mujoco.MjData(model)
    kind = np.zeros(model.ngeom, np.int64)   # 1 floor, 2 wall, 3 ceiling, 0 a geom of the ant
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        kind[geom] = 1 if name == "floor" else 2 if name.startswith("block_") else 3 if name == "ceiling" else 0
    out = np.zeros(states.shape[:2] + (5,), np.float32)
    for t in range(states.shape[0]):
        for e in range(states.shape[1]):
            data.qpos[:], data.qvel[:] = states[t, e, :15], states[t, e, 15:]
            mujoco.mj_forward(model, data)
            kinds = set()
            for contact in data.contact[:data.ncon]:
                kinds.add(int(max(kind[contact.geom1], kind[contact.geom2])))
            out[t, e] = (data.ncon, 1 in kinds, 2 in kinds, 3 in kinds, data.energy[1])
    return out


def replay(task):
    """One environment's first episode in C MuJoCo under the recorded actions, three ways.

    Returns per environment step the torso's distance from the Warp recording for our settings and for the
    reference settings, the distance between our settings and their copy started one micrometre higher, between
    our settings and their copy whose state is rounded to single precision after every environment step, and
    between our settings and the reference settings.
    """
    import mujoco
    import numpy as np
    states, actions = task          # [L + 1, 29] recorded states (the last one may be the reset spawn), [L, 8]
    steps = len(actions)
    paths = {}
    for name, setting, lift, single in (("ours", OURS, 0.0, False), ("ours_moved", OURS, 1e-6, False),
                                        ("reference", REFERENCE, 0.0, False), ("ours_single", OURS, 0.0, True)):
        model = model_of(setting)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = states[0, :15], states[0, 15:]
        data.qpos[2] += lift
        path = np.zeros((steps, 3))
        for t in range(steps):
            data.ctrl[:] = np.clip(actions[t], -1.0, 1.0)
            for _ in range(setting[4]):
                mujoco.mj_step(model, data)
            if single:
                # the state as single precision would hold it: before 0.123456789012 after 0.12345679104
                data.qpos[:] = data.qpos.astype(np.float32)
                data.qvel[:] = data.qvel.astype(np.float32)
            path[t] = data.qpos[0:3]
        paths[name] = path
    recorded = states[1:steps + 1, 0:3].astype(np.float64)
    usable = steps - 1              # the state after the last step is the reset spawn, not the step's end
    gap = lambda a, b: np.linalg.norm(a[:usable] - b[:usable], axis=1)
    return (gap(paths["ours"], recorded), gap(paths["reference"], recorded), gap(paths["ours"], paths["ours_moved"]),
            gap(paths["ours"], paths["ours_single"]), gap(paths["ours"], paths["reference"]))


def first_beyond(gaps, bound):
    """The first environment step at which a distance passes a bound; the series' length if it never does.

    before: [0.0, 0.01, 0.2, 0.5], bound 0.1   after: 2
    """
    import numpy as np
    over = np.nonzero(gaps > bound)[0]
    return int(over[0]) if len(over) else len(gaps)


def main():
    """The three measurements; one record file and printed tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=Path, required=True, help="path of the rollout's files without the suffix")
    parser.add_argument("--jump", type=float, default=4.0, help="metres per second within one environment step")
    parser.add_argument("--replay-environments", type=int, default=2, help="recorded environments per policy to replay")
    parser.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    started = time.time()
    recorded = np.load(str(args.name) + "_trajectories.npz")
    actors = np.load(str(args.name) + "_actors.npz")
    states, actions = recorded["state"], recorded["action"]              # [T, E, 29], [T, E, 8]
    done = recorded["terminated"] | recorded["truncated"]                # [T, E]: the step ended its episode
    T, E = done.shape
    configurations, per_cell = actors["configurations"].tolist(), int(actors["policies_per_configuration"])
    per_policy = E // actors["W0"].shape[0]
    cell_of = (np.arange(E) // per_policy) // per_cell
    record = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
              "measured_iso": datetime.now().astimezone().isoformat(), "rollout": args.name.name,
              "recorded_steps": int(T), "recorded_environments": int(E), "configurations": configurations,
              "jump_metres_per_second_within_one_step": args.jump}

    # GOAL 1: what touches the ant in every recorded state.
    with Pool(args.processes) as pool:
        parts = pool.map(touches, [states[:, chunk] for chunk in np.array_split(np.arange(E), args.processes * 4)])
    touch = np.concatenate(parts, axis=1)                                # [T, E, 5]
    none, floor, wall, ceiling = touch[..., 0] == 0, touch[..., 1] > 0, touch[..., 2] > 0, touch[..., 3] > 0
    height = states[..., 2]
    speed = np.linalg.norm(states[..., 15:18], axis=-1)
    turning = np.linalg.norm(states[..., 18:21], axis=-1)
    joint = np.abs(states[..., 21:29]).max(-1)
    record["touch_per_configuration"] = []
    print("== what touches the ant (share of recorded states)")
    for cell, configuration in enumerate(configurations):
        pick = cell_of == cell
        entry = {"configuration": configuration, "states": int(pick.sum() * T),
                 "no_contact": float(none[:, pick].mean()), "floor": float(floor[:, pick].mean()),
                 "wall": float(wall[:, pick].mean()), "ceiling": float(ceiling[:, pick].mean()),
                 "no_contact_and_above_1_0_m": float((none & (height > 1.0))[:, pick].mean()),
                 "median_height": float(np.median(height[:, pick])), "median_speed": float(np.median(speed[:, pick])),
                 "speed_quantiles_50_90_99_999": np.quantile(speed[:, pick], [0.5, 0.9, 0.99, 0.999]).round(2).tolist(),
                 "height_quantiles_50_90_99_999": np.quantile(height[:, pick], [0.5, 0.9, 0.99, 0.999]).round(2).tolist(),
                 "turning_quantiles_50_90_99_999": np.quantile(turning[:, pick], [0.5, 0.9, 0.99, 0.999]).round(1).tolist(),
                 "median_kinetic_energy": float(np.median(touch[:, pick, 4])),
                 "kinetic_energy_quantile_99": float(np.quantile(touch[:, pick, 4], 0.99))}
        record["touch_per_configuration"].append(entry)
        print(f"   learning rate {configuration[0]:g}, weight {configuration[2]:g}: no contact {entry['no_contact']:.3f}, floor "
              f"{entry['floor']:.3f}, wall {entry['wall']:.3f}, ceiling {entry['ceiling']:.3f}; height quantiles "
              f"{entry['height_quantiles_50_90_99_999']}, speed {entry['speed_quantiles_50_90_99_999']}, turning "
              f"{entry['turning_quantiles_50_90_99_999']}", flush=True)

    # GOAL 2: what precedes a jump of the torso's speed (both states in the same episode).
    rise = speed[1:] - speed[:-1]
    jumps = (rise > args.jump) & ~done[:-1]
    before = lambda array: array[:-1][jumps]
    count = int(jumps.sum())
    record["speed_jumps"] = {
        "count": count, "per_million_recorded_steps": count / (jumps.size / 1e6),
        "no_contact_before": float(before(none).mean()) if count else None,
        "floor_before": float(before(floor).mean()) if count else None,
        "wall_before": float(before(wall).mean()) if count else None,
        "ceiling_before": float(before(ceiling).mean()) if count else None,
        "median_height_before": float(np.median(before(height))) if count else None,
        "median_turning_before": float(np.median(before(turning))) if count else None,
        "median_largest_joint_speed_before": float(np.median(before(joint))) if count else None,
        "median_speed_before_and_after": [float(np.median(before(speed))), float(np.median(speed[1:][jumps]))] if count else None,
        "for_comparison_all_states": {"no_contact": float(none.mean()), "floor": float(floor.mean()), "wall": float(wall.mean()),
                                      "ceiling": float(ceiling.mean()), "median_turning": float(np.median(turning)),
                                      "median_largest_joint_speed": float(np.median(joint))}}
    print("== speed jumps:", json.dumps(record["speed_jumps"]), flush=True)
    # the fastest recorded states: what touched the ant one and two steps earlier
    fast = (speed[2:] > 10.0) & ~done[1:-1] & ~done[:-2]
    record["states_above_10_metres_per_second"] = {
        "count": int(fast.sum()), "share_of_recorded_states": float(fast.mean()),
        "no_contact_one_step_before": float(none[1:-1][fast].mean()) if fast.any() else None,
        "wall_or_ceiling_one_or_two_steps_before": float(((wall | ceiling)[1:-1][fast] | (wall | ceiling)[:-2][fast]).mean()) if fast.any() else None,
        "median_turning_two_steps_before": float(np.median(turning[:-2][fast])) if fast.any() else None,
        "median_height": float(np.median(height[2:][fast])) if fast.any() else None}
    print("== states above 10 m/s:", json.dumps(record["states_above_10_metres_per_second"]), flush=True)

    # GOAL 3: open loop, the first episode of some recorded environments.
    tasks, cells = [], []
    for e in range(E):
        if e % per_policy >= args.replay_environments:
            continue
        ends = np.nonzero(done[:, e])[0]
        length = int(ends[0]) + 1 if len(ends) else T - 1
        tasks.append((states[:length + 1, e].astype(np.float64), actions[:length, e].astype(np.float64)))
        cells.append(int(cell_of[e]))
    with Pool(args.processes) as pool:
        gaps = pool.map(replay, tasks, chunksize=4)
    parted = {"warp_against_c_mujoco_ours": [first_beyond(g[0], 0.1) for g in gaps],
              "warp_against_c_mujoco_reference": [first_beyond(g[1], 0.1) for g in gaps],
              "c_mujoco_ours_against_its_copy_moved_by_one_micrometre": [first_beyond(g[2], 0.1) for g in gaps],
              "c_mujoco_ours_against_its_copy_rounded_to_single_precision": [first_beyond(g[3], 0.1) for g in gaps],
              "c_mujoco_ours_against_c_mujoco_reference": [first_beyond(g[4], 0.1) for g in gaps]}
    record["open_loop"] = {"episodes_replayed": len(tasks), "bound_metres": 0.1,
                           "first_step_beyond_the_bound_quantiles_10_50_90": {
                               name: np.quantile(values, [0.1, 0.5, 0.9]).round(0).tolist() for name, values in parted.items()},
                           "gap_after_5_10_20_steps_median_metres": {
                               name: [float(np.median([g[i][k] for g in gaps if len(g[i]) > k])) for k in (5, 10, 20)]
                               for i, name in enumerate(("warp_against_c_mujoco_ours", "warp_against_c_mujoco_reference",
                                                         "c_mujoco_ours_against_its_copy_moved_by_one_micrometre",
                                                         "c_mujoco_ours_against_its_copy_rounded_to_single_precision",
                                                         "c_mujoco_ours_against_c_mujoco_reference"))}}
    print("== open loop:", json.dumps(record["open_loop"]), flush=True)
    record["seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out} after {record['seconds']:.0f} s")


if __name__ == "__main__":
    main()
