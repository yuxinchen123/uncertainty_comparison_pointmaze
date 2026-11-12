"""Which ants touch the AntMaze ceiling, and how high do ordinary ants get? C MuJoCo, JAX and Warp on processors.

Writes one JSON record with four parts, all on the U-maze model with its ceiling, one physics step
(0.01 s) at a time for 10 seconds:

1. the nine ants of `tests/envs/test_antmaze_ceiling.py` (thrown upward, pushing with all four
   ankles, standing, walking, falling over, on the back) in C MuJoCo, in the JAX implementation
   and in the Warp implementation: largest torso height, physics steps with a ceiling contact,
   distance moved;
2. an ant thrown upward at 5 to 40 m/s in C MuJoCo: the largest torso height, which says how fast
   an ant must hit the ceiling for its torso centre to pass the wall tops;
3. 36 periodic torque patterns in C MuJoCo (three torque sizes, three frequencies, two leg
   orders, two ankle timings): distance moved, largest torso height, ceiling contacts, and whether
   the ant ends upright;
4. one physics step from C-MuJoCo states in ceiling contact, in JAX and in Warp, against C MuJoCo;
5. random torques in C MuJoCo (float64, the reference's own stepper) on the model with the ceiling
   and on the same model without it, with the same torques: whether ants rise above the walls and
   cross them in the reference physics too, and what the ceiling changes there.

Run on a compute node with the platform_jax_warp interpreter, never on a login node:
  <platform_jax_warp python> diagnose_ordinary_and_thrown_ants.py --out <file.json>
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests" / "envs"))


def xml_without_the_ceiling(xml: str) -> str:
    """The model as it was before the ceiling: the plane removed, the torso's collision bits back
    at ant.xml's default."""
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    worldbody.remove(worldbody.find("geom[@name='ceiling']"))
    del root.find(".//geom[@name='torso_geom']").attrib["contype"]
    return ET.tostring(root, encoding="unicode")


def c_random_torques(xml: str, cfg, n_ants: int, steps: int) -> dict:
    """`n_ants` ants in C MuJoCo from the spawn for `steps` environment steps of 5 physics steps,
    with random torques (standard normal, clipped to [-1, 1], fresh at every environment step: the
    torques of units/diagnose_arrivals_with_random_torques.py), no reset and no goal.

    Every ant has its own generator, keyed by its number (rng-seeding rule), so the two models get
    the same torques. Returns counts over all environment steps, looking at the torso after the
    last physics step of each.
    """
    import mujoco
    import numpy as np
    from exploration_platform.envs.antmaze.am_common import (CEILING_CONTACT_HEIGHT, CEILING_HEIGHT,
                                                             CEILING_ZONE_HEIGHT, SCALING, cell_center)
    from exploration_platform.envs.pointmaze.pm_common import MAPS
    model = mujoco.MjModel.from_xml_string(xml)
    maze = np.asarray(MAPS[cfg.map_name])
    rows, cols = maze.shape
    height, speed = np.zeros((n_ants, steps)), np.zeros((n_ants, steps))
    in_wall, outside = np.zeros((n_ants, steps), bool), np.zeros((n_ants, steps), bool)
    for ant in range(n_ants):
        # before: ant 3; after: a generator seeded with the low 32 bits of sha256("0::c random torques::3")
        seed = int(hashlib.sha256(f"0::c random torques::{ant}".encode()).hexdigest(), 16) & 0xFFFFFFFF
        generator = np.random.default_rng(seed)
        data = mujoco.MjData(model)
        data.qpos[0:2] = cell_center(cfg.start_cell, rows, cols)
        mujoco.mj_forward(model, data)
        for k in range(steps):
            data.ctrl[:] = np.clip(generator.standard_normal(model.nu), -1.0, 1.0)
            for _ in range(5):
                mujoco.mj_step(model, data)
            height[ant, k], speed[ant, k] = data.qpos[2], np.linalg.norm(data.qvel[0:3])
            # the maze cell under the torso: row 0 on top
            row, col = int(np.floor(rows / 2.0 - data.qpos[1] / SCALING)), int(np.floor(data.qpos[0] / SCALING + cols / 2.0))
            inside = 0 <= row < rows and 0 <= col < cols
            outside[ant, k] = not inside
            in_wall[ant, k] = inside and maze[row, col] == 1
    above = height > CEILING_HEIGHT
    return {"ants": n_ants, "environment_steps": int(n_ants * steps),
            "steps_with_the_torso_above_the_walls_top": int(above.sum()),
            "ants_with_the_torso_above_the_walls_top": int(above.any(1).sum()),
            "steps_with_the_torso_in_the_ceiling_zone": int((height > CEILING_ZONE_HEIGHT).sum()),
            "ants_with_the_torso_in_the_ceiling_zone": int((height > CEILING_ZONE_HEIGHT).any(1).sum()),
            "steps_with_the_torso_at_the_ceiling": int((height > CEILING_CONTACT_HEIGHT).sum()),
            "steps_with_the_torso_inside_a_wall_cell": int(in_wall.sum()),
            "ants_with_the_torso_inside_a_wall_cell": int(in_wall.any(1).sum()),
            "steps_with_the_torso_outside_the_maze": int(outside.sum()),
            "steps_faster_than_5_metres_per_second": int((speed > 5.0).sum()),
            "steps_faster_than_10_metres_per_second": int((speed > 10.0).sum()),
            "largest_torso_height": float(height.max()), "largest_torso_speed": float(speed.max()),
            "all_finite": bool(np.isfinite(height).all() and np.isfinite(speed).all())}


def main():
    """Measure the five parts and write the record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--c-ants", type=int, default=1024, help="ants of part 5")
    parser.add_argument("--c-steps", type=int, default=700, help="environment steps per ant of part 5")
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import test_antmaze_ceiling as ants
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml

    # GOAL: part 1, the nine ants in C MuJoCo and in both implementations.
    ref = ants.reference()
    qpos0, qvel0, torques = ref["inputs"]
    record = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
              "measured_iso": datetime.now().astimezone().isoformat(),
              "physics_steps": ants.PHYSICS_STEPS,
              "nine_ants": {"c_mujoco": ants.case_numbers(ref["position"], ref["ceiling_contact"])},
              "one_step_in_ceiling_contact": {}, "fault_record": {}}
    for implementation in ("jax", "warp"):
        run = ants.stepped(implementation)
        record["nine_ants"][implementation] = ants.case_numbers(run["position"], run["ceiling_contacts"])
        record["one_step_in_ceiling_contact"][implementation] = ants.one_step_in_ceiling_contact(implementation)
        record["fault_record"][implementation] = run["env"].fault_window(run["state"])[0]
        print(implementation, json.dumps(record["nine_ants"][implementation]), flush=True)

    # GOAL: part 2, the ant thrown upward at several speeds in C MuJoCo.
    record["thrown_upward_c_mujoco"] = []
    for speed in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0):
        launch = qvel0[0:1].copy()
        launch[0, 2] = speed
        thrown = ants.c_rollout(qpos0[0:1], launch, torques[:200, 0:1])
        record["thrown_upward_c_mujoco"].append({
            "speed": speed, "largest_torso_height": float(thrown["position"][:, 0, 2].max()),
            "physics_steps_with_a_ceiling_contact": int(thrown["ceiling_contact"].sum())})

    # GOAL: part 3, the periodic torque patterns in C MuJoCo.
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(ants.CFG))
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    # the ant's mass in kg and its motors' torque at full control, in N m
    record["ant_mass"] = float(model.body_mass.sum())
    record["motor_torque_at_full_control"] = float(np.abs(model.actuator_gear[:, 0]).max())
    orders = (("diagonal leg pairs in turn", (np.pi, 0.0, np.pi, 0.0)),
              ("one leg after the other", (1.5 * np.pi, 0.0, 0.5 * np.pi, np.pi)))
    lags = (("a quarter period behind its hip", -0.5 * np.pi), ("a quarter period ahead of its hip", 0.5 * np.pi))
    record["periodic_patterns_c_mujoco"] = []
    for amplitude in (0.25, 0.5, 1.0):
        for hertz in (0.5, 1.0, 2.0):
            for order, phase in orders:
                for timing, lag in lags:
                    pattern = ants.walking_torques(amplitude, hertz, phase, lag)[:, None]
                    out = ants.c_rollout(qpos0[4:5], qvel0[4:5], pattern)
                    # how upright the ant ends: the vertical part of the torso's own z axis
                    data = mujoco.MjData(model)
                    data.qpos[:] = out["qpos"][-1, 0]
                    mujoco.mj_forward(model, data)
                    record["periodic_patterns_c_mujoco"].append({
                        "torque": amplitude, "hertz": hertz, "leg_order": order, "ankle": timing,
                        "moved_on_the_floor_plane": float(np.linalg.norm(
                            out["position"][-1, 0, 0:2] - out["position"][0, 0, 0:2])),
                        "largest_torso_height": float(out["position"][:, 0, 2].max()),
                        "physics_steps_with_a_ceiling_contact": int(out["ceiling_contact"].sum()),
                        "torso_z_axis_vertical_part_at_the_end": float(data.xmat[torso].reshape(3, 3)[2, 2])})

    # GOAL: part 5, random torques in C MuJoCo with and without the ceiling (the five-step
    # environment step of the study's configuration, the same torques for both models).
    from exploration_platform.envs.antmaze.am_common import preset
    xml = build_antmaze_xml(preset("umaze"))
    record["random_torques_c_mujoco"] = {
        "with_the_ceiling": c_random_torques(xml, preset("umaze"), args.c_ants, args.c_steps),
        "without_the_ceiling": c_random_torques(xml_without_the_ceiling(xml), preset("umaze"),
                                                args.c_ants, args.c_steps)}
    print("random torques in C MuJoCo:", json.dumps(record["random_torques_c_mujoco"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
