"""Environment gates of one AntMaze map (Medium or Large) in MuJoCo Warp, against C MuJoCo, on processors.

The environment is the one the map's recorded configuration file describes (MuJoCo Warp, implicitfast,
4 solver and 8 line-search iterations, 0.005 s with 10 physics steps per environment step, the fault
guard at 30 m/s and 100 rad/s, the episode ending at the goal). Every gate measures numbers, compares
them with a bound, and the script writes all of them as one JSON record:

  file        the file's map values equal the environment's preset; the model's wall boxes cover
              exactly the wall cells of the map
  reset       every environment starts at the start cell's centre, 0.75 m high, at rest
  goal        one step from torsos placed around the goal: termination exactly when the torso ends
              within 0.45 m, reward 1 exactly then, the next observation the spawn
  step_limit  1,000 steps from the spawn (no torque, and random torques): one truncation at step 1,000
  walls       an ant thrown at 8 m/s toward every wall face next to an open cell: it touches the wall,
              no body origin or geom centre enters a wall box, the torso stays in open cells, in Warp
              and in C MuJoCo; one physics step from C states in wall contact agrees with C
  ceiling     ants thrown upward at 10, 20 and 30 m/s at the spawn, the goal and the open cells nearest
              the four corners touch the ceiling and stay below the wall tops (2 m), in Warp and C
  parity      the speed study's gates against C MuJoCo at this configuration: one environment step from
              50 states of a random-torque trajectory (1e-4 m contact-free, 5e-2 m in contact), 20
              shared-torque steps (1e-4 m before the first contact, under one cell over all), and the
              resting height after 1,200 passive steps
  guard       the six forced ants of tests/envs/test_antmaze_fault_guard.py (one 2 m outside this
              map's outer wall): exactly the four broken ones are respawned

Run on a compute node with the platform_jax_warp interpreter, JAX_PLATFORMS=cpu:
  python map_gates.py --map medium --out records/gates_medium.json
"""
import argparse
import json
import sys
import time
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

PLATFORM = Path("/p/rlprojects/RND/10_jax_exploration_platform")
for folder in ("src", "scripts", "benchmarks/env", "tests/envs"):
    sys.path.insert(0, str(PLATFORM / folder))
import exploration_platform  # noqa: E402,F401  (64-bit statistics on, as in training)
import antmaze_warp_checks as checks  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from experiment_configuration import read_experiment_configuration  # noqa: E402
from exploration_platform.envs.antmaze import faults  # noqa: E402
from exploration_platform.envs.antmaze.am_common import (CEILING_CONTACT_HEIGHT, CEILING_HEIGHT,  # noqa: E402
                                                         CEILING_ZONE_HEIGHT, HEIGHT, SCALING, AntMazeConfig,
                                                         build_antmaze_xml, cell_center, maze_half_extent,
                                                         merged_wall_boxes, preset)
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import MAPS  # noqa: E402
from test_antmaze_fault_guard import ANTS, BROKEN, forced_state  # noqa: E402

TAIL = ("warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_"
        "cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml")
F32 = jnp.float32
DIRECTIONS = {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}   # (row, column) steps
THROW_SPEED = 8.0          # metres per second, toward the wall
WALL_STEPS = 300           # physics steps of 0.005 s: 1.5 s
CEILING_STEPS = 400        # 2 s
LAUNCH_SPEEDS = (10.0, 20.0, 30.0)


def configuration(map_name):
    """The map's environment as its recorded configuration file gives it, and the file's path."""
    path = PLATFORM / "configs/experiments" / f"antmaze_{map_name}_{TAIL}"
    return AntMazeConfig(**read_experiment_configuration(path)["environment"]), path


def wall_depth(points, boxes):
    """How deep each point lies inside the deepest wall box, in metres; negative when outside all.

    points: [..., 3] world positions; boxes: (x, y, half_x, half_y) of the merged wall boxes, 0 to 2 m high.
    before: one point (-12.1, -10.0, 0.5) and one box (-14, 0, 2, 16) (the medium map's left wall)
    after:  0.1 (it is 0.1 m inside the box through its right face)
    """
    box = np.asarray(boxes, np.float64)                      # [K, 4]
    p = np.asarray(points, np.float64)[..., None, :]         # [..., 1, 3]
    # the four distances to the box's faces broadcast to [..., K]: x faces, y faces, top, floor
    across = np.minimum(box[:, 2] - np.abs(p[..., 0] - box[:, 0]), box[:, 3] - np.abs(p[..., 1] - box[:, 1]))
    depth = np.minimum(across, np.minimum(HEIGHT * SCALING - p[..., 2], p[..., 2]))
    return depth.max(-1)


def warp_contacts_with(env, data, geom_ids):
    """[B] int32: each environment's contacts with any geom of `geom_ids` in the shared pool after its last step."""
    impl = data._impl
    slot = jnp.arange(impl.contact__dist.shape[0])
    touches = jnp.isin(impl.contact__geom, jnp.asarray(geom_ids)).any(-1)
    live = (slot < impl.nacon.max()) & touches & (impl.contact__dist < impl.contact__includemargin)
    return jnp.zeros((env.B,), jnp.int32).at[impl.contact__worldid].add(live.astype(jnp.int32))


def warp_physics_rollout(env, qpos0, qvel0, steps, geom_ids):
    """Ants in Warp, one physics step per environment step, no torque.

    Returns per step the body origins [steps, B, nbody, 3], the geom centres [steps, B, ngeom, 3], the
    contacts with `geom_ids` [steps, B], and the final state.
    """
    def body(state, _):
        """One environment step of all ants with zero torques."""
        state, *_ = env.step(state, jnp.zeros((env.C, env.N, 8), F32))
        return state, (state.data.xpos, state.data.geom_xpos, warp_contacts_with(env, state.data, geom_ids))
    start = checks.with_state(env, env.reset(), qpos0, qvel0)
    state, (xpos, geom_xpos, contacts) = jax.jit(lambda s: jax.lax.scan(body, s, None, length=steps))(start)
    return np.asarray(xpos, np.float64), np.asarray(geom_xpos, np.float64), np.asarray(contacts), state


def c_physics_rollout(model, qpos0, qvel0, steps, geom_ids):
    """The same ants in C MuJoCo (double precision): body origins, geom centres, contacts with `geom_ids`,
    which of `geom_ids` were touched, and every state (qpos, qvel include the start state)."""
    n = len(qpos0)
    out = {"xpos": np.zeros((steps, n, model.nbody, 3)), "geom_xpos": np.zeros((steps, n, model.ngeom, 3)),
           "contacts": np.zeros((steps, n), np.int64),
           "qpos": np.zeros((steps + 1, n, model.nq)), "qvel": np.zeros((steps + 1, n, model.nv)), "touched": set()}
    wanted = np.asarray(geom_ids)
    for b in range(n):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = qpos0[b], qvel0[b]
        mujoco.mj_forward(model, data)
        out["qpos"][0, b], out["qvel"][0, b] = data.qpos, data.qvel
        for k in range(steps):
            mujoco.mj_step(model, data)
            out["xpos"][k, b], out["geom_xpos"][k, b] = data.xpos, data.geom_xpos
            pairs = data.contact.geom[:data.ncon]
            out["contacts"][k, b] = int(np.isin(pairs, wanted).any(-1).sum())
            out["touched"].update(int(g) for g in pairs.reshape(-1) if g in wanted)
            out["qpos"][k + 1, b], out["qvel"][k + 1, b] = data.qpos, data.qvel
    return out


def gate_file(cfg, path):
    """The file's map values are the preset's; the wall boxes cover exactly the wall cells."""
    reference = preset(cfg.map_name)
    map_fields = ("map_name", "start_cell", "goal_cell", "max_episode_steps", "goal_radius")
    maze = np.asarray(MAPS[cfg.map_name])
    rows, cols = maze.shape
    boxes = merged_wall_boxes(cfg.map_name)
    # GOAL: a 7 x 7 grid of points inside every cell, 1 cm from its edges, 1 m high: inside a box exactly
    # when the cell is a wall cell
    offsets = np.linspace(-SCALING / 2 + 0.01, SCALING / 2 - 0.01, 7)
    wrong = []
    for i in range(rows):
        for j in range(cols):
            x, y = cell_center((i, j), rows, cols)
            points = np.asarray([(x + a, y + b, 1.0) for a in offsets for b in offsets])
            inside = wall_depth(points, boxes) > 0
            if inside.all() != bool(maze[i, j]) or inside.any() != bool(maze[i, j]):
                wrong.append([i, j])
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    result = {"file": str(path), "map_values": {name: getattr(cfg, name) for name in map_fields},
              "map_values_equal_the_preset": all(getattr(cfg, n) == getattr(reference, n) for n in map_fields),
              "other_fields_differing_from_the_preset": sorted(
                  f.name for f in fields(cfg) if f.name not in map_fields and getattr(cfg, f.name) != getattr(reference, f.name)),
              "rows_columns": [rows, cols], "wall_cells": int(maze.sum()), "open_cells": int((maze == 0).sum()),
              "wall_boxes": len(boxes), "model_wall_geoms": sum(name.startswith("block_") for name in names),
              "cells_whose_points_disagree_with_the_boxes": wrong,
              "start_and_goal_open": bool(maze[cfg.start_cell] == 0 and maze[cfg.goal_cell] == 0),
              "start_xy": cell_center(cfg.start_cell, rows, cols), "goal_xy": cell_center(cfg.goal_cell, rows, cols),
              "half_extent": maze_half_extent(cfg.map_name), "model_geoms": model.ngeom,
              "ceiling_geoms": names.count("ceiling")}
    result["passed"] = (result["map_values_equal_the_preset"] and not wrong and result["start_and_goal_open"]
                        and result["wall_boxes"] == result["model_wall_geoms"] and result["ceiling_geoms"] == 1)
    return result


def gate_reset(cfg):
    """Every environment of a reset batch is at the start cell's centre, 0.75 m high, at rest."""
    env = WarpAntMaze(cfg, 2, 3)
    state = env.reset()
    start = np.float32(cell_center(cfg.start_cell, env.rows, env.cols))
    qpos, qvel = np.asarray(state.data.qpos), np.asarray(state.data.qvel)
    _, obs = env.respawn(state)
    obs = np.asarray(obs)
    result = {"start_xy": start.tolist(), "largest_xy_difference": float(np.abs(qpos[:, 0:2] - start).max()),
              "torso_heights": sorted(set(qpos[:, 2].tolist())), "largest_speed": float(np.abs(qvel).max()),
              "observation_shape": list(obs.shape),
              "observation_xy_equals_start": bool((obs[..., 0:2] == start).all())}
    result["passed"] = (result["largest_xy_difference"] == 0.0 and result["torso_heights"] == [0.75]
                        and result["largest_speed"] == 0.0 and result["observation_xy_equals_start"])
    return result


def gate_goal(cfg):
    """One step with no torque from torsos placed at 0 to 1 m from the goal in four directions."""
    distances, angles = (0.0, 0.2, 0.40, 0.44, 0.46, 0.50, 1.0), (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
    placed = [(d, a) for d in distances for a in angles]
    env = WarpAntMaze(cfg, len(placed), 1)
    goal = np.asarray(cell_center(cfg.goal_cell, env.rows, env.cols))
    # before: the spawn pose for every ant; after: torso xy = goal + d (cos a, sin a)
    qpos = np.tile(np.asarray(env.init_qpos), (len(placed), 1))
    qpos[:, 0:2] = goal + np.asarray([(d * np.cos(a), d * np.sin(a)) for d, a in placed])
    start = checks.with_state(env, env.reset(), qpos, np.zeros((len(placed), 14), np.float32))
    state, obs, reward, terminated, truncated, final_obs = jax.jit(env.step)(start, jnp.zeros((len(placed), 1, 8), F32))
    final_obs, obs = np.asarray(final_obs)[:, 0], np.asarray(obs)[:, 0]
    reward, terminated, truncated = (np.asarray(x)[:, 0] for x in (reward, terminated, truncated))
    after = np.sqrt(((final_obs[:, 0:2] - goal) ** 2).sum(-1))
    spawn = np.concatenate([np.asarray(env.init_qpos), np.zeros(14, np.float32)])
    within = after <= cfg.goal_radius
    result = {"goal_xy": goal.tolist(), "placed_distances": [d for d, _ in placed],
              "distance_after_the_step": after.round(4).tolist(),
              "largest_move_during_the_step": float(np.abs(after - np.asarray([d for d, _ in placed])).max()),
              "terminated": terminated.tolist(), "reward": reward.tolist(),
              "terminated_exactly_within_the_radius": bool((terminated == within).all()),
              "reward_one_exactly_when_terminated": bool((reward == terminated.astype(np.float32)).all()),
              "no_truncation": bool(not truncated.any()),
              "terminated_restart_at_the_spawn": bool((obs[terminated] == spawn[None]).all()),
              "others_continue": bool(np.array_equal(obs[~terminated], final_obs[~terminated])),
              "episodes_counted": np.asarray(state.reset_count)[:, 0].tolist()}
    result["passed"] = all(result[k] for k in ("terminated_exactly_within_the_radius", "reward_one_exactly_when_terminated",
                                               "no_truncation", "terminated_restart_at_the_spawn", "others_continue"))
    result["passed"] = result["passed"] and int(within.sum()) == 16 and int((~within).sum()) == 12
    return result


def gate_step_limit(cfg):
    """1,000 steps from the spawn for two ants (no torque, keyed random torques): one truncation at step 1,000."""
    env = WarpAntMaze(cfg, 2, 1)
    limit = cfg.max_episode_steps
    torques = np.array(checks.torques("medium-large readiness step limit", limit, 2))
    torques[:, 0] = 0.0
    state, (obs, reward, terminated, truncated, final_obs) = checks.rollout_call(env)(env.reset(), jnp.asarray(torques))
    truncated, terminated = np.asarray(truncated)[:, :, 0], np.asarray(terminated)[:, :, 0]
    spawn = np.concatenate([np.asarray(env.init_qpos), np.zeros(14, np.float32)])
    result = {"step_limit": limit, "steps_with_a_truncation": [np.flatnonzero(truncated[:, c]).tolist() for c in range(2)],
              "terminations": int(terminated.sum()), "respawns_for_a_broken_state": np.asarray(state.nan_count)[:, 0].tolist(),
              "episodes_counted": np.asarray(state.reset_count)[:, 0].tolist(),
              "observation_after_the_limit_is_the_spawn": bool(np.array_equal(np.asarray(obs)[-1, :, 0], np.tile(spawn, (2, 1)))),
              "largest_distance_from_the_start": float(np.abs(np.asarray(final_obs)[:, :, 0, 0:2] - spawn[0:2]).max()),
              "pool": env.overflow_report(state)}
    result["passed"] = (result["steps_with_a_truncation"] == [[limit - 1], [limit - 1]] and result["terminations"] == 0
                        and result["respawns_for_a_broken_state"] == [0, 0] and result["episodes_counted"] == [1, 1]
                        and result["observation_after_the_limit_is_the_spawn"] and not result["pool"]["dropped"])
    return result


def gate_walls(cfg):
    """Ants thrown at every wall face next to an open cell, in Warp and in C MuJoCo, one physics step per step."""
    physics = replace(cfg, frame_skip=1, max_episode_steps=10 ** 6, continuing_task=True)
    maze = np.asarray(MAPS[cfg.map_name])
    rows, cols = maze.shape
    boxes = merged_wall_boxes(cfg.map_name)
    # GOAL: one ant per (open cell, wall neighbour): at the cell's centre, torso moving toward that wall.
    cases = [((i, j), name) for i in range(rows) for j in range(cols) if maze[i, j] == 0
             for name, (di, dj) in DIRECTIONS.items() if maze[i + di, j + dj] == 1]
    env = WarpAntMaze(physics, len(cases), 1)
    model = env.mj_model
    walls = [g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("block_")]
    ant_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0]
    qpos0 = np.tile(np.asarray(env.init_qpos, np.float64), (len(cases), 1))
    qvel0 = np.zeros((len(cases), 14))
    for b, ((i, j), name) in enumerate(cases):
        di, dj = DIRECTIONS[name]
        qpos0[b, 0:2] = cell_center((i, j), rows, cols)
        # a step down one row is a step down in y (row 0 is at the top)
        qvel0[b, 0:2] = THROW_SPEED * np.asarray([dj, -di])
    warp_xpos, warp_geom_xpos, warp_contacts, state = warp_physics_rollout(env, qpos0, qvel0, WALL_STEPS, walls)
    ref = c_physics_rollout(model, qpos0, qvel0, WALL_STEPS, walls)

    def summary(xpos, geom_xpos, contacts):
        """Per case: wall contact seen, deepest body origin and geom centre inside a wall, torso always in open cells."""
        torso = xpos[:, :, 1]
        cell_i = np.clip((rows / 2.0 - torso[..., 1] / SCALING).astype(int), 0, rows - 1)
        cell_j = np.clip((torso[..., 0] / SCALING + cols / 2.0).astype(int), 0, cols - 1)
        return {"cases_with_a_wall_contact": int((contacts > 0).any(0).sum()),
                "deepest_body_origin_in_a_wall_m": float(wall_depth(xpos[:, :, 1:], boxes).max()),
                "deepest_ant_geom_centre_in_a_wall_m": float(wall_depth(geom_xpos[:, :, ant_geoms], boxes).max()),
                "torso_centre_closest_to_a_wall_face_m": float(-wall_depth(torso, boxes).max()),
                "torso_steps_in_a_wall_cell": int(maze[cell_i, cell_j].sum()),
                "largest_torso_height_m": float(torso[..., 2].max())}

    # GOAL: one physics step in Warp from C states in wall contact (the state before each case's first wall
    # contact and the three after it) against C's next state.
    picks = []
    for b in range(len(cases)):
        touching = np.flatnonzero(ref["contacts"][:, b] > 0)
        if touching.size:
            picks += [(b, k) for k in range(max(int(touching[0]) - 1, 0), int(touching[0]) + 3)]
    parity_env = WarpAntMaze(physics, len(picks), 1)
    picked_q = np.asarray([ref["qpos"][k, b] for b, k in picks])
    picked_v = np.asarray([ref["qvel"][k, b] for b, k in picks])
    stepped, *_ = jax.jit(parity_env.step)(checks.with_state(parity_env, parity_env.reset(), picked_q, picked_v),
                                           jnp.zeros((len(picks), 1, 8), F32))
    difference = np.abs(np.asarray(stepped.data.qpos, np.float64) - np.asarray([ref["qpos"][k + 1, b] for b, k in picks])).max(-1)
    result = {"cases": len(cases), "throw_speed_m_per_s": THROW_SPEED, "physics_steps": WALL_STEPS,
              "wall_boxes": len(walls), "warp": summary(warp_xpos, warp_geom_xpos, warp_contacts),
              "c_mujoco": summary(ref["xpos"], ref["geom_xpos"], ref["contacts"]),
              "warp_respawns": int(np.asarray(state.nan_count).sum()), "warp_pool": env.overflow_report(state),
              "one_physics_step_from_c_wall_contact_states": {
                  "states": len(picks), "largest_m": float(difference.max()), "median_m": float(np.median(difference))},
              "wall_boxes_touched_in_c_mujoco": len(ref["touched"])}
    passed = True
    for name in ("warp", "c_mujoco"):
        s = result[name]
        passed &= (s["cases_with_a_wall_contact"] == len(cases) and s["deepest_body_origin_in_a_wall_m"] < 0.0
                   and s["deepest_ant_geom_centre_in_a_wall_m"] < 0.0 and s["torso_steps_in_a_wall_cell"] == 0
                   and s["torso_centre_closest_to_a_wall_face_m"] > 0.2)
    result["passed"] = bool(passed and result["warp_respawns"] == 0 and not result["warp_pool"]["dropped"]
                            and result["wall_boxes_touched_in_c_mujoco"] == len(walls)
                            and result["one_physics_step_from_c_wall_contact_states"]["largest_m"] < 5e-2)
    return result


def gate_ceiling(cfg):
    """Ants thrown upward at the spawn, the goal and the open cells nearest the four corners, and one standing ant.

    The fault guard is off here, as in the U-maze's ceiling test: with it on, the guard respawned every ant
    thrown at 30 m/s (its speed bound) at the first environment step (2026-11-13 11:18 PT,
    gates_<map>_physics.json, the run with the guard on), while C MuJoCo has no guard.
    """
    physics = replace(cfg, frame_skip=1, max_episode_steps=10 ** 6, continuing_task=True,
                      fault_torso_speed=0.0, fault_torso_angular_speed=0.0)
    maze = np.asarray(MAPS[cfg.map_name])
    rows, cols = maze.shape
    opens = np.argwhere(maze == 0)
    corners = [tuple(int(v) for v in opens[np.abs(opens - np.asarray(c)).sum(-1).argmin()])
               for c in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1))]
    cells = list(dict.fromkeys([tuple(cfg.start_cell), tuple(cfg.goal_cell), *corners]))
    throws = [(cell, speed) for cell in cells for speed in LAUNCH_SPEEDS] + [(tuple(cfg.start_cell), 0.0)]
    env = WarpAntMaze(physics, len(throws), 1)
    model = env.mj_model
    ceiling = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")]
    qpos0 = np.tile(np.asarray(env.init_qpos, np.float64), (len(throws), 1))
    qvel0 = np.zeros((len(throws), 14))
    for b, (cell, speed) in enumerate(throws):
        qpos0[b, 0:2], qvel0[b, 2] = cell_center(cell, rows, cols), speed
    warp_xpos, _, warp_contacts, state = warp_physics_rollout(env, qpos0, qvel0, CEILING_STEPS, ceiling)
    ref = c_physics_rollout(model, qpos0, qvel0, CEILING_STEPS, ceiling)
    rows_out = []
    for b, (cell, speed) in enumerate(throws):
        rows_out.append({"cell": list(cell), "launch_speed": speed,
                         "warp_largest_height_m": round(float(warp_xpos[:, b, 1, 2].max()), 4),
                         "c_largest_height_m": round(float(ref["xpos"][:, b, 1, 2].max()), 4),
                         "warp_steps_touching_the_ceiling": int((warp_contacts[:, b] > 0).sum()),
                         "c_steps_touching_the_ceiling": int((ref["contacts"][:, b] > 0).sum())})
    thrown, standing = rows_out[:-1], rows_out[-1]
    result = {"cells": [list(c) for c in cells], "throws": rows_out,
              "warp_respawns_per_ant": np.asarray(state.nan_count)[:, 0].tolist(), "warp_pool": env.overflow_report(state)}
    result["passed"] = bool(
        all(r["warp_steps_touching_the_ceiling"] > 0 and r["c_steps_touching_the_ceiling"] > 0
            and CEILING_CONTACT_HEIGHT < r["warp_largest_height_m"] < CEILING_HEIGHT
            and CEILING_CONTACT_HEIGHT < r["c_largest_height_m"] < CEILING_HEIGHT for r in thrown)
        and standing["warp_steps_touching_the_ceiling"] == 0 and standing["c_steps_touching_the_ceiling"] == 0
        and standing["warp_largest_height_m"] < CEILING_ZONE_HEIGHT
        and sum(result["warp_respawns_per_ant"]) == 0 and not result["warp_pool"]["dropped"])
    return result


def gate_parity(cfg):
    """The speed study's three gates against C MuJoCo, on this map's model at this configuration."""
    jax_cfg = replace(cfg, physics="jax")
    parity = checks.one_step_parity(cfg=jax_cfg, with_jax=False)
    divergence = checks.trajectory_divergence(cfg=jax_cfg)
    resting = checks.resting_height(changes={"map_name": cfg.map_name, "start_cell": cfg.start_cell,
                                             "goal_cell": cfg.goal_cell, "physics_time_step": cfg.physics_time_step,
                                             "frame_skip": cfg.frame_skip}, with_jax=False)
    free, contact = parity["contact_free"]["warp_against_c"], parity["in_contact"]["warp_against_c"]
    result = {"one_step": parity, "trajectory": divergence, "resting": resting}
    result["passed"] = bool(
        free["count"] > 0 and contact["count"] > 0 and free["largest"] < 1e-4 and contact["largest"] < 5e-2
        and not parity["pool"]["dropped"] and parity["non_finite_resets"] == 0
        and all(divergence[name]["all_finite"] and divergence[name]["largest_before_the_first_contact"] < 1e-4
                and divergence[name]["largest_over_all_steps"] < 4.0 for name in ("warp", "jax"))
        and abs(resting["warp"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
        and abs(resting["c_mujoco"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
        and resting["warp"]["c_settled_state_moved_by"] < 1e-3)
    return result


def gate_guard(cfg):
    """The six forced ants with this map's goal and outer wall; then the volume bound on both sides of each face."""
    env = WarpAntMaze(cfg, len(ANTS), 1)
    state, obs, reward, terminated, truncated, final_obs = jax.jit(env.step)(forced_state(env), jnp.zeros((len(ANTS), 1, 8), F32))
    spawn = np.concatenate([np.asarray(env.init_qpos), np.zeros(14, np.float32)])
    obs, final_obs = np.asarray(obs)[:, 0], np.asarray(final_obs)[:, 0]
    reward, terminated, truncated = (np.asarray(x)[:, 0] for x in (reward, terminated, truncated))
    respawned = [b for b in range(len(ANTS)) if np.array_equal(final_obs[b], spawn) and truncated[b]]
    # GOAL: the volume bound at this map's outer faces: 1 cm inside is kept, 1 cm outside is respawned.
    hx, hy = maze_half_extent(cfg.map_name)
    xy = jnp.asarray([[[hx - 0.01, 0.0], [hx + 0.01, 0.0], [0.0, -(hy - 0.01)], [0.0, -(hy + 0.01)]]], F32)
    flags = faults.beyond_bounds(jnp.full((1, 4), 0.7, F32), jnp.ones((1, 4), F32), xy, jnp.ones((1, 4), F32),
                                 cfg.fault_torso_speed, cfg.fault_torso_angular_speed, (hx, hy))
    result = {"ants": list(ANTS), "respawned": respawned, "expected_respawned": list(BROKEN),
              "reward": reward.tolist(), "terminated": terminated.tolist(),
              "physics_fault_respawns_per_copy": np.asarray(state.physics_fault_respawns).tolist(),
              "outside_wall_ant_xy": [maze_half_extent(cfg.map_name)[0] + 2.0, 0.0],
              "volume_bound_inside_outside": np.asarray(flags).tolist(), "half_extent": [hx, hy],
              "bounds": [cfg.fault_torso_speed, cfg.fault_torso_angular_speed]}
    result["passed"] = bool(respawned == list(BROKEN) and all(reward[b] == 0.0 for b in BROKEN)
                            and terminated.tolist() == [False, False, True, False, False, False]
                            and result["volume_bound_inside_outside"] == [False, True, False, True])
    return result


GATES = {"file": None, "reset": gate_reset, "goal": gate_goal, "step_limit": gate_step_limit,
         "guard": gate_guard, "walls": gate_walls, "ceiling": gate_ceiling, "parity": gate_parity}


def main():
    """Run the chosen gates on one map; write the record after every gate, so a stopped job keeps the finished ones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, choices=["umaze", "medium", "large"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gates", default=",".join(GATES))
    arguments = parser.parse_args()
    cfg, path = configuration(arguments.map)
    record = {"map": arguments.map, "configuration": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
              "started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "gates": {}}
    for name in arguments.gates.split(","):
        start = time.perf_counter()
        result = gate_file(cfg, path) if name == "file" else GATES[name](cfg)
        result["seconds"] = round(time.perf_counter() - start, 1)
        record["gates"][name] = result
        print(f"{arguments.map} {name}: {'passed' if result['passed'] else 'FAILED'} ({result['seconds']} s)", flush=True)
        arguments.out.write_text(json.dumps(record, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    record["all_passed"] = all(r["passed"] for r in record["gates"].values())
    record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    arguments.out.write_text(json.dumps(record, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(f"{arguments.map}: all gates passed: {record['all_passed']}", flush=True)


if __name__ == "__main__":
    main()
