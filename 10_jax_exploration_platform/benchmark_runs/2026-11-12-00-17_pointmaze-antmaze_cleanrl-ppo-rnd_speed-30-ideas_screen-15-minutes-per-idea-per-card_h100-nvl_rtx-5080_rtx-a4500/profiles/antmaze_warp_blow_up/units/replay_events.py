"""Play recorded onsets of a very large torso speed again, one ant at a time, on processors.

`roll_out_checkpoint_policies.py` keeps, for every onset, the state before each of the environment steps
that lead up to it and the actions of those steps. This script starts from the state `lead` environment
steps before the onset and applies the recorded actions again:

1. in C MuJoCo (float64), with several solver settings: does the same ant reach the large speed there
   too, and what is it touching when the speed starts to grow (contacts by geom name and depth, constraint
   rows, solver iterations used, joints beyond their limits);
2. in MuJoCo Warp on processors (float32, the training physics), with several solver settings.

It writes one JSON record. Run on a compute node with the platform_jax_warp interpreter, never on a
login node:
  <platform_jax_warp python> replay_events.py --events <..._events.npz> --out <file.json>
"""
import argparse
import json
import os
import socket
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))

# (name, integrator, solver iterations, line-search iterations, physics time step in seconds); an
# environment step is always 0.05 s, so 5 physics steps of 0.01 s or 10 of 0.005 s
C_SETTINGS = (("as trained: implicitfast, 4 and 8", "implicitfast", 4, 8, 0.01),
              ("implicitfast, 8 and 16", "implicitfast", 8, 16, 0.01),
              ("implicitfast, 16 and 16", "implicitfast", 16, 16, 0.01),
              ("implicitfast, 100 and 50 (MuJoCo's defaults)", "implicitfast", 100, 50, 0.01),
              ("RK4, 100 and 50 (the reference model's settings)", "RK4", 100, 50, 0.01),
              ("RK4, 4 and 8", "RK4", 4, 8, 0.01),
              ("implicit (with the velocity derivatives of the Coriolis forces), 4 and 8", "implicit", 4, 8, 0.01),
              ("Euler, 4 and 8", "Euler", 4, 8, 0.01),
              ("implicitfast, 4 and 8, time step 0.005 s", "implicitfast", 4, 8, 0.005),
              ("implicitfast, 4 and 8, time step 0.0025 s", "implicitfast", 4, 8, 0.0025),
              ("implicit, 4 and 8, time step 0.005 s", "implicit", 4, 8, 0.005),
              ("RK4, 4 and 8, time step 0.005 s", "RK4", 4, 8, 0.005))
WARP_SETTINGS = ((4, 8), (8, 16), (16, 16), (50, 50))
LARGE = 100.0   # metres per second


def usable_lead(terminated, truncated, rows_valid):
    """How many environment steps before the onset belong to the onset's own episode and were kept.

    before: 25 rows, the episode before ended at row 20 -> the onset's episode starts at row 21
    after:  3 (rows 21, 22, 23 lead up to the onset in row 24)
    """
    import numpy as np
    last = len(terminated) - 1
    ended = np.nonzero((terminated | truncated)[:last])[0]
    first_row = max(len(terminated) - rows_valid, ended[-1] + 1 if len(ended) else 0)
    return last - first_row


def c_model(cfg, integrator, iterations, ls_iterations, timestep=0.01):
    """The C MuJoCo model of the environment with the given solver settings and physics time step."""
    import mujoco
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(
        replace(cfg, integrator=integrator, solver_iterations=iterations, ls_iterations=ls_iterations)))
    model.opt.timestep = timestep
    return model


def c_replay(model, obs, action, warm, detail=False):
    """One ant in C MuJoCo from the state `obs[0]` under the clipped actions, five physics steps each;
    the last action is held for 4 more environment steps.

    Returns the torso speed after every physics step, and with `detail` one dictionary per physics step
    (contacts with names and depths, rows, solver iterations, joints beyond their limits).
    """
    import mujoco
    import numpy as np
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = obs[0, :15], obs[0, 15:]
    data.qacc_warmstart[:] = warm[0]
    speeds, steps = [], []
    actions = list(action) + [action[-1]] * 4
    # an environment step is 0.05 s whatever the physics time step
    substeps = int(round(0.05 / model.opt.timestep))
    for index, torque in enumerate(actions):
        data.ctrl[:] = np.clip(torque, -1.0, 1.0)
        for sub in range(substeps):
            mujoco.mj_step(model, data)
            speed = float(np.linalg.norm(data.qvel[0:3]))
            speeds.append(speed if np.isfinite(speed) else float("inf"))
            c_replay.spin.append(float(np.linalg.norm(data.qvel[3:6])) if np.isfinite(data.qvel[3:6]).all() else float("inf"))
            if detail:
                contacts = []
                for contact in data.contact[:data.ncon]:
                    contacts.append([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                                     mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                                     round(float(contact.dist), 4)])
                # hinge joints beyond their range, in radians (the 8 leg joints: qpos 7..14)
                beyond = []
                for joint in range(1, model.njnt):
                    low, high = model.jnt_range[joint]
                    angle = data.qpos[model.jnt_qposadr[joint]]
                    excess = max(low - angle, angle - high, 0.0)
                    if excess > 0.02:
                        beyond.append([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint), round(float(excess), 3)])
                steps.append({"environment_step": index, "physics_step": sub, "torso_speed": round(speed, 3),
                              "torso_height": round(float(data.qpos[2]), 3),
                              "largest_joint_speed": round(float(np.abs(data.qvel[6:]).max()), 2),
                              "largest_angular_speed": round(float(np.abs(data.qvel[3:6]).max()), 2),
                              "constraint_rows": int(data.nefc), "solver_iterations_used": int(data.solver_niter[0]),
                              "contacts": contacts, "joints_beyond_range": beyond})
    return speeds, steps


# the torso's angular speed after every physics step of the replay that ran last, in rad/s
c_replay.spin = []


def warp_replay(env, obs, action, warm):
    """All given ants at once in MuJoCo Warp on processors, one environment each, from `obs[:, 0]` under
    the clipped actions (the last held for 4 more environment steps).

    obs [E, k+1, 29], action [E, k+1, 8], warm [E, k+1, 14]. Returns the torso speed after every
    physics step, [E, 5 * (k + 5)], non-finite speeds as infinity, and the largest row count per ant.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mujoco import mjx
    with jax.enable_x64(False):
        state = env.reset()
        data = state.data.replace(qpos=jnp.asarray(obs[:, 0, :15], jnp.float32), qvel=jnp.asarray(obs[:, 0, 15:], jnp.float32),
                                  qacc_warmstart=jnp.asarray(warm[:, 0], jnp.float32))
        physics = jax.jit(lambda d: jax.vmap(mjx.step, in_axes=(None, 0))(env.model, d))
        speeds, rows = [], np.zeros(len(obs), np.int64)
        steps = list(range(action.shape[1])) + [action.shape[1] - 1] * 4
        for index in steps:
            data = data.replace(ctrl=jnp.clip(jnp.asarray(action[:, index], jnp.float32), -1.0, 1.0))
            for _ in range(5):
                data = physics(data)
                velocity = np.asarray(data.qvel[:, 0:3], np.float64)
                speed = np.sqrt((velocity * velocity).sum(-1))
                speeds.append(np.where(np.isfinite(speed), speed, np.inf))
                rows = np.maximum(rows, np.asarray(data._impl.nefc).reshape(-1))
    return np.stack(speeds, 1), rows


def main():
    """Replay every kept onset in both implementations and write the record."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--kind", default="large", choices=["large", "moderate"])
    parser.add_argument("--leads", default="1,2,4,8,24")
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--detailed", type=int, default=12, help="onsets whose C MuJoCo replay is written step by step")
    parser.add_argument("--skip-warp", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from exploration_platform.envs.antmaze.am_common import preset

    cfg = replace(preset("umaze"), continuing_task=False)
    events = np.load(args.events)
    kind = args.kind
    obs, action, warm = events[f"{kind}/obs"], events[f"{kind}/action"], events[f"{kind}/warm"]
    count = min(len(obs), args.max_events)
    lead_of = [usable_lead(events[f"{kind}/terminated"][e], events[f"{kind}/truncated"][e],
                           int(events[f"{kind}/rows_valid"][e])) for e in range(count)]
    leads = [int(text) for text in args.leads.split(",")]
    last = obs.shape[1] - 1
    record = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
              "measured_iso": datetime.now().astimezone().isoformat(), "events_file": args.events.name,
              "kind": kind, "onsets": count, "leads": leads, "large_speed": LARGE,
              "onset_speed_recorded": events[f"{kind}/speed"][:count, last].tolist(),
              "usable_lead": lead_of, "c_mujoco": {}, "warp_on_processors": {}, "detail": []}

    # GOAL: C MuJoCo, every setting and lead: how many of the onsets reach the large speed again.
    for name, integrator, iterations, ls_iterations, timestep in C_SETTINGS:
        model = c_model(cfg, integrator, iterations, ls_iterations, timestep)
        record["c_mujoco"][name] = {}
        for lead in leads:
            which = [e for e in range(count) if lead_of[e] >= lead]
            peaks, spins = [], []
            for e in which:
                c_replay.spin = []
                peaks.append(max(c_replay(model, obs[e, last - lead:], action[e, last - lead:], warm[e, last - lead:])[0]))
                spins.append(max(c_replay.spin))
            record["c_mujoco"][name][str(lead)] = {
                "onsets_replayed": len(which), "reached_the_large_speed": int(sum(p > LARGE for p in peaks)),
                "reached_30_metres_per_second": int(sum(p > 30.0 for p in peaks)),
                "largest_speeds_sorted_top_5": [float(min(p, 1e30)) for p in sorted(peaks)[-5:]],
                "median_largest_speed": float(np.median(peaks)) if peaks else None,
                "median_largest_angular_speed": float(np.median(spins)) if spins else None,
                "largest_angular_speeds_sorted_top_5": [float(min(s, 1e30)) for s in sorted(spins)[-5:]]}
            print("C MuJoCo", name, "lead", lead, json.dumps(record["c_mujoco"][name][str(lead)]), flush=True)

    # GOAL: the torso's angular speed in the kept lead-up rows themselves (rad/s), per row before the onset.
    spin_rows = np.sqrt((obs[:count, :, 18:21].astype(np.float64) ** 2).sum(-1))
    record["recorded_angular_speed_before_each_lead_step_median"] = np.median(spin_rows, axis=0).round(1).tolist()
    record["recorded_angular_speed_before_each_lead_step_largest"] = spin_rows.max(0).round(1).tolist()
    record["recorded_speed_after_each_lead_step_median"] = np.median(events[f"{kind}/speed"][:count], axis=0).round(2).tolist()
    record["recorded_height_after_each_lead_step_median"] = np.median(events[f"{kind}/height"][:count], axis=0).round(2).tolist()
    print("recorded angular speed before each lead step, median:", record["recorded_angular_speed_before_each_lead_step_median"], flush=True)
    print("recorded speed after each lead step, median:", record["recorded_speed_after_each_lead_step_median"], flush=True)
    print("recorded height after each lead step, median:", record["recorded_height_after_each_lead_step_median"], flush=True)

    # GOAL: the first onsets step by step in C MuJoCo with the training settings, from 2 steps before.
    model = c_model(cfg, "implicitfast", 4, 8)
    for e in [e for e in range(count) if lead_of[e] >= 2][:args.detailed]:
        speeds, steps = c_replay(model, obs[e, last - 2:], action[e, last - 2:], warm[e, last - 2:], detail=True)
        record["detail"].append({"onset": e, "policy": int(events[f"{kind}/policy"][e]),
                                 "recorded_speed_after_each_environment_step": events[f"{kind}/speed"][e, last - 2:].tolist(),
                                 "recorded_height_after_each_environment_step": events[f"{kind}/height"][e, last - 2:].tolist(),
                                 "state_before": obs[e, last - 2].tolist(),
                                 "actions_clipped": np.clip(action[e, last - 2:], -1, 1).round(3).tolist(),
                                 "physics_steps": steps})

    # GOAL: MuJoCo Warp on processors, every setting and lead.
    if not args.skip_warp:
        from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze
        for iterations, ls_iterations in WARP_SETTINGS:
            name = f"{iterations} and {ls_iterations}"
            record["warp_on_processors"][name] = {}
            for lead in leads:
                which = [e for e in range(count) if lead_of[e] >= lead]
                if not which:
                    continue
                env = WarpAntMaze(replace(cfg, solver_iterations=iterations, ls_iterations=ls_iterations,
                                          physics="warp"), 1, len(which))
                speeds, rows = warp_replay(env, obs[which][:, last - lead:], action[which][:, last - lead:],
                                           warm[which][:, last - lead:])
                peaks = speeds.max(1)
                record["warp_on_processors"][name][str(lead)] = {
                    "onsets_replayed": len(which), "reached_the_large_speed": int((peaks > LARGE).sum()),
                    "largest_speeds_sorted_top_5": [float(min(p, 1e30)) for p in np.sort(peaks)[-5:]],
                    "median_largest_speed": float(np.median(peaks)), "largest_constraint_rows": int(rows.max())}
                print("Warp", name, "lead", lead, json.dumps(record["warp_on_processors"][name][str(lead)]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
