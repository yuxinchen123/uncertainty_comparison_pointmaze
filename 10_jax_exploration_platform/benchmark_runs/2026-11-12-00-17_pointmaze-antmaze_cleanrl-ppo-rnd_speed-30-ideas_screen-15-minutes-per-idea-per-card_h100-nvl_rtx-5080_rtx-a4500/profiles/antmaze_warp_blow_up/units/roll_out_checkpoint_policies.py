"""Roll the policies of a training checkpoint out in the Warp AntMaze and record every onset of a very
large torso speed.

The checkpoint of one training unit holds its policies and the environments as they were in training.
This script rebuilds the unit's trainer from the unit's configuration file (optionally with other solver
settings), restores the checkpoint, and then steps the environments with the saved policies, with fresh
action noise and no learning. After every environment step it reads the torso speed. An environment step
whose speed passes `--event-speed` while the step before it (in the same episode) had not is an onset.
For each onset it keeps the positions, velocities, solver start values and actions of the `--lead-steps`
environment steps before it, so that the event can be played again one environment at a time on
processors (`replay_events.py`).

It writes two files: `<out>.json` (counts, the speed histogram, timings) and `<out>_events.npz` (the
kept onsets). With `--time-iterations` it then times whole training iterations from the same state.

With `--environment-from <recorded configuration file>` the environment of the measurement is the one that
file describes, read by the training command's own loader and built by the environment's own code: the
gate with trained policies for a physics configuration (`units/trained_policy_gate.sh`).

Run on a card through the serval05 queue as a profile unit, never on a login node.
"""
import argparse
import json
import os
import socket
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

SPEED_EDGES = (0.0, 15.0, 30.0, 100.0, 1e3, 1e6, float("inf"))
# the torso's angular speed in rad/s and the largest joint speed in rad/s, after an environment step
SPIN_EDGES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 400.0, 1e3, float("inf"))
JOINT_EDGES = (0.0, 20.0, 40.0, 80.0, 160.0, 1e3, float("inf"))


def parse():
    """The command line of this script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration-file", type=Path, required=True, help="the unit's configuration file")
    parser.add_argument("--solver-iterations", type=int, default=0, help="0: the configuration file's value")
    parser.add_argument("--ls-iterations", type=int, default=0, help="0: the configuration file's value")
    parser.add_argument("--integrator", default="", help="empty: the configuration file's value")
    parser.add_argument("--environment-from", type=Path,
                        help="a recorded configuration file whose [environment] the measurement runs with, in place "
                             "of the unit's (the map and the task must be the unit's); the options below then "
                             "change that environment")
    parser.add_argument("--timestep", type=float, default=0.0,
                        help="physics time step in seconds (the field physics_time_step); 0: the file's value")
    parser.add_argument("--frame-skip", type=int, default=0, help="physics steps per environment step; 0: the file's value")
    parser.add_argument("--fault-torso-speed", type=float, default=-1.0,
                        help="the fault guard's bound in metres per second; negative: the configuration file's value")
    parser.add_argument("--fault-torso-angular-speed", type=float, default=-1.0,
                        help="the fault guard's bound in radians per second; negative: the configuration file's value")
    parser.add_argument("--rollout-seconds", type=float, default=300.0)
    parser.add_argument("--chunk-steps", type=int, default=64)
    parser.add_argument("--event-speed", type=float, default=100.0, help="metres per second")
    parser.add_argument("--moderate-speed", type=float, default=30.0, help="metres per second")
    parser.add_argument("--lead-steps", type=int, default=24)
    parser.add_argument("--max-events", type=int, default=300)
    parser.add_argument("--max-moderate", type=int, default=60)
    parser.add_argument("--time-iterations", type=int, default=0, help="whole training iterations to time at the end")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    return parser.parse_args()


def onsets(over, done, carried_over, carried_done):
    """Which environment steps are the first of a run of steps above a speed.

    over, done: [T, B] bool, the speed after the step is above the bound / the step ended its episode.
    carried_over, carried_done: [B] bool, the same two for the step before the chunk.

    before: over [[F], [T], [T], [T]], done [[F], [F], [T], [F]], carried both [F]
    after:  [[F], [T], [F], [T]]   (the fourth step starts a new episode, so it is a new onset)
    """
    import numpy as np
    previous_over = np.concatenate([carried_over[None], over[:-1]])
    previous_done = np.concatenate([carried_done[None], done[:-1]])
    return over & ~(previous_over & ~previous_done)


def main():
    """Restore, roll out, record onsets, time."""
    args = parse()
    started = time.time()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import run_training as rt
    from exploration_platform.agents.ppo.networks import actor_mean
    from exploration_platform.bonuses.registry import make_bonus
    from exploration_platform.training.checkpoint import read_checkpoint, restore_state
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.state import named_params, stored_params

    # GOAL: the unit's trainer, with the solver settings of this measurement in the environment. The
    # unit's configuration file was written before later environment fields existed, and the loader
    # wants every field: a copy of the file gets each missing field at the field's default.
    import tempfile
    from dataclasses import fields
    from exploration_platform.envs.antmaze.am_common import AntMazeConfig
    text = args.configuration_file.read_text()
    missing = [f"{field.name} = {json.dumps(field.default)}" for field in fields(AntMazeConfig)
               if f"\n{field.name} = " not in text]
    head, rest = text.split("[environment]\n", 1)
    complete = Path(tempfile.mkdtemp()) / args.configuration_file.name
    complete.write_text(head + "[environment]\n" + "".join(line + "\n" for line in missing) + rest)
    unit_args = rt.parse_args(["--run-dir", str(args.out.parent / "unused_run_folder"), "--unit-id", "unused",
                               "--configuration-file", str(complete),
                               "--reward-setting", "arrival_plus_one"])
    config, env_cfg = rt.build_config(unit_args), rt.build_env_config(unit_args)
    if args.environment_from:
        # GOAL: the environment of another recorded file, through the training command's loader.
        # before: the unit's environment (the physics it was trained with)
        # after:  the environment that file describes; it must be the unit's map and task
        other = rt.build_env_config(rt.parse_args([
            "--run-dir", str(args.out.parent / "unused_run_folder"), "--unit-id", "unused",
            "--configuration-file", str(args.environment_from), "--reward-setting", "arrival_plus_one"]))
        same = ("map_name", "start_cell", "goal_cell", "max_episode_steps", "continuing_task", "goal_radius")
        if any(getattr(other, name) != getattr(env_cfg, name) for name in same):
            raise ValueError(f"{args.environment_from} describes another task than the unit's: "
                             + ", ".join(f"{name} {getattr(other, name)!r} against {getattr(env_cfg, name)!r}"
                                         for name in same if getattr(other, name) != getattr(env_cfg, name)))
        env_cfg = other
    changes = {name: value for name, value in (("solver_iterations", args.solver_iterations),
                                               ("ls_iterations", args.ls_iterations),
                                               ("integrator", args.integrator),
                                               ("frame_skip", args.frame_skip),
                                               ("physics_time_step", args.timestep)) if value}
    if args.fault_torso_speed >= 0.0:
        changes["fault_torso_speed"] = args.fault_torso_speed
    if args.fault_torso_angular_speed >= 0.0:
        changes["fault_torso_angular_speed"] = args.fault_torso_angular_speed
    env_cfg = replace(env_cfg, **changes)
    saved = read_checkpoint(args.checkpoint)
    trainer = Runner(config, bonus=make_bonus(unit_args.bonus, rt.bonus_options_of(unit_args)), env_cfg=env_cfg,
                     bonus_constants=saved["bonus_constants"])
    env = trainer.env
    # GOAL: restore. The checkpoint was written before the environment state got its count of
    # physics-fault respawns and its largest angular speed (2026-11-12), so it holds two arrays
    # fewer than today's state: they are put in as zeros at their places in the tree's order.
    template = trainer.init_state(run_seed=unit_args.run_seed)
    leaves = list(saved["leaves"])
    fresh = jax.tree.leaves(template)
    if len(leaves) == len(fresh) - 2:
        for name in ("physics_fault_respawns", "largest_torso_angular_speed"):
            place = next(index for index, leaf in enumerate(fresh) if leaf is getattr(template.env_state, name))
            leaves.insert(place, np.zeros(fresh[place].shape, fresh[place].dtype))
    state = restore_state(template, leaves)
    del saved
    C, N, T = trainer.cfg.n_copies, trainer.cfg.n_envs, args.chunk_steps
    B = C * N
    actor = named_params(stored_params(state, trainer.layout), trainer.layout)["agent"]["actor"]
    print(f"restored iteration {int(state.step)}: {C} policies x {N} environments, solver "
          f"{env_cfg.solver_iterations}/{env_cfg.ls_iterations}, integrator {env_cfg.integrator}, time step "
          f"{env_cfg.physics_time_step} s x {env_cfg.frame_skip} physics steps, fault guard "
          f"{env_cfg.fault_torso_speed} m/s and {env_cfg.fault_torso_angular_speed} rad/s, "
          f"{time.time() - started:.0f} s since the start", flush=True)

    @jax.jit
    def chunk(env_state, obs, key):
        """T environment steps with the saved policies; per step and environment: the observation
        before, the action, the solver start values, the torso speed and height after, the two ends."""
        noise = jax.random.normal(key, (T, C, N, env.act_dim), jnp.float32)

        def body(carry, z):
            """One environment step of every environment."""
            env_state, obs = carry
            action = actor_mean(actor, obs) + jnp.exp(actor["logstd"])[:, None, :] * z
            warm = env_state.data.qacc_warmstart
            env_state, next_obs, _, terminated, truncated, final_obs = env.step(env_state, action)
            velocity, turning = final_obs[..., 15:18], final_obs[..., 18:21]
            speed = jnp.sqrt((velocity * velocity).sum(-1))
            spin = jnp.sqrt((turning * turning).sum(-1))
            joint = jnp.abs(final_obs[..., 21:29]).max(-1)
            out = (obs.reshape(B, -1), action.reshape(B, -1), warm, speed.reshape(B),
                   final_obs[..., 2].reshape(B), terminated.reshape(B), truncated.reshape(B),
                   spin.reshape(B), joint.reshape(B))
            return (env_state, next_obs), out

        (env_state, obs), outs = jax.lax.scan(body, (env_state, obs), noise)
        return env_state, obs, outs

    # GOAL: roll out for the given time; after every chunk find the onsets and keep their lead-up.
    key = jax.random.PRNGKey(args.noise_seed)
    env_state, obs = state.env_state, state.obs
    nan_before = int(np.asarray(env_state.nan_count).sum())
    histogram = np.zeros(len(SPEED_EDGES) - 1, np.int64)
    spin_histogram = np.zeros(len(SPIN_EDGES) - 1, np.int64)
    joint_histogram = np.zeros(len(JOINT_EDGES) - 1, np.int64)
    # the same angular-speed histogram for the steps that end with the torso above 1.0 m (off the floor)
    spin_histogram_high = np.zeros(len(SPIN_EDGES) - 1, np.int64)
    respawns_before = int(np.asarray(env_state.physics_fault_respawns).sum())
    counts = {"environment_steps": 0, "episodes_ended": 0, "arrivals": 0, "steps_above_1_5_m": 0,
              "onsets_large": 0, "onsets_moderate": 0}
    carried = {"large": np.zeros(B, bool), "moderate": np.zeros(B, bool), "done": np.zeros(B, bool)}
    kept = {"large": [], "moderate": []}
    limits = {"large": args.max_events, "moderate": args.max_moderate}
    previous, chunk_seconds, chunks = None, [], 0
    roll_start = time.time()
    while time.time() - roll_start < args.rollout_seconds:
        tick = time.time()
        env_state, obs, outs = chunk(env_state, obs, jax.random.fold_in(key, chunks))
        speed, height = np.asarray(outs[3]), np.asarray(outs[4])
        terminated, truncated = np.asarray(outs[5]), np.asarray(outs[6])
        chunk_seconds.append(time.time() - tick)
        done = terminated | truncated
        histogram += np.histogram(speed, bins=SPEED_EDGES)[0]
        spin, joint = np.asarray(outs[7]), np.asarray(outs[8])
        spin_histogram += np.histogram(spin, bins=SPIN_EDGES)[0]
        spin_histogram_high += np.histogram(spin[height > 1.0], bins=SPIN_EDGES)[0]
        joint_histogram += np.histogram(joint, bins=JOINT_EDGES)[0]
        counts["environment_steps"] += speed.size
        counts["episodes_ended"] += int(done.sum())
        counts["arrivals"] += int(terminated.sum())
        counts["steps_above_1_5_m"] += int((height > 1.5).sum())
        for kind, low, high in (("large", args.event_speed, np.inf), ("moderate", args.moderate_speed, args.event_speed)):
            # a moderate onset is a step in the moderate band; a later large speed is its own onset
            over = (speed > low) & (speed <= high)
            first = onsets(over, done, carried[kind], carried["done"])
            carried[kind] = over[-1]
            counts[f"onsets_{kind}"] += int(first.sum())
            for t, b in zip(*np.nonzero(first)):
                if len(kept[kind]) >= limits[kind]:
                    break
                # before: the onset is step t of this chunk; after: rows t-L..t of the two chunks
                # joined, per array [L+1, ...] (fewer rows when the run has just started)
                L = args.lead_steps
                parts = []
                for index in range(7):
                    now = outs[index][max(0, t - L):t + 1, b]
                    if t < L and previous is not None:
                        now = jnp.concatenate([previous[index][T - (L - t):, b], now])
                    parts.append(np.asarray(now))
                after = speed[t + 1:t + 9, b]
                kept[kind].append({"policy": b // N, "environment": b % N, "chunk": chunks, "step": int(t),
                                   "obs": parts[0], "action": parts[1], "warm": parts[2], "speed": parts[3],
                                   "height": parts[4], "terminated": parts[5], "truncated": parts[6],
                                   "speed_after": after})
        carried["done"] = done[-1]
        previous = outs
        chunks += 1
    roll_seconds = time.time() - roll_start
    fault_speed = np.asarray(env_state.largest_torso_speed)
    steady = chunk_seconds[2:] or chunk_seconds
    record = {
        "host": socket.gethostname(), "measured_iso": datetime.now().astimezone().isoformat(),
        "checkpoint": args.checkpoint.name, "checkpoint_iteration": int(state.step),
        "configuration_file": str(args.configuration_file), "environment_config": asdict(env_cfg),
        "policies": C, "environments_per_policy": N, "chunk_steps": T, "chunks": chunks,
        "event_speed": args.event_speed, "moderate_speed": args.moderate_speed, **counts,
        "non_finite_respawns": int(np.asarray(env_state.nan_count).sum()) - nan_before,
        "speed_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in SPEED_EDGES],
        "speed_histogram_counts": histogram.tolist(),
        "angular_speed_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in SPIN_EDGES],
        "angular_speed_histogram_counts": spin_histogram.tolist(),
        "angular_speed_histogram_counts_torso_above_1_m": spin_histogram_high.tolist(),
        "largest_joint_speed_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in JOINT_EDGES],
        "largest_joint_speed_histogram_counts": joint_histogram.tolist(),
        "physics_time_step": env_cfg.physics_time_step,
        "environment_from": str(args.environment_from) if args.environment_from else None,
        "share_of_steps_above_1_5_m": counts["steps_above_1_5_m"] / max(counts["environment_steps"], 1),
        "physics_fault_respawns": int(np.asarray(env_state.physics_fault_respawns).sum()) - respawns_before,
        "largest_speed_after_any_physics_step_per_policy_over_1e2": int((fault_speed > 1e2).sum()),
        "largest_speed_after_any_physics_step": float(fault_speed.max()),
        "largest_angular_speed_after_any_physics_step": float(np.asarray(env_state.largest_torso_angular_speed).max()),
        "pool": env.overflow_report(env_state),
        "rollout_seconds": roll_seconds, "seconds_per_chunk_steady": float(np.mean(steady)),
        "rollout_environment_steps_per_second": T * B / float(np.mean(steady)),
        "events_kept": {kind: len(rows) for kind, rows in kept.items()},
    }
    print(json.dumps({name: record[name] for name in (
        "environment_steps", "onsets_large", "onsets_moderate", "non_finite_respawns", "physics_fault_respawns",
        "speed_histogram_counts", "angular_speed_histogram_counts", "angular_speed_histogram_counts_torso_above_1_m",
        "largest_joint_speed_histogram_counts", "largest_speed_after_any_physics_step",
        "largest_angular_speed_after_any_physics_step", "share_of_steps_above_1_5_m", "seconds_per_chunk_steady",
        "events_kept")}), flush=True)

    # GOAL: the kept onsets as arrays (rows padded at the front to L+1 with the first row).
    arrays = {}
    for kind, rows in kept.items():
        if not rows:
            continue
        L1 = args.lead_steps + 1
        pad = lambda a: np.concatenate([np.repeat(a[:1], L1 - len(a), axis=0), a]) if len(a) < L1 else a
        for name in ("obs", "action", "warm", "speed", "height", "terminated", "truncated"):
            arrays[f"{kind}/{name}"] = np.stack([pad(row[name]) for row in rows])
        arrays[f"{kind}/rows_valid"] = np.asarray([len(row["obs"]) for row in rows])
        arrays[f"{kind}/speed_after"] = np.stack([np.pad(row["speed_after"], (0, 8 - len(row["speed_after"])),
                                                          constant_values=np.nan) for row in rows])
        for name in ("policy", "environment", "chunk", "step"):
            arrays[f"{kind}/{name}"] = np.asarray([row[name] for row in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out) + "_events.npz", **arrays)

    # GOAL: whole training iterations from the same state, timed after two warm-up calls.
    if args.time_iterations > 0:
        timed_state = state._replace(env_state=env_state, obs=obs)
        seconds = []
        for call in range(args.time_iterations + 2):
            iteration = int(timed_state.step) + 1
            tick = time.time()
            timed_state, _ = trainer.iterate(timed_state, trainer.lr_argument(iteration, unit_args.iterations))
            jax.block_until_ready(timed_state.obs)
            seconds.append(time.time() - tick)
            print(f"training iteration call {call}: {seconds[-1]:.3f} s", flush=True)
        timed = seconds[2:]
        per_iteration = float(np.mean(timed))
        steps = trainer.cfg.num_steps * B
        record["training_iteration"] = {
            "copies": C, "environments_per_copy": N, "calls_timed": len(timed),
            "compile_and_first_call_seconds": seconds[0], "seconds_per_iteration": per_iteration,
            "seconds_per_iteration_smallest": float(np.min(timed)),
            "total_steps_per_second": steps / per_iteration,
            "steps_per_second_per_copy": steps / per_iteration / C,
            "hours_per_million_steps_per_copy": 1e6 / (3600.0 * steps / per_iteration / C)}
    record["total_seconds"] = time.time() - started
    Path(str(args.out) + ".json").write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}.json and {args.out}_events.npz after {record['total_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
