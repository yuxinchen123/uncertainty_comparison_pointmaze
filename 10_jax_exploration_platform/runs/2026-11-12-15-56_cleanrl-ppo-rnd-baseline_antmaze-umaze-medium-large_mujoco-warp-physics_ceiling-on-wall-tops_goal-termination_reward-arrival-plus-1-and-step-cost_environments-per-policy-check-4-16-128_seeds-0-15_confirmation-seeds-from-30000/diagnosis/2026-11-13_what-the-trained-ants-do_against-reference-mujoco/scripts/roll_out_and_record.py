"""Roll the policies of a training checkpoint out in the Warp AntMaze they were trained in, from the spawn,
and record what the ants do.

The trainer of the checkpoint's unit is rebuilt from the unit's configuration file and the checkpoint is
restored, as in the blow-up diagnosis (`profiles/antmaze_warp_blow_up/units/roll_out_checkpoint_policies.py`
of the speed study). Every environment is then put back at the spawn and stepped with the saved policies,
with fresh action noise and no learning. The physics is the unit's own (MuJoCo Warp, the fault guard on).

It writes four files beside `--out`:
  <out>_summary.json       counts per policy and per configuration, histograms of height and speed
  <out>_episodes.npz       one row per finished episode: policy, length, arrived, largest height, speed
                           and turning speed, steps above 1.0 and 1.5 m, steps with the torso centre
                           inside a wall cell
  <out>_trajectories.npz   for the first `--record-environments` environments of every policy and the
                           first `--record-steps` environment steps: the state before each step, the
                           action, and the two episode ends (for the C MuJoCo work on processors)
  <out>_actors.npz         the policies' weights (for closed-loop runs in C MuJoCo)

Run on a card through Slurm or the serval05 queue, never on a login node.
"""
import argparse
import json
import socket
import sys
import tempfile
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

HEIGHT_EDGES = (0.0, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, float("inf"))
SPEED_EDGES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, float("inf"))
SPIN_EDGES = (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, float("inf"))
MAZE = ((1, 1, 1, 1, 1), (1, 0, 0, 0, 1), (1, 1, 1, 0, 1), (1, 0, 0, 0, 1), (1, 1, 1, 1, 1))   # the U-maze, row 0 on top


def parse():
    """The command line of this script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration-file", type=Path, required=True, help="the unit's configuration file")
    parser.add_argument("--steps", type=int, default=2800, help="environment steps to roll out")
    parser.add_argument("--chunk-steps", type=int, default=70)
    parser.add_argument("--record-environments", type=int, default=4, help="per policy")
    parser.add_argument("--record-steps", type=int, default=2100)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    return parser.parse_args()


def in_wall(x, y):
    """Whether a point of the U-maze's floor plan lies in a wall cell (cells of 4 m, row 0 on top).

    before: x -4.0, y -4.0 (the spawn, cell (3, 1))   after: False
    before: x -4.0, y 0.0 (cell (2, 1), the wall between spawn and goal)   after: True
    """
    import numpy as np
    column = np.clip(np.floor(x / 4.0 + 2.5).astype(int), 0, 4)
    row = np.clip(np.floor(2.5 - y / 4.0).astype(int), 0, 4)
    return np.asarray(MAZE, bool)[row, column]


def main():
    """Restore, respawn, roll out, record."""
    args = parse()
    started = time.time()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import run_training as rt
    from exploration_platform.agents.ppo.networks import actor_mean
    from exploration_platform.bonuses.registry import make_bonus
    from exploration_platform.envs.antmaze.am_common import AntMazeConfig
    from exploration_platform.training.checkpoint import read_checkpoint, restore_state
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.state import named_params, stored_params

    # GOAL: the unit's trainer. A configuration file written before a later environment field existed
    # gets that field at its default, because the loader wants every field.
    text = args.configuration_file.read_text()
    missing = [f"{field.name} = {json.dumps(field.default)}" for field in fields(AntMazeConfig)
               if f"\n{field.name} = " not in text]
    head, rest = text.split("[environment]\n", 1)
    complete = Path(tempfile.mkdtemp()) / args.configuration_file.name
    complete.write_text(head + "[environment]\n" + "".join(line + "\n" for line in missing) + rest)
    unit_args = rt.parse_args(["--run-dir", str(args.out.parent / "unused_run_folder"), "--unit-id", "unused",
                               "--configuration-file", str(complete), "--reward-setting", "arrival_plus_one"])
    config, env_cfg = rt.build_config(unit_args), rt.build_env_config(unit_args)
    saved = read_checkpoint(args.checkpoint)
    trainer = Runner(config, bonus=make_bonus(unit_args.bonus, rt.bonus_options_of(unit_args)), env_cfg=env_cfg,
                     bonus_constants=saved["bonus_constants"])
    env = trainer.env
    template = trainer.init_state(run_seed=unit_args.run_seed)
    state = restore_state(template, list(saved["leaves"]))
    del saved
    C, N, T = trainer.cfg.n_copies, trainer.cfg.n_envs, args.chunk_steps
    B, R = C * N, min(args.record_environments, trainer.cfg.n_envs)
    actor = named_params(stored_params(state, trainer.layout), trainer.layout)["agent"]["actor"]
    settings = [list(map(float, row)) for row in trainer.cfg.parameter_settings]
    per_cell = C // len(settings)
    print(f"restored iteration {int(state.step)}: {C} policies x {N} environments, {len(settings)} configurations of "
          f"{per_cell} policies, integrator {env_cfg.integrator}, time step {env_cfg.physics_time_step} s x "
          f"{env_cfg.frame_skip}, solver {env_cfg.solver_iterations}/{env_cfg.ls_iterations}, guard "
          f"{env_cfg.fault_torso_speed} m/s and {env_cfg.fault_torso_angular_speed} rad/s, "
          f"{time.time() - started:.0f} s since the start", flush=True)

    @jax.jit
    def chunk(env_state, obs, key):
        """T environment steps of every environment with the saved policies.

        Per step: for all environments the torso's x, y, height, speed and turning speed after the
        step and the two ends; for the recorded environments the state before the step and the action.
        """
        noise = jax.random.normal(key, (T, C, N, env.act_dim), jnp.float32)

        def body(carry, z):
            """One environment step of every environment."""
            env_state, obs = carry
            action = actor_mean(actor, obs) + jnp.exp(actor["logstd"])[:, None, :] * z
            env_state, next_obs, _, terminated, truncated, final_obs = env.step(env_state, action)
            velocity, turning = final_obs[..., 15:18], final_obs[..., 18:21]
            out = (final_obs[..., 0].reshape(B), final_obs[..., 1].reshape(B), final_obs[..., 2].reshape(B),
                   jnp.sqrt((velocity * velocity).sum(-1)).reshape(B), jnp.sqrt((turning * turning).sum(-1)).reshape(B),
                   terminated.reshape(B), truncated.reshape(B),
                   obs[:, :R].reshape(C * R, -1), action[:, :R].reshape(C * R, -1),
                   terminated[:, :R].reshape(C * R), truncated[:, :R].reshape(C * R))
            return (env_state, next_obs), out

        (env_state, obs), outs = jax.lax.scan(body, (env_state, obs), noise)
        return env_state, obs, outs

    # GOAL: every environment at the spawn, then roll out and count.
    env_state, obs = env.respawn(state.env_state)
    spawn_obs = np.asarray(obs[0, 0])
    policy_of = np.repeat(np.arange(C), N)
    per_policy = {name: np.zeros(C, np.int64) for name in (
        "steps", "above_1_0_m", "above_1_5_m", "speed_over_5", "speed_over_10", "speed_over_15", "turning_over_40",
        "turning_over_60", "turning_over_80", "episodes", "arrivals", "arrival_steps", "in_wall_cell")}
    sums = {name: np.zeros(C) for name in ("height", "speed")}
    histograms = {"height": np.zeros((len(settings), len(HEIGHT_EDGES) - 1), np.int64),
                  "speed": np.zeros((len(settings), len(SPEED_EDGES) - 1), np.int64),
                  "turning": np.zeros((len(settings), len(SPIN_EDGES) - 1), np.int64)}
    episode = {name: np.zeros(B) for name in ("length", "largest_height", "largest_speed", "largest_turning",
                                              "steps_above_1_0", "steps_above_1_5", "steps_in_wall_cell")}
    rows = []
    recorded = {"state": [], "action": [], "terminated": [], "truncated": []}
    key = jax.random.PRNGKey(args.noise_seed)
    seconds = []
    for index in range(args.steps // T):
        tick = time.time()
        env_state, obs, outs = chunk(env_state, obs, jax.random.fold_in(key, index))
        x, y, height, speed, turning = (np.asarray(outs[i]) for i in range(5))
        terminated, truncated = np.asarray(outs[5]), np.asarray(outs[6])
        seconds.append(time.time() - tick)
        if index * T < args.record_steps:
            for name, value in zip(recorded, outs[7:]):
                recorded[name].append(np.asarray(value))
        wall = in_wall(x, y)
        for name, mask in (("above_1_0_m", height > 1.0), ("above_1_5_m", height > 1.5), ("speed_over_5", speed > 5.0),
                           ("speed_over_10", speed > 10.0), ("speed_over_15", speed > 15.0), ("turning_over_40", turning > 40.0),
                           ("turning_over_60", turning > 60.0), ("turning_over_80", turning > 80.0), ("in_wall_cell", wall)):
            per_policy[name] += mask.reshape(T, C, N).sum((0, 2))
        per_policy["steps"] += T * N
        sums["height"] += height.reshape(T, C, N).sum((0, 2))
        sums["speed"] += speed.reshape(T, C, N).sum((0, 2))
        for cell in range(len(settings)):
            pick = slice(cell * per_cell * N, (cell + 1) * per_cell * N)
            histograms["height"][cell] += np.histogram(height[:, pick], bins=HEIGHT_EDGES)[0]
            histograms["speed"][cell] += np.histogram(speed[:, pick], bins=SPEED_EDGES)[0]
            histograms["turning"][cell] += np.histogram(turning[:, pick], bins=SPIN_EDGES)[0]
        # GOAL: per episode, what it reached; an episode's row is written at the step that ends it.
        # before: episode accumulators of 2,048 environments; step t ends environments 7 and 90
        # after:  two rows [policy, length, arrived, largest height, ...] and those two accumulators at zero
        for t in range(T):
            episode["length"] += 1
            episode["largest_height"] = np.maximum(episode["largest_height"], height[t])
            episode["largest_speed"] = np.maximum(episode["largest_speed"], speed[t])
            episode["largest_turning"] = np.maximum(episode["largest_turning"], turning[t])
            episode["steps_above_1_0"] += height[t] > 1.0
            episode["steps_above_1_5"] += height[t] > 1.5
            episode["steps_in_wall_cell"] += wall[t]
            done = terminated[t] | truncated[t]
            if done.any():
                ended = np.nonzero(done)[0]
                rows.append(np.stack([policy_of[ended], episode["length"][ended], terminated[t][ended],
                                      episode["largest_height"][ended], episode["largest_speed"][ended],
                                      episode["largest_turning"][ended], episode["steps_above_1_0"][ended],
                                      episode["steps_above_1_5"][ended], episode["steps_in_wall_cell"][ended]], 1))
                np.add.at(per_policy["episodes"], policy_of[ended], 1)
                arrived = ended[terminated[t][ended]]
                np.add.at(per_policy["arrivals"], policy_of[arrived], 1)
                np.add.at(per_policy["arrival_steps"], policy_of[arrived], episode["length"][arrived].astype(np.int64))
                for name in episode:
                    episode[name][ended] = 0
    table = np.concatenate(rows) if rows else np.zeros((0, 9))
    respawns = np.asarray(env_state.physics_fault_respawns)
    by_cell = lambda values: [float(np.asarray(values)[cell * per_cell:(cell + 1) * per_cell].sum()) for cell in range(len(settings))]
    steps_by_cell = np.asarray(by_cell(per_policy["steps"]))
    summary = {
        "host": socket.gethostname(), "measured_iso": datetime.now().astimezone().isoformat(),
        "checkpoint": args.checkpoint.name, "checkpoint_iteration": int(state.step),
        "environment_config": asdict(env_cfg), "policies": C, "environments_per_policy": N,
        "environment_steps_per_policy": int(per_policy["steps"][0]), "configuration_fields": ["learning_rate", "ent_coef", "int_coef", "clip_coef"],
        "configurations": settings, "policies_per_configuration": per_cell,
        "per_policy": {name: values.tolist() for name, values in per_policy.items()},
        "per_policy_mean_height": (sums["height"] / per_policy["steps"]).tolist(),
        "per_policy_mean_speed": (sums["speed"] / per_policy["steps"]).tolist(),
        "per_policy_fault_respawns": respawns.tolist(),
        "per_configuration": {name: by_cell(values) for name, values in per_policy.items()},
        "per_configuration_fault_respawns": by_cell(respawns),
        "per_configuration_share_above_1_5_m": (np.asarray(by_cell(per_policy["above_1_5_m"])) / steps_by_cell).tolist(),
        "height_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in HEIGHT_EDGES],
        "speed_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in SPEED_EDGES],
        "turning_histogram_edges": [edge if np.isfinite(edge) else "inf" for edge in SPIN_EDGES],
        "height_histogram_per_configuration": histograms["height"].tolist(),
        "speed_histogram_per_configuration": histograms["speed"].tolist(),
        "turning_histogram_per_configuration": histograms["turning"].tolist(),
        "largest_torso_speed_after_any_physics_step_per_policy": np.asarray(env_state.largest_torso_speed).tolist(),
        "largest_turning_speed_after_any_physics_step_per_policy": np.asarray(env_state.largest_torso_angular_speed).tolist(),
        "non_finite_or_guard_respawns_total": int(np.asarray(env_state.nan_count).sum()),
        "pool": env.overflow_report(env_state), "seconds_per_chunk_after_the_first_two": float(np.mean(seconds[2:] or seconds)),
        "episode_table_columns": ["policy", "length", "arrived", "largest_height", "largest_speed", "largest_turning",
                                  "steps_above_1_0_m", "steps_above_1_5_m", "steps_in_wall_cell"],
        "episodes": int(len(table)), "recorded_environments_per_policy": R,
        "recorded_steps": int(len(recorded["state"]) * T), "total_seconds": time.time() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + "_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(str(args.out) + "_episodes.npz", table=table.astype(np.float32))
    np.savez(str(args.out) + "_trajectories.npz", spawn_obs=spawn_obs,
             **{name: np.concatenate(parts) for name, parts in recorded.items()})
    np.savez(str(args.out) + "_actors.npz", configurations=np.asarray(settings), policies_per_configuration=per_cell,
             **{name: np.asarray(value) for name, value in actor.items()})
    cells = summary["per_configuration"]
    for cell, setting in enumerate(settings):
        print(f"learning rate {setting[0]:g}, weight {setting[2]:g}: episodes {cells['episodes'][cell]:.0f}, arriving "
              f"{cells['arrivals'][cell] / max(cells['episodes'][cell], 1):.3f}, above 1.5 m "
              f"{cells['above_1_5_m'][cell] / cells['steps'][cell]:.3f}, above 1.0 m {cells['above_1_0_m'][cell] / cells['steps'][cell]:.3f}, "
              f"speed over 5 m/s {cells['speed_over_5'][cell] / cells['steps'][cell]:.4f}, over 10 m/s "
              f"{cells['speed_over_10'][cell] / cells['steps'][cell]:.5f}, turning over 60 rad/s "
              f"{cells['turning_over_60'][cell] / cells['steps'][cell]:.5f}, in a wall cell {cells['in_wall_cell'][cell]:.0f}, "
              f"guard respawns {summary['per_configuration_fault_respawns'][cell]:.0f}", flush=True)
    print(f"wrote {args.out}_summary.json and three more files after {summary['total_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
