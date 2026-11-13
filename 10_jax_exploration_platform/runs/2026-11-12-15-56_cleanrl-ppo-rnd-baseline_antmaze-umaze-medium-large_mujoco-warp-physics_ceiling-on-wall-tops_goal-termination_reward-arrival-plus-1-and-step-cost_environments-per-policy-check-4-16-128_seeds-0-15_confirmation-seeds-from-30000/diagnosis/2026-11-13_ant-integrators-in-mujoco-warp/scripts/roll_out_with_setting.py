"""Roll the policies of a training checkpoint out in the Warp AntMaze with one physics setting, and count
guard respawns, arrivals and the largest torso speeds.

Adapted from `diagnosis/2026-11-13_what-the-trained-ants-do_against-reference-mujoco/scripts/
roll_out_and_record.py`. The unit's trainer is rebuilt from its configuration file with the setting's
integrator, physics time step and physics steps per environment step written in
(`integrator_options.with_setting`), and the checkpoint is restored into it: the policies are the trained
ones, the physics is the setting's, the guard (30 m/s, 100 rad/s), the ceiling, the goal termination and the
step limit are the unit's. Every environment is put back at the spawn and stepped with the saved policies,
with fresh action noise (the same noise for every setting) and no learning.

An episode that ends by truncation before the step limit was ended by a respawn for a broken physics state
(non-finite, or beyond the guard's bounds): the environment truncates only at the limit otherwise.

It writes beside `--out`:
  <out>_summary.json    per configuration: episodes, arrivals, respawns, largest speeds; timing
  <out>_episodes.npz    one row per finished episode: policy, length, arrived, respawned, largest height,
                        speed and turning speed in the observations

Run on a card through Slurm, never on a login node.
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

BASE = Path("/p/rlprojects/RND/10_jax_exploration_platform")
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrator_options import prepare, with_setting  # noqa: E402


def parse():
    """The command line of this script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration-file", type=Path, required=True, help="the unit's configuration file")
    parser.add_argument("--setting", required=True, help="a slug of integrator_options.SETTINGS")
    parser.add_argument("--steps", type=int, default=2800, help="environment steps per environment")
    parser.add_argument("--chunk-steps", type=int, default=70)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="path without extension")
    return parser.parse_args()


def main():
    """Restore with the setting's physics, respawn, roll out, count."""
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

    # GOAL: the unit's trainer with the setting's physics. A configuration file written before a later
    # environment field existed gets that field at its default, because the loader wants every field.
    text = args.configuration_file.read_text()
    missing = [f"{field.name} = {json.dumps(field.default)}" for field in fields(AntMazeConfig)
               if f"\n{field.name} = " not in text]
    head, rest = text.split("[environment]\n", 1)
    complete = Path(tempfile.mkdtemp()) / args.configuration_file.name
    complete.write_text(with_setting(head + "[environment]\n" + "".join(line + "\n" for line in missing) + rest,
                                     args.setting))
    prepare(args.setting)
    unit_args = rt.parse_args(["--run-dir", str(Path(tempfile.mkdtemp()) / "unused_run_folder"), "--unit-id", "unused",
                               "--configuration-file", str(complete), "--reward-setting", "arrival_plus_one"])
    config, env_cfg = rt.build_config(unit_args), rt.build_env_config(unit_args)
    saved = read_checkpoint(args.checkpoint)
    trainer = Runner(config, bonus=make_bonus(unit_args.bonus, rt.bonus_options_of(unit_args)), env_cfg=env_cfg,
                     bonus_constants=saved["bonus_constants"])
    env = trainer.env
    state = restore_state(trainer.init_state(run_seed=unit_args.run_seed), list(saved["leaves"]))
    del saved
    C, N, T = trainer.cfg.n_copies, trainer.cfg.n_envs, args.chunk_steps
    B, limit = C * N, env_cfg.max_episode_steps
    actor = named_params(stored_params(state, trainer.layout), trainer.layout)["agent"]["actor"]
    settings = [list(map(float, row)) for row in trainer.cfg.parameter_settings]
    per_cell = C // len(settings)
    print(f"restored iteration {int(state.step)}: {C} policies x {N} environments, {len(settings)} configurations of "
          f"{per_cell} policies, integrator {env_cfg.integrator} (Warp model number {int(env.model.opt.integrator)}), "
          f"time step {env_cfg.physics_time_step} s x {env_cfg.frame_skip}, solver {env_cfg.solver_iterations}/"
          f"{env_cfg.ls_iterations}, guard {env_cfg.fault_torso_speed} m/s and {env_cfg.fault_torso_angular_speed} "
          f"rad/s, {time.time() - started:.0f} s since the start", flush=True)

    @jax.jit
    def chunk(env_state, obs, key):
        """T environment steps of every environment with the saved policies; per step the torso's height,
        speed and turning speed in the final observation, and the two episode ends."""
        noise = jax.random.normal(key, (T, C, N, env.act_dim), jnp.float32)

        def body(carry, z):
            """One environment step of every environment."""
            env_state, obs = carry
            action = actor_mean(actor, obs) + jnp.exp(actor["logstd"])[:, None, :] * z
            env_state, next_obs, _, terminated, truncated, final_obs = env.step(env_state, action)
            velocity, turning = final_obs[..., 15:18], final_obs[..., 18:21]
            out = (final_obs[..., 2].reshape(B), jnp.sqrt((velocity * velocity).sum(-1)).reshape(B),
                   jnp.sqrt((turning * turning).sum(-1)).reshape(B), terminated.reshape(B), truncated.reshape(B))
            return (env_state, next_obs), out

        (env_state, obs), outs = jax.lax.scan(body, (env_state, obs), noise)
        return env_state, obs, outs

    # GOAL: every environment at the spawn, then roll out and keep one row per finished episode.
    env_state, obs = env.respawn(state.env_state)
    nan_count_before = np.asarray(env_state.nan_count).copy()
    policy_of = np.repeat(np.arange(C), N)
    episode = {name: np.zeros(B) for name in ("length", "largest_height", "largest_speed", "largest_turning")}
    rows, seconds = [], []
    key = jax.random.PRNGKey(args.noise_seed)
    for index in range(args.steps // T):
        tick = time.time()
        env_state, obs, outs = chunk(env_state, obs, jax.random.fold_in(key, index))
        height, speed, turning, terminated, truncated = (np.asarray(x) for x in outs)
        seconds.append(time.time() - tick)
        # before: accumulators of B environments; step t ends environments 7 and 90
        # after:  two rows [policy, length, arrived, respawned, largest height, speed, turning], those two at zero
        for t in range(T):
            episode["length"] += 1
            episode["largest_height"] = np.maximum(episode["largest_height"], height[t])
            episode["largest_speed"] = np.maximum(episode["largest_speed"], speed[t])
            episode["largest_turning"] = np.maximum(episode["largest_turning"], turning[t])
            done = terminated[t] | truncated[t]
            if done.any():
                ended = np.nonzero(done)[0]
                respawned = truncated[t][ended] & (episode["length"][ended] < limit)
                rows.append(np.stack([policy_of[ended], episode["length"][ended], terminated[t][ended], respawned,
                                      episode["largest_height"][ended], episode["largest_speed"][ended],
                                      episode["largest_turning"][ended]], 1))
                for name in episode:
                    episode[name][ended] = 0
        if index % 10 == 0:
            print(f"[progress] chunk {index + 1} of {args.steps // T}, {seconds[-1]:.2f} s", flush=True)
    table = np.concatenate(rows) if rows else np.zeros((0, 7))

    # GOAL: per configuration (cell of per_cell policies): episodes, arrivals, respawns, largest speeds.
    cell_of_row = (table[:, 0] // per_cell).astype(int)
    respawns_counter = np.asarray(env_state.physics_fault_respawns)
    largest_speed = np.asarray(env_state.largest_torso_speed)
    largest_turning = np.asarray(env_state.largest_torso_angular_speed)
    cells = []
    for cell, setting in enumerate(settings):
        mine = table[cell_of_row == cell]
        pick = slice(cell * per_cell, (cell + 1) * per_cell)
        episodes, arrivals, respawned = len(mine), int(mine[:, 2].sum()), int(mine[:, 3].sum())
        cells.append({
            "learning_rate": setting[0], "ent_coef": setting[1], "int_coef": setting[2], "episodes": episodes,
            "arrivals": arrivals, "arrival_rate": arrivals / max(episodes, 1),
            "arrival_rate_without_respawned_episodes": arrivals / max(episodes - respawned, 1),
            "respawned_episodes": respawned, "respawns_per_10000_episodes": 1e4 * respawned / max(episodes, 1),
            "respawns_by_the_environment_counter": int(respawns_counter[pick].sum()),
            "mean_steps_of_arriving_episodes": float(mine[mine[:, 2] == 1, 1].mean()) if arrivals else None,
            "largest_torso_speed_after_any_physics_step": float(largest_speed[pick].max()),
            "largest_torso_angular_speed_after_any_physics_step": float(largest_turning[pick].max()),
            "episodes_with_observed_speed_over_15": int((mine[:, 5] > 15.0).sum()),
            "episodes_with_observed_speed_over_20": int((mine[:, 5] > 20.0).sum()),
            "median_of_largest_observed_speed_per_episode": float(np.median(mine[:, 5])) if episodes else None})
    total = {"episodes": int(len(table)), "arrivals": int(table[:, 2].sum()), "respawned_episodes": int(table[:, 3].sum()),
             "respawns_by_the_environment_counter": int(respawns_counter.sum()),
             "non_finite_or_guard_respawns_during_the_rollout": int((np.asarray(env_state.nan_count) - nan_count_before).sum()),
             "largest_torso_speed_after_any_physics_step": float(largest_speed.max()),
             "largest_torso_angular_speed_after_any_physics_step": float(largest_turning.max())}
    total["arrival_rate"] = total["arrivals"] / max(total["episodes"], 1)
    total["respawns_per_10000_episodes"] = 1e4 * total["respawned_episodes"] / max(total["episodes"], 1)
    summary = {
        "host": socket.gethostname(), "device": jax.local_devices()[0].device_kind,
        "measured_iso": datetime.now().astimezone().isoformat(), "setting": args.setting,
        "checkpoint": args.checkpoint.name, "checkpoint_iteration": int(state.step),
        "environment_steps_per_policy_at_the_checkpoint": int(state.step) * trainer.cfg.num_steps * N,
        "environment_config": asdict(env_cfg), "warp_model_integrator_number": int(env.model.opt.integrator),
        "policies": C, "environments_per_policy": N, "rollout_steps_per_environment": (args.steps // T) * T,
        "noise_seed": args.noise_seed, "configuration_fields": ["learning_rate", "ent_coef", "int_coef"],
        "per_configuration": cells, "total": total, "pool": env.overflow_report(env_state),
        "seconds_per_chunk_after_the_first_two": float(np.mean(seconds[2:] or seconds)), "chunk_steps": T,
        "environment_steps_per_second_after_the_first_two": B * T / float(np.mean(seconds[2:] or seconds)),
        "episode_table_columns": ["policy", "length", "arrived", "respawned", "largest_height", "largest_speed",
                                  "largest_turning"],
        "total_seconds": time.time() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + "_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(str(args.out) + "_episodes.npz", table=table.astype(np.float32))
    for row in cells:
        print(f"learning rate {row['learning_rate']:g}, weight {row['int_coef']:g}: episodes {row['episodes']}, arriving "
              f"{row['arrival_rate']:.3f}, respawned {row['respawned_episodes']} ({row['respawns_per_10000_episodes']:.1f} "
              f"per 10,000), largest speed {row['largest_torso_speed_after_any_physics_step']:.1f} m/s", flush=True)
    print(f"total: {json.dumps(total)}; wrote {args.out}_summary.json after {summary['total_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
