"""Train SAC on PointMaze with a switchable intrinsic-exploration bonus (RND / VisitCount /
EllipticalBonus / none), selected by ``--algorithm`` through the rnd_exploration.methods registry.

This is the single entry point (it replaces the old numbered drivers 01–04). The training loop, env
stack, SAC setup, and callbacks are unchanged from 04_many_exploration_method.py; the per-algorithm
if/elif dispatch is replaced by the registry factory ``build_intrinsic_model``.
"""
import argparse
import collections
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import wandb
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rnd_exploration.common.debug import print_or_wandb_log
from rnd_exploration.callbacks import (
    TrainEpisodeStatsCallback,
    WandbEvalLoggingCallback,
    DistanceLoggingCallback,
)
from rnd_exploration.envs.point_maze_utils import (
    select_fixed_cell,
    select_fixed_goal_bottom_left,
    select_fixed_goal_top_right,
)
from rnd_exploration.envs.point_maze_wrappers import (
    FixedGoalWrapper,
    FixedStartWrapper,
    RemoveGoalWrapper,
    TerminateOnTimeLimitWrapper,
    PositionVisitCountWrapper,
    PositionVelocityVisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
)
from rnd_exploration.buffers.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
from rnd_exploration.methods import ALGORITHM_NAMES, REGISTRY, EnvContext, build_intrinsic_model


@dataclass
class Config:
    """All training knobs (the union of 04's argparse flags). Defaults are 04's argparse defaults;
    the wandb sweep overrides them at run time (see parse_config)."""
    env_name: str = "PointMaze_Large-v3"
    a_seed: int = 1
    total_timesteps: int = 1000
    eval_freq: int = 500
    n_eval_episodes: int = 10
    n_eval_episodes_final: int = 11
    # standalone eval is OFF by default (the run-3.1.1 standard): the deterministic n_eval_episodes
    # rollout every eval_freq roughly doubles env steps, so we score on the training-episode reward
    # (TrainEpisodeStatsCallback) instead. eval_freq still drives the train-reward snapshot cadence and
    # the cheap visit-count coverage. Set --eval_standalone True to restore the rollout (e.g. run-2 repro).
    eval_standalone: bool = False
    device: str = "cuda"
    use_wandb: bool = False
    beta: float = 0.01
    algorithm: str = "rnd_linear_next_state"
    discount_factor: float = 0.99
    env_max_episode: int = 400
    goal_position: str = "bottom_left"
    rnd_obs_norm: bool = True
    rnd_distance: str = "mse"
    rnd_output_dim: int = 128
    n_predictors: int = 1
    # elliptical-bonus knobs (rnd_elliptical / rnd_elliptical_global); ignored by non-elliptical algos.
    elliptical_regularization: float = 1e-2          # ridge λ on the covariance diagonal
    elliptical_feature_normalization: str = "unit"   # ν_φ: "unit" (default) | "rms_unit" | "none"
    elliptical_update_timing: str = "sample"         # covariance update fires on "sample" | "add"
    elliptical_feature_input: str = "state_action"   # encoder input x: "state_action" (s,a) | "next_state" s'
    apply_termination_wrapper: bool = False
    # distance-to-GT computation is OFF by default: gridding the whole maze through the intrinsic model at every
    # eval is overhead, and most runs only need the reward curves. Set --log_distance True to log distance_to_gt/*.
    log_distance: bool = False
    # run-2 additions:
    g_algo_beta: str = ""               # "algorithm|beta" — when set, overrides algorithm + beta (one grid sweep)
    z_logging_mode: str = "wandb_full"  # "wandb_full" | "wandb_param_only" | "local" (z_ so the grid sweeps it innermost)
    local_log_dir: str = ""             # if set, write a per-run JSON (config + eval/train/distance history + runtime)
    run_id: int = -1                    # 0-based position of this run in its sweep ("i" of i/total); -1 = standalone run
    run_total: int = 0                  # sweep size ("total" of i/total); 0 = standalone -> descriptive (non-id) filename
    # performance switches (default off = original behavior; benchmarked under analysis/2026-06-25-run-profiling)
    opt_torch_reward: bool = False      # intrinsic buffer: combine reward in torch (no numpy round-trip)
    opt_polyak_foreach: bool = False    # bit-exact torch._foreach_ polyak target update (replaces SB3 zip_strict)
    sac_train_freq: int = 1             # SAC train_freq: env steps between gradient-update bursts
    sac_gradient_steps: int = 1         # SAC gradient_steps per burst (with train_freq=N, gradient_steps=N keeps total updates equal)


def _str2bool(x: str) -> bool:
    """Parse a string flag to bool (the sweep passes --use_wandb=True etc. as strings)."""
    return x.lower() in ["true", "1", "yes"]


def parse_config() -> Config:
    """Build a Config from argparse, then let the wandb sweep override every field (04's pattern)."""
    # one argparse flag per Config field; string-bool flags keep 04's lambda so --flag=True works
    parser = argparse.ArgumentParser(
        description="Train SAC on PointMaze with a switchable intrinsic bonus (rnd_exploration)"
    )
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3")
    parser.add_argument("--a_seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1000)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--n_eval_episodes_final", type=int, default=11,
                        help="Eval episodes at the final step (when num_timesteps >= total_timesteps)")
    parser.add_argument("--eval_standalone", default=False, type=_str2bool,
                        help="If true, run the deterministic standalone eval rollout at eval_freq. OFF by "
                             "default (run-3.1.1 standard): score on the training-episode reward to save compute.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_wandb", default=False, type=_str2bool)
    parser.add_argument("--beta", type=float, default=0.01, help="Intrinsic reward coefficient")
    parser.add_argument("--algorithm", type=str, default="rnd_linear_next_state",
                        choices=list(ALGORITHM_NAMES), help="Exploration algorithm")
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--env_max_episode", type=int, default=400)
    parser.add_argument("--goal_position", type=str, default="bottom_left",
                        choices=["bottom_left", "top_right", "random"])
    parser.add_argument("--rnd_obs_norm", default=True, type=_str2bool)
    parser.add_argument("--rnd_distance", type=str, default="mse", choices=["mse", "abs"])
    parser.add_argument("--rnd_output_dim", type=int, default=128)
    parser.add_argument("--n_predictors", type=int, default=1)
    parser.add_argument("--elliptical_regularization", type=float, default=1e-2,
                        help="Elliptical ridge λ on the covariance diagonal (rnd_elliptical[_global]).")
    parser.add_argument("--elliptical_feature_normalization", type=str, default="unit",
                        choices=["unit", "rms_unit", "none"], help="Elliptical feature normalization ν_φ.")
    parser.add_argument("--elliptical_update_timing", type=str, default="sample",
                        choices=["sample", "add"], help="When the elliptical covariance update fires.")
    parser.add_argument("--elliptical_feature_input", type=str, default="state_action",
                        choices=["state_action", "next_state"],
                        help="Elliptical encoder input x: (s,a) or next-state-only s'.")
    parser.add_argument("--apply_termination_wrapper", default=False, type=_str2bool,
                        help="If true, truncation -> termination (else SB3 timeout handling).")
    parser.add_argument("--log_distance", default=False, type=_str2bool,
                        help="If true, log distance_to_gt/* (grids the maze each eval). OFF by default.")
    parser.add_argument("--g_algo_beta", type=str, default="",
                        help="Combined 'algorithm|beta' (run-2 sweep): pins each algorithm to its beta in one grid sweep.")
    parser.add_argument("--z_logging_mode", type=str, default="wandb_full",
                        choices=["wandb_full", "wandb_param_only", "local"],
                        help="wandb_full = log metrics to wandb; wandb_param_only = wandb.init for params but log locally only; "
                             "local = no wandb at all, log locally (used by the local work-queue, use_wandb=False).")
    parser.add_argument("--local_log_dir", type=str, default="",
                        help="If set, write a per-run JSON (config + eval/train/distance history + runtime) under <dir>/<z_logging_mode>/.")
    parser.add_argument("--run_id", type=int, default=-1,
                        help="0-based position of this run in its sweep ('i' of i/total). Set by the local work queue; "
                             "drives the id-based per-run JSON filename. -1 (default) = standalone run -> descriptive name.")
    parser.add_argument("--run_total", type=int, default=0,
                        help="Sweep size ('total' of i/total). 0 (default) = standalone run.")
    # performance switches (default off = baseline)
    parser.add_argument("--opt_torch_reward", default=False, type=_str2bool,
                        help="Intrinsic buffer: combine reward in torch (no numpy round-trip).")
    parser.add_argument("--opt_polyak_foreach", default=False, type=_str2bool,
                        help="Bit-exact torch._foreach_ polyak target update (replaces SB3 zip_strict).")
    parser.add_argument("--sac_train_freq", type=int, default=1, help="SAC train_freq (env steps per update burst).")
    parser.add_argument("--sac_gradient_steps", type=int, default=1, help="SAC gradient_steps per burst.")

    # argparse -> Config dataclass
    args = parser.parse_args()
    cfg = Config(**{f: getattr(args, f) for f in Config.__dataclass_fields__})

    # wandb sweep overrides each field (keeps the sweep as the source of run config); WANDB_DIR (set
    # by the slurm launcher) makes wandb write its local run dir into the per-run train_runs folder
    if cfg.use_wandb:
        wandb.init(config=vars(cfg), dir=os.environ.get("WANDB_DIR"))
        for key in vars(cfg):
            if key in wandb.config:
                setattr(cfg, key, wandb.config[key])
    # run-2: a combined (algorithm|beta) sweep param pins each algorithm to its beta in ONE grid sweep
    # (wandb grid can't zip two axes). Split it and record the resolved values back to wandb.config so the
    # downstream analysis can still group by algorithm/beta.
    if cfg.g_algo_beta:
        algo, beta_str = cfg.g_algo_beta.split("|")
        cfg.algorithm = algo
        cfg.beta = float(beta_str)
        if cfg.use_wandb and wandb.run is not None:
            wandb.config.update({"algorithm": cfg.algorithm, "beta": cfg.beta}, allow_val_change=True)
    return cfg


def _run_name(cfg: Config) -> str:
    """The run's JSON filename stem. A sweep run (run_total > 0) is named by its id, so each run owns one
    short, sweep-sortable file; a standalone run (run_total == 0) keeps the descriptive pipe-delimited name."""
    # sweep run: id-based name. before: run_id=42, run_total=600 -> after: "042_of_600" (id zero-padded to
    # run_total's width so names sort in sweep order). The descriptive fields live inside the JSON, not the name.
    if cfg.run_total > 0:
        return f"{cfg.run_id:0{len(str(cfg.run_total))}d}_of_{cfg.run_total}"
    # standalone run (no sweep id): descriptive pipe-delimited name with every load-bearing knob spelled out
    parts = [
        f"algorithm={cfg.algorithm}",
        cfg.env_name.replace("/", "-"),
        f"seed={cfg.a_seed}",
        f"goal={cfg.goal_position}",
        f"beta={cfg.beta}",
        f"discount_factor={cfg.discount_factor}",
        f"env_max_episode={cfg.env_max_episode}",
        f"apply_termination_wrapper={cfg.apply_termination_wrapper}",
    ]
    return "|".join(str(p) for p in parts)


def make_base_env(cfg: Config) -> gym.Env:
    """Create the base PointMaze env (+ optional truncation->termination wrapper). gym.register_envs
    must already have run."""
    base_env = gym.make(
        cfg.env_name,
        continuing_task=True,
        reset_target=False,
        max_episode_steps=cfg.env_max_episode,
    )
    if cfg.apply_termination_wrapper:
        base_env = TerminateOnTimeLimitWrapper(base_env)
    return base_env


def select_cells(cfg: Config, base_env: gym.Env, seed: int):
    """Pick the fixed goal and start cells (bottom_left / top_right / random), exactly as 04."""
    if cfg.goal_position == "bottom_left":
        goal_cell = select_fixed_goal_bottom_left(base_env)
        start_cell = select_fixed_goal_top_right(base_env)
    elif cfg.goal_position == "top_right":
        goal_cell = select_fixed_goal_top_right(base_env)
        start_cell = select_fixed_goal_bottom_left(base_env)
    else:
        goal_cell = select_fixed_cell(base_env, seed)
        start_cell = select_fixed_cell(base_env, seed, exclude_cells=[goal_cell])
    return goal_cell, start_cell


def build_env_stack(cfg: Config, base_env, start_cell, goal_cell, count_map_refs=None, update_counts=True):
    """Build the wrapper stack FixedStart -> FixedGoal -> RemoveGoal -> PositionVisitCount ->
    PositionVelocityVisitCount -> Flatten. Returns (flat_env, position_wrapper, position_velocity_wrapper).

    One helper builds both the train and eval stacks (04 hand-copied them). For eval, pass
    count_map_refs=(train position counts, train position-velocity counts) and update_counts=False so
    eval reads the train counts without mutating them.
    """
    # fixed start/goal + drop the goal from the observation
    start_env = FixedStartWrapper(base_env, start_cell)
    goal_env = FixedGoalWrapper(start_env, goal_cell)
    no_goal_env = RemoveGoalWrapper(goal_env)
    # position wrapper feeds the heatmap; position-velocity wrapper feeds the distance-to-GT field
    if count_map_refs is None:
        position_wrapper = PositionVisitCountWrapper(no_goal_env)
        position_velocity_wrapper = PositionVelocityVisitCountWrapper(position_wrapper)
    else:
        position_ref, position_velocity_ref = count_map_refs
        position_wrapper = PositionVisitCountWrapper(
            no_goal_env, count_map_ref=position_ref, update_counts=update_counts)
        position_velocity_wrapper = PositionVelocityVisitCountWrapper(
            position_wrapper, count_map_ref=position_velocity_ref, update_counts=update_counts)
    # flatten Dict obs to a (4,) vector [x, y, vx, vy]
    flat_env = FlattenObservation(position_velocity_wrapper)
    return flat_env, position_wrapper, position_velocity_wrapper


def wrap_for_rollout(flat_env, cfg: Config, intrinsic_model, seed: int) -> DummyVecEnv:
    """Wrap a flat env for SB3 rollout: ComputeIntrinsicReward (logging only) -> Monitor -> DummyVecEnv,
    seeded and reset (the actual reward = ext + beta*int is formed in the replay buffer, not here)."""
    intrinsic_env = ComputeIntrinsicRewardWrapper(flat_env, beta=cfg.beta, intrinsic_reward_model=intrinsic_model)
    monitored_env = Monitor(intrinsic_env, filename=None)
    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env.seed(seed)
    vec_env.reset()
    return vec_env


def build_sac(cfg: Config, vec_env, intrinsic_model, seed: int) -> SAC:
    """Build SAC with the intrinsic replay buffer when beta>0, else the stock buffer (04's plumbing)."""
    # perf switch: install the bit-exact torch._foreach_ polyak target update over SB3's zip_strict loop
    if cfg.opt_polyak_foreach:
        from rnd_exploration.common.sb3_patches import patch_polyak_foreach
        patch_polyak_foreach()
    # beta>0 swaps in the buffer that recomputes reward = ext + beta*intrinsic at sample time
    if cfg.beta > 0:
        replay_buffer_class = VectorIntrinsicReplayBuffer
        replay_buffer_kwargs = {
            "beta": cfg.beta, "intrinsic_reward_model": intrinsic_model,
            "opt_torch_reward": cfg.opt_torch_reward,  # perf switch: torch-only reward combine
        }
    else:
        replay_buffer_class = None
        replay_buffer_kwargs = None
    return SAC(
        "MlpPolicy",
        vec_env,
        verbose=0 if cfg.use_wandb else 1,
        seed=seed,
        device=cfg.device,
        gamma=cfg.discount_factor,
        tensorboard_log=None,
        # train_freq/gradient_steps default to 1/1 (SB3 default) = baseline; raising both together batches updates
        train_freq=cfg.sac_train_freq,
        gradient_steps=cfg.sac_gradient_steps,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )


def _write_local_log(cfg: "Config", runtime_seconds: float, eval_history, distance_history, train_history,
                     train_episode_history) -> None:
    """Write THIS run's own JSON to <local_log_dir>/<z_logging_mode>/<run_name>.json.

    Logging design (see .claude/rules/run-id-and-logging.md):
    - One file per run (named by the run's id, run_name=NNN_of_TOTAL) -> no two runs ever write the same
      path, so there is no file-lock contention across the 512 concurrent workers.
    - Group-style, single write: every metric is accumulated in memory by its callback during training
      (eval_history / train_history / distance_history) and the whole record is flushed here in ONE
      file write at run end -- file access is slow, so we never write incrementally per eval step.
    The three history lists are the logging GROUPS; each row in a group is one eval-cadence snapshot."""
    # one file per run, under the logging-mode subdir; the dir is created lazily on first write
    out_dir = os.path.join(cfg.local_log_dir, cfg.z_logging_mode)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        # --- identity / config group (written once at the end) ---
        "run_id": cfg.run_id,                 # this run's position in the sweep ("i")
        "run_total": cfg.run_total,           # sweep size ("total"); run_id/run_total is the run's identity
        "algorithm": cfg.algorithm,
        "beta": cfg.beta,
        "a_seed": cfg.a_seed,
        "z_logging_mode": cfg.z_logging_mode,
        "total_timesteps": cfg.total_timesteps,
        "eval_freq": cfg.eval_freq,
        "eval_standalone": cfg.eval_standalone,   # whether the standalone eval rollout ran (OFF by default)
        "runtime_seconds": runtime_seconds,
        # --- metric groups (each a list of per-eval-cadence snapshots, accumulated then flushed here) ---
        "eval_history": eval_history,         # eval group: eval/mean_extrinsic_reward, visit_counts/*, ... (all algos)
        "train_history": train_history,       # train group: train/mean_extrinsic_reward over the past
                                              # n_eval_episodes training episodes, ... (all algos)
        "train_episode_history": train_episode_history,  # episode-level: one row per completed training episode
                                              # (run-4 convention) -> full first-to-last training trajectory
        "distance_history": distance_history, # distance group: distance_to_gt/* (only ALGORITHMS_NO_ACTION algos)
    }
    # elliptical-only knobs: recorded so the analysis can group runs by covariance rule (batch/global),
    # update timing (sample/add), encoder input (s,a vs next-state), and ridge λ. Omitted for non-elliptical
    # algorithms (which never read them) so their JSON stays free of irrelevant defaults.
    if REGISTRY[cfg.algorithm].kind == "elliptical":
        record["elliptical_regularization"] = cfg.elliptical_regularization
        record["elliptical_update_timing"] = cfg.elliptical_update_timing
        record["elliptical_feature_input"] = cfg.elliptical_feature_input
        record["elliptical_feature_normalization"] = cfg.elliptical_feature_normalization
    with open(os.path.join(out_dir, _run_name(cfg) + ".json"), "w") as f:
        json.dump(record, f)


def build_callbacks(cfg, train_vec, eval_vec, position_wrapper, position_velocity_wrapper,
                    intrinsic_model, goal_cell, start_cell, run_name, log_to_wandb):
    """Build the three training callbacks (eval + heatmap, train episode stats, distance-to-GT).
    log_to_wandb gates whether metrics are sent to wandb (False in the wandb_param_only mode)."""
    wandb_eval_callback = WandbEvalLoggingCallback(
        eval_vec, cfg.eval_freq, cfg.n_eval_episodes, cfg.use_wandb,
        visit_count_env=position_wrapper, goal_cell=goal_cell, start_cell=start_cell, run_name=run_name,
        beta=cfg.beta,
        total_timesteps=cfg.total_timesteps,
        n_eval_episodes_final=cfg.n_eval_episodes_final,
        eval_standalone=cfg.eval_standalone,
        log_to_wandb=log_to_wandb,
    )
    train_stats_callback = TrainEpisodeStatsCallback(
        train_env=train_vec, eval_freq=cfg.eval_freq, n_eval_episodes=cfg.n_eval_episodes,
        use_wandb=cfg.use_wandb, beta=cfg.beta, log_to_wandb=log_to_wandb,
    )
    distance_logging_callback = DistanceLoggingCallback(
        algorithm=cfg.algorithm,
        intrinsic_reward_model=intrinsic_model,
        visit_count_env_position_velocity=position_velocity_wrapper,
        eval_freq=cfg.eval_freq,
        use_wandb=cfg.use_wandb,
        enabled=cfg.log_distance,  # OFF by default; no-op (empty history) unless --log_distance True
        log_to_wandb=log_to_wandb,
    )
    return [wandb_eval_callback, train_stats_callback, distance_logging_callback]


def run(cfg: Config) -> None:
    """Run one training job for `cfg` (one wandb run / one (algorithm, beta, seed) point)."""
    # no_exploration forces beta=0 (no intrinsic model, stock buffer) — registry says builds_model=False
    if not REGISTRY[cfg.algorithm].builds_model:
        cfg.beta = 0
    seed = cfg.a_seed
    # wandb_param_only: wandb.init already ran (params fetched) but metrics stay local (log_to_wandb=False)
    log_to_wandb = cfg.use_wandb and (cfg.z_logging_mode != "wandb_param_only")
    # runtime clock: from here (params in hand, training about to set up + run) to program end
    t_start = time.time()

    # resolve device with cuda->cpu fallback
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    device_type = "gpu" if cfg.device == "cuda" else "cpu"

    # seed every RNG together (random / numpy / torch / cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    gym.register_envs(gymnasium_robotics)

    # base env is shared by cell selection and the train stack (matches 04's object identity)
    base_env = make_base_env(cfg)
    goal_cell, start_cell = select_cells(cfg, base_env, seed)
    manhattan_dist = abs(goal_cell[0] - start_cell[0]) + abs(goal_cell[1] - start_cell[1])
    log_dict = collections.OrderedDict([
        ("device_type", device_type),
        ("start_goal/manhattan_distance", manhattan_dist),
        ("algorithm", cfg.algorithm),
        ("apply_termination_wrapper", cfg.apply_termination_wrapper),
    ])
    print_or_wandb_log(cfg.use_wandb, log_dict, "Setup")
    print(f"Fixed goal cell: {goal_cell}, start cell: {start_cell}")
    print(f"algorithm={cfg.algorithm}")

    # train env stack + spaces
    train_flat, train_position, train_position_velocity = build_env_stack(cfg, base_env, start_cell, goal_cell)
    full_obs_dim = int(np.prod(train_flat.observation_space.shape))
    obs_shape = (full_obs_dim,)
    action_dim = int(np.prod(train_flat.action_space.shape))

    # build the intrinsic model from the registry (only when beta>0, exactly as 04)
    intrinsic_model = None
    if cfg.beta > 0:
        ctx = EnvContext(
            obs_shape=obs_shape,
            action_dim=action_dim,
            observation_space=train_flat.observation_space,
            action_space=train_flat.action_space,
            position_wrapper=train_position,
            position_velocity_wrapper=train_position_velocity,
        )
        intrinsic_model = build_intrinsic_model(cfg.algorithm, cfg, ctx)

    train_vec = wrap_for_rollout(train_flat, cfg, intrinsic_model, seed)
    model = build_sac(cfg, train_vec, intrinsic_model, seed)

    # eval env stack reuses the train visit counts without updating them
    eval_base = make_base_env(cfg)
    eval_flat, _, _ = build_env_stack(
        cfg, eval_base, start_cell, goal_cell,
        count_map_refs=(train_position.visit_counts, train_position_velocity.visit_counts),
        update_counts=False,
    )
    eval_vec = wrap_for_rollout(eval_flat, cfg, intrinsic_model, seed)

    callbacks = build_callbacks(
        cfg, train_vec, eval_vec, train_position, train_position_velocity,
        intrinsic_model, goal_cell, start_cell, _run_name(cfg), log_to_wandb,
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    # runtime = setup + training (wandb param-fetch wait is excluded — t_start is after params)
    runtime_seconds = time.time() - t_start
    if log_to_wandb:
        wandb.log({"runtime_seconds": runtime_seconds})
    if cfg.local_log_dir:
        # callbacks = [eval, train-episode-stats, distance]; save all three histories
        _write_local_log(cfg, runtime_seconds, callbacks[0].history, callbacks[2].history, callbacks[1].history,
                         callbacks[1].episode_history)
    print(f"runtime_seconds={runtime_seconds:.2f} mode={cfg.z_logging_mode}")

    if not cfg.use_wandb:
        print("-------------Program Finished-------------")
    else:
        wandb.finish()


def main():
    """Parse config and run one job."""
    run(parse_config())


if __name__ == "__main__":
    main()
