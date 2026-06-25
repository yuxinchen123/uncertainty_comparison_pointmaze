"""Train SAC on PointMaze with a switchable intrinsic-exploration bonus (RND / VisitCount /
EllipticalBonus / none), selected by ``--algorithm`` through the rnd_exploration.methods registry.

This is the single entry point (it replaces the old numbered drivers 01–04). The training loop, env
stack, SAC setup, and callbacks are unchanged from 04_many_exploration_method.py; the per-algorithm
if/elif dispatch is replaced by the registry factory ``build_intrinsic_model``.
"""
import argparse
import collections
import os
import random
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
    apply_termination_wrapper: bool = False


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
    parser.add_argument("--apply_termination_wrapper", default=False, type=_str2bool,
                        help="If true, truncation -> termination (else SB3 timeout handling).")

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
    return cfg


def _run_name(cfg: Config) -> str:
    """Pipe-delimited run name (identical fields/order to 04's _args_to_run_name, so existing
    image/<run_name>/ folders and analysis joins keep matching)."""
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
    # beta>0 swaps in the buffer that recomputes reward = ext + beta*intrinsic at sample time
    if cfg.beta > 0:
        replay_buffer_class = VectorIntrinsicReplayBuffer
        replay_buffer_kwargs = {"beta": cfg.beta, "intrinsic_reward_model": intrinsic_model}
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
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )


def build_callbacks(cfg, train_vec, eval_vec, position_wrapper, position_velocity_wrapper,
                    intrinsic_model, goal_cell, start_cell, run_name):
    """Build the three training callbacks (eval + heatmap, train episode stats, distance-to-GT)."""
    wandb_eval_callback = WandbEvalLoggingCallback(
        eval_vec, cfg.eval_freq, cfg.n_eval_episodes, cfg.use_wandb,
        visit_count_env=position_wrapper, goal_cell=goal_cell, start_cell=start_cell, run_name=run_name,
        beta=cfg.beta,
        total_timesteps=cfg.total_timesteps,
        n_eval_episodes_final=cfg.n_eval_episodes_final,
    )
    train_stats_callback = TrainEpisodeStatsCallback(
        train_env=train_vec, eval_freq=cfg.eval_freq, n_eval_episodes=cfg.n_eval_episodes,
        use_wandb=cfg.use_wandb, beta=cfg.beta,
    )
    distance_logging_callback = DistanceLoggingCallback(
        algorithm=cfg.algorithm,
        intrinsic_reward_model=intrinsic_model,
        visit_count_env_position_velocity=position_velocity_wrapper,
        eval_freq=cfg.eval_freq,
        use_wandb=cfg.use_wandb,
    )
    return [wandb_eval_callback, train_stats_callback, distance_logging_callback]


def run(cfg: Config) -> None:
    """Run one training job for `cfg` (one wandb run / one (algorithm, beta, seed) point)."""
    # no_exploration forces beta=0 (no intrinsic model, stock buffer) — registry says builds_model=False
    if not REGISTRY[cfg.algorithm].builds_model:
        cfg.beta = 0
    seed = cfg.a_seed

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
        intrinsic_model, goal_cell, start_cell, _run_name(cfg),
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    if not cfg.use_wandb:
        print("-------------Program Finished-------------")
    else:
        wandb.finish()


def main():
    """Parse config and run one job."""
    run(parse_config())


if __name__ == "__main__":
    main()
