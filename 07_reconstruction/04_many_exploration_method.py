"""
SAC on PointMaze with switchable intrinsic reward: RND, VisitCount, or EllipticalBonus.
Same env and training setup as 03_rnd_my_implementation.py; choice of method via --intrinsic_method.
"""
import argparse
import collections
import random
import wandb

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utilities.debug import print_or_wandb_log
from utilities.callbacks import TrainEpisodeStatsCallback, WandbEvalLoggingCallback
from env_wrapper.point_maze_utils import (
    select_fixed_cell,
    select_fixed_goal_top_left,
    select_fixed_goal_bottom_right,
)
from env_wrapper.point_maze_wrappers import (
    FixedGoalWrapper,
    FixedStartWrapper,
    RemoveGoalWrapper,
    TerminateOnTimeLimitWrapper,
    PositionVisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
)
from intrinsic.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
from intrinsic.intrinsic_method import RND, VisitCount, EllipticalBonus

# PointMaze flat obs after RemoveGoal + Flatten: (4,) = [x, y, vx, vy]
RND_POS_DIM = 2
INTRINSIC_METHODS = ("rnd", "visit_count", "elliptical")


def _args_to_run_name(args) -> str:
    parts = [
        getattr(args, "env_name", "env").replace("/", "-"),
        f"seed={getattr(args, 'a_seed', 0)}",
        f"goal={getattr(args, 'goal_position', 'top_left')}",
        f"beta={getattr(args, 'beta', 0)}",
        f"method={getattr(args, 'intrinsic_method', 'rnd')}",
        f"discount_factor={getattr(args, 'discount_factor', 0.99)}",
        f"env_max_episode={getattr(args, 'env_max_episode', 300)}",
    ]
    if getattr(args, "intrinsic_method", None) == "rnd":
        parts.extend([
            f"n_predictors={getattr(args, 'n_predictors', 5)}",
            f"linear_rnd={getattr(args, 'linear_rnd', False)}",
        ])
    elif getattr(args, "intrinsic_method", None) == "visit_count":
        parts.append(f"decay={getattr(args, 'intrinsic_decay_rate', -0.5)}")
    elif getattr(args, "intrinsic_method", None) == "elliptical":
        parts.extend([
            f"elliptical_feature_dim={getattr(args, 'elliptical_feature_dim', 128)}",
            f"elliptical_reg={getattr(args, 'elliptical_regularization', 1e-6)}",
        ])
    return "|".join(str(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Train SAC on PointMaze with RND, VisitCount, or EllipticalBonus (07_reconstruction)"
    )
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3")
    parser.add_argument("--a_seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--beta", type=float, default=0.01, help="Intrinsic reward coefficient")
    parser.add_argument(
        "--intrinsic_method",
        type=str,
        default="elliptical",
        choices=INTRINSIC_METHODS,
        help="Intrinsic reward model: rnd, visit_count, or elliptical",
    )
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--env_max_episode", type=int, default=300)
    parser.add_argument("--goal_position", type=str, default="top_left", choices=["top_left", "bottom_right", "random"])

    # RND
    parser.add_argument("--rnd_obs_norm", default=True, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--rnd_distance", type=str, default="mse", choices=["mse", "abs"])
    parser.add_argument("--rnd_input", type=str, default="position", choices=["position", "all"])
    parser.add_argument("--rnd_output_dim", type=int, default=128)
    parser.add_argument("--n_predictors", type=int, default=5)
    parser.add_argument("--beta_std", type=float, default=0.0)
    parser.add_argument("--linear_rnd", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])

    # VisitCount
    parser.add_argument("--intrinsic_decay_rate", type=float, default=-0.5, help="VisitCount decay: -0.5 = 1/sqrt(n), -1 = 1/n")

    # EllipticalBonus
    parser.add_argument("--elliptical_feature_dim", type=int, default=128)
    parser.add_argument("--elliptical_regularization", type=float, default=1e-6)

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(config=vars(args))
        for key in vars(args):
            if key in wandb.config:
                setattr(args, key, wandb.config[key])
    seed = args.a_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device_type = "gpu" if args.device == "cuda" else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    gym.register_envs(gymnasium_robotics)

    base_env = gym.make(
        args.env_name,
        continuing_task=True,
        reset_target=False,
        max_episode_steps=args.env_max_episode,
    )
    base_env = TerminateOnTimeLimitWrapper(base_env)
    if args.goal_position == "top_left":
        fixed_goal_cell = select_fixed_goal_top_left(base_env)
        fixed_start_cell = select_fixed_goal_bottom_right(base_env)
    elif args.goal_position == "bottom_right":
        fixed_goal_cell = select_fixed_goal_bottom_right(base_env)
        fixed_start_cell = select_fixed_goal_top_left(base_env)
    else:
        fixed_goal_cell = select_fixed_cell(base_env, seed)
        fixed_start_cell = select_fixed_cell(base_env, seed, exclude_cells=[fixed_goal_cell])
    manhattan_dist = abs(fixed_goal_cell[0] - fixed_start_cell[0]) + abs(fixed_goal_cell[1] - fixed_start_cell[1])
    print_or_wandb_log(
        args.use_wandb,
        collections.OrderedDict([
            ("device_type", device_type),
            ("start_goal/manhattan_distance", manhattan_dist),
            ("intrinsic_method", args.intrinsic_method),
        ]),
        "Setup",
    )
    print(f"Fixed goal cell: {fixed_goal_cell}, start cell: {fixed_start_cell}")
    print(f"Intrinsic method: {args.intrinsic_method}")

    start_env = FixedStartWrapper(base_env, fixed_start_cell)
    goal_env = FixedGoalWrapper(start_env, fixed_goal_cell)
    no_goal_env = RemoveGoalWrapper(goal_env)
    visit_count_env = PositionVisitCountWrapper(no_goal_env)
    flat_env = FlattenObservation(visit_count_env)

    full_obs_dim = int(np.prod(flat_env.observation_space.shape))
    obs_shape = (full_obs_dim,)
    replay_buffer_class = VectorIntrinsicReplayBuffer if args.beta > 0 else None
    intrinsic_model = None

    if args.beta > 0:
        if args.intrinsic_method == "rnd":
            rnd_obs_slice = (0, RND_POS_DIM) if args.rnd_input == "position" else None
            intrinsic_model = RND(
                obs_shape=obs_shape,
                output_dim=args.rnd_output_dim,
                lr=0.001,
                batch_size=256,
                device=args.device,
                use_obs_norm=args.rnd_obs_norm,
                distance=args.rnd_distance,
                obs_slice=rnd_obs_slice,
                n_predictors=args.n_predictors,
                beta_std=args.beta_std,
                linear_rnd=args.linear_rnd,
            )
            if args.rnd_obs_norm and getattr(intrinsic_model, "obs_rms", None) is not None:
                obs_buf = []
                for _ in range(200):
                    obs_buf.append(np.asarray(flat_env.observation_space.sample(), dtype=np.float32))
                obs_arr = np.stack(obs_buf, axis=0)
                if intrinsic_model.obs_slice is not None:
                    start, end = intrinsic_model.obs_slice
                    obs_arr = obs_arr[..., start:end]
                intrinsic_model.obs_rms.update(obs_arr)
        elif args.intrinsic_method == "visit_count":
            intrinsic_model = VisitCount(visit_count_env, args.intrinsic_decay_rate)
        elif args.intrinsic_method == "elliptical":
            action_dim = int(np.prod(flat_env.action_space.shape))
            intrinsic_model = EllipticalBonus(
                obs_shape=obs_shape,
                action_dim=action_dim,
                feature_dim=args.elliptical_feature_dim,
                device=args.device,
                regularization=args.elliptical_regularization,
            )
        else:
            raise ValueError(f"intrinsic_method must be one of {INTRINSIC_METHODS}; got {args.intrinsic_method!r}")

        replay_buffer_kwargs = {
            "beta": args.beta,
            "intrinsic_reward_model": intrinsic_model,
        }
    else:
        replay_buffer_kwargs = None

    intrinsic_env = ComputeIntrinsicRewardWrapper(flat_env, beta=args.beta, intrinsic_reward_model=intrinsic_model)
    monitored_env = Monitor(intrinsic_env, filename=None)
    env = DummyVecEnv([lambda: monitored_env])
    env.seed(seed)
    env.reset()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=0 if args.use_wandb else 1,
        seed=seed,
        device=args.device,
        gamma=args.discount_factor,
        tensorboard_log=None,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    eval_base = gym.make(
        args.env_name,
        continuing_task=True,
        reset_target=False,
        max_episode_steps=args.env_max_episode,
    )
    eval_base = TerminateOnTimeLimitWrapper(eval_base)
    eval_start_env = FixedStartWrapper(eval_base, fixed_start_cell)
    eval_goal_env = FixedGoalWrapper(eval_start_env, fixed_goal_cell)
    eval_no_goal_env = RemoveGoalWrapper(eval_goal_env)
    eval_visit_count_env = PositionVisitCountWrapper(
        eval_no_goal_env,
        count_map_ref=visit_count_env.visit_counts,
        update_counts=False,
    )
    eval_flat_env = FlattenObservation(eval_visit_count_env)
    eval_intrinsic_env = ComputeIntrinsicRewardWrapper(
        eval_flat_env, beta=args.beta, intrinsic_reward_model=intrinsic_model
    )
    eval_monitored = Monitor(eval_intrinsic_env, filename=None)
    eval_env = DummyVecEnv([lambda: eval_monitored])
    eval_env.seed(seed)
    eval_env.reset()

    run_name = _args_to_run_name(args)
    wandb_eval_callback = WandbEvalLoggingCallback(
        eval_env, args.eval_freq, args.n_eval_episodes, args.use_wandb,
        visit_count_env=visit_count_env, goal_cell=fixed_goal_cell, start_cell=fixed_start_cell, run_name=run_name,
        beta=args.beta,
    )
    train_stats_callback = TrainEpisodeStatsCallback(
        train_env=env, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
        use_wandb=args.use_wandb, beta=args.beta,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[wandb_eval_callback, train_stats_callback],
    )

    if not args.use_wandb:
        print("-------------Program Finished-------------")
    else:
        wandb.finish()


if __name__ == "__main__":
    main()
