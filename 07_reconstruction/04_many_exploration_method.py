"""
SAC on PointMaze with switchable intrinsic reward: RND, VisitCount, or EllipticalBonus.
Same env and training setup as 03_rnd_my_implementation.py; choice of exploration via --algorithm.
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
from utilities.callbacks import TrainEpisodeStatsCallback, WandbEvalLoggingCallback, DistanceLoggingCallback
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
    PositionVelocityVisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
)
from intrinsic.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
from intrinsic.intrinsic_method import RND, VisitCount, EllipticalBonus
from distance_to_GT.algorithm_vector import ALGORITHM_NAMES

# PointMaze flat obs after RemoveGoal + Flatten: (4,) = [x, y, vx, vy]
RND_POS_DIM = 2


def _algorithm_to_config(algorithm: str):
    """
    Map ALGORITHM_NAME to (intrinsic_method, rnd_feature, linear_rnd).
    no_exploration: beta=0, no intrinsic model.
    """
    if algorithm == "no_exploration":
        return ("none", None, False)
    if algorithm == "gt_position":
        return ("visit_count", None, False)
    if algorithm == "gt_position_velocity":
        return ("visit_count", None, False)
    if algorithm == "rnd_next_state":
        return ("rnd", "rnd_next_state", False)
    if algorithm == "rnd_next_state_position_only":
        return ("rnd", "rnd_next_state_position_only", False)
    if algorithm == "rnd_state":
        return ("rnd", "rnd_state", False)
    if algorithm == "rnd_state_action":
        return ("rnd", "rnd_state_action", False)
    if algorithm == "rnd_state_action_next_state":
        return ("rnd", "rnd_state_action_next_state", False)
    if algorithm == "rnd_linear_next_state":
        return ("rnd", "rnd_next_state", True)
    if algorithm == "rnd_elliptical":
        return ("elliptical", None, False)
    raise ValueError(f"algorithm must be one of {ALGORITHM_NAMES}; got {algorithm!r}")


def _args_to_run_name(args) -> str:
    parts = [
        f"algorithm={args.algorithm}",
        getattr(args, "env_name", "env").replace("/", "-"),
        f"seed={getattr(args, 'a_seed', 0)}",
        f"goal={getattr(args, 'goal_position', 'top_left')}",
        f"beta={getattr(args, 'beta', 0)}",
        f"discount_factor={getattr(args, 'discount_factor', 0.99)}",
        f"env_max_episode={getattr(args, 'env_max_episode', 300)}",
    ]
    return "|".join(str(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Train SAC on PointMaze with RND, VisitCount, or EllipticalBonus (07_reconstruction)"
    )
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3")
    parser.add_argument("--a_seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_0)
    parser.add_argument("--eval_freq", type=int, default=5_00)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--n_eval_episodes_final", type=int, default=11, help="Eval episodes at final step (when num_timesteps >= total_timesteps)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--beta", type=float, default=0.01, help="Intrinsic reward coefficient")
    parser.add_argument("--algorithm", type=str, default="gt_position_velocity", choices=list(ALGORITHM_NAMES), help="Exploration algorithm")
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--env_max_episode", type=int, default=400)
    parser.add_argument("--goal_position", type=str, default="top_left", choices=["top_left", "bottom_right", "random"])

    # RND (only args used in 04_wandb_sweep or not class-default)
    parser.add_argument("--rnd_obs_norm", default=True, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--rnd_distance", type=str, default="mse", choices=["mse", "abs"])
    parser.add_argument("--rnd_output_dim", type=int, default=128)
    parser.add_argument("--n_predictors", type=int, default=1)

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(config=vars(args))
        for key in vars(args):
            if key in wandb.config:
                setattr(args, key, wandb.config[key])

    method, rnd_feature, linear_rnd = _algorithm_to_config(args.algorithm)
    args.intrinsic_method = method  # internal use only (rnd / visit_count / elliptical / none)
    args._rnd_feature = rnd_feature
    args._linear_rnd = linear_rnd
    if args.algorithm == "no_exploration":
        args.beta = 0
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
    log_dict = collections.OrderedDict([
        ("device_type", device_type),
        ("start_goal/manhattan_distance", manhattan_dist),
        ("algorithm", args.algorithm),
    ])
    print_or_wandb_log(args.use_wandb, log_dict, "Setup")
    print(f"Fixed goal cell: {fixed_goal_cell}, start cell: {fixed_start_cell}")
    print(f"algorithm={args.algorithm}")

    start_env = FixedStartWrapper(base_env, fixed_start_cell)
    goal_env = FixedGoalWrapper(start_env, fixed_goal_cell)
    no_goal_env = RemoveGoalWrapper(goal_env)
    # Always use both wrappers: position for heatmap, position_velocity for distance-to-GT (GT = gt_position_velocity visit count vector).
    visit_count_env_position = PositionVisitCountWrapper(no_goal_env)
    visit_count_env_position_velocity = PositionVelocityVisitCountWrapper(visit_count_env_position)
    # Intrinsic reward uses one of them depending on algorithm; heatmap always uses position.
    flat_env = FlattenObservation(visit_count_env_position_velocity)

    full_obs_dim = int(np.prod(flat_env.observation_space.shape))
    obs_shape = (full_obs_dim,)
    replay_buffer_class = VectorIntrinsicReplayBuffer if args.beta > 0 else None
    intrinsic_model = None

    if args.beta > 0:
        if args.intrinsic_method == "rnd":
            action_dim = int(np.prod(flat_env.action_space.shape))
            intrinsic_model = RND(
                obs_shape=obs_shape,
                output_dim=args.rnd_output_dim,
                lr=0.001,
                batch_size=256,
                device=args.device,
                use_obs_norm=args.rnd_obs_norm,
                distance=args.rnd_distance,
                n_predictors=args.n_predictors,
                beta_std=0.0,
                linear_rnd=args._linear_rnd,
                feature=args._rnd_feature,
                action_dim=action_dim,
            )
            if args.rnd_obs_norm and getattr(intrinsic_model, "obs_rms", None) is not None:
                # Init obs_rms with RND input shape (obs only, or obs+action, or obs+action+next_obs)
                obs_buf = []
                next_buf = []
                act_buf = []
                for _ in range(200):
                    obs = np.asarray(flat_env.observation_space.sample(), dtype=np.float32)
                    next_obs = np.asarray(flat_env.observation_space.sample(), dtype=np.float32)
                    action = np.asarray(flat_env.action_space.sample(), dtype=np.float32)
                    obs_buf.append(obs)
                    next_buf.append(next_obs)
                    act_buf.append(action)
                samples = {
                    "observations": np.stack(obs_buf, axis=0),
                    "next_observations": np.stack(next_buf, axis=0),
                    "actions": np.stack(act_buf, axis=0),
                }
                x = intrinsic_model._get_feature_tensor(samples)
                intrinsic_model.obs_rms.update(x.detach().cpu().numpy())
        elif args.intrinsic_method == "visit_count":
            intrinsic_model = VisitCount(
                visit_count_env_position_velocity if args.algorithm == "gt_position_velocity" else visit_count_env_position,
            )
        elif args.intrinsic_method == "elliptical":
            action_dim = int(np.prod(flat_env.action_space.shape))
            intrinsic_model = EllipticalBonus(
                obs_shape=obs_shape,
                action_dim=action_dim,
                feature_dim=128,
                device=args.device,
                regularization=1e-6,
            )
        else:
            raise ValueError(f"unexpected algorithm {args.algorithm!r}")

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
    eval_visit_count_position = PositionVisitCountWrapper(
        eval_no_goal_env,
        count_map_ref=visit_count_env_position.visit_counts,
        update_counts=False,
    )
    eval_visit_count_position_velocity = PositionVelocityVisitCountWrapper(
        eval_visit_count_position,
        count_map_ref=visit_count_env_position_velocity.visit_counts,
        update_counts=False,
    )
    eval_flat_env = FlattenObservation(eval_visit_count_position_velocity)
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
        visit_count_env=visit_count_env_position, goal_cell=fixed_goal_cell, start_cell=fixed_start_cell, run_name=run_name,
        beta=args.beta,
        total_timesteps=args.total_timesteps,
        n_eval_episodes_final=args.n_eval_episodes_final,
    )
    train_stats_callback = TrainEpisodeStatsCallback(
        train_env=env, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
        use_wandb=args.use_wandb, beta=args.beta,
    )
    distance_logging_callback = DistanceLoggingCallback(
        algorithm=args.algorithm,
        intrinsic_reward_model=intrinsic_model,
        visit_count_env_position_velocity=visit_count_env_position_velocity,
        eval_freq=args.eval_freq,
        use_wandb=args.use_wandb,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[wandb_eval_callback, train_stats_callback, distance_logging_callback],
    )

    if not args.use_wandb:
        print("-------------Program Finished-------------")
    else:
        wandb.finish()


if __name__ == "__main__":
    main()
