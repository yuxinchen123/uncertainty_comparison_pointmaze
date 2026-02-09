"""
Self-contained training script: SAC on PointMaze.

This folder (07_reconstruction) is independent of all other project folders.
No imports from 01-06; all logic is local or from standard packages.

"""
import argparse
import os
import collections
import random
import wandb

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

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
from intrinsic.intrinsic_replay_buffer import IntrinsicReplayBuffer

from env_wrapper.point_maze_wrappers import (
    FixedGoalWrapper,
    FixedStartWrapper,
    RemoveGoalWrapper,
    VisitCountWrapper,
)

# RND from rllte: uses predictor vs target network distillation for intrinsic reward
from rllte.xplore.reward.rnd import RND

from intrinsic.RLeXplore_utilities import (
    make_fake_vec_env_for_rnd,
    make_rnd_intrinsic_reward_fn,
)


def _args_to_run_name(args) -> str:
    """Sanitized string from args for folder/image naming."""
    parts = [
        getattr(args, "env_name", "env").replace("/", "-"),
        f"seed={getattr(args, 'a_seed', 0)}",
        f"goal={getattr(args, 'goal_position', 'top_left')}",
        f"beta={getattr(args, 'beta', 0)}",
        f"decay={getattr(args, 'intrinsic_decay_rate', -0.5)}",
        f"discount_factor={getattr(args, 'discount_factor', 0.99)}",
        f"env_max_episode={getattr(args, 'env_max_episode', 300)}",
        # f"eval{getattr(args, 'eval_freq', 0)}",
        # f"tot{getattr(args, 'total_timesteps', 0)}",
        # f"n_eval{getattr(args, 'n_eval_episodes', 0)}",
        # getattr(args, "device", "cpu"),
        # f"cont{getattr(args, 'continuing_task', False)}",
    ]
    return "|".join(str(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="Train SAC on PointMaze (07_reconstruction, self-contained)")
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3", help="PointMaze env id")
    parser.add_argument("--a_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100_0, help="Training steps")
    parser.add_argument("--eval_freq", type=int, default=5_0, help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--beta", type=float, default=0, help="Intrinsic reward coefficient; 0 = tracking only")
    parser.add_argument("--intrinsic_decay_rate", type=float, default=-0.5, help="Intrinsic decay: -0.5 = 1/sqrt(n), -1 = 1/n")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--env_max_episode", type=int, default=300, help="Max episode length (steps)")
    parser.add_argument("--goal_position", type=str, default="top_left", choices=["top_left", "bottom_right", "random"], help="Fixed goal corner: top_left or bottom_right or random")
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(config=vars(args))
        for key in vars(args):
            if key in wandb.config:
                setattr(args, key, wandb.config[key])
    seed = args.a_seed

    if args.device == "cuda":
        if torch.cuda.is_available():
            actual_device = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cuda"
        else:
            actual_device = "cpu (cuda requested but not available)"
            args.device = "cpu"
    else:
        actual_device = "cpu"

    # Fix random generators for reproducibility
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
    device_type = "gpu" if args.device == "cuda" else "cpu"
    print_or_wandb_log(
        args.use_wandb,
        collections.OrderedDict([
            ("device_type", device_type),
            ("actual_device", actual_device),
            ("start_goal/manhattan_distance", manhattan_dist),
        ]),
        "Setup (device and start–goal)",
    )
    print(f"Fixed goal cell (a_seed={seed}): {fixed_goal_cell}")
    print(f"Fixed start cell (a_seed={seed}): {fixed_start_cell}")

    start_env = FixedStartWrapper(base_env, fixed_start_cell)
    goal_env = FixedGoalWrapper(start_env, fixed_goal_cell)
    no_goal_env = RemoveGoalWrapper(goal_env)
    # When using RND for intrinsic reward, env only tracks visits (beta=0); buffer adds beta * rnd.compute at sample time
    visit_count_env = VisitCountWrapper(no_goal_env, beta=0.0, intrinsic_decay_rate=args.intrinsic_decay_rate)
    monitored_env = Monitor(visit_count_env, filename=None)
    env = DummyVecEnv([lambda: monitored_env])
    env.seed(seed)
    env.reset()

    replay_buffer_class = IntrinsicReplayBuffer if args.beta > 0 else None
    if args.beta > 0:
        fake_vec_env = make_fake_vec_env_for_rnd(
            no_goal_env.observation_space,
            no_goal_env.action_space,
            no_goal_env,
            num_envs=1,
        )
        rnd_module = RND(
            envs=fake_vec_env,
            device=args.device,
            beta=1.0,
            kappa=0.0,
            gamma=None,
            rwd_norm_type="rms",
            obs_norm_type="rms",
            latent_dim=128,
            lr=0.001,
            batch_size=256,
            update_proportion=1.0,
            encoder_model="mnih",
            weight_init="orthogonal",
        )
        intrinsic_reward_fn = make_rnd_intrinsic_reward_fn(rnd_module, args.device)
        replay_buffer_kwargs = {
            "intrinsic_reward_fn": intrinsic_reward_fn,
            "beta": args.beta,
            "rnd_module": rnd_module,
        }
    else:
        replay_buffer_kwargs = None

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=0 if args.use_wandb else 1,
        seed=seed,
        device=args.device,
        gamma=args.discount_factor,
        tensorboard_log=None,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )

    # Eval env: same stack as train; reuse train count map for intrinsic reward but do not increment during eval
    eval_base = gym.make(
        args.env_name,
        continuing_task=True,
        reset_target=False,
        max_episode_steps=args.env_max_episode,
    )
    eval_start_env = FixedStartWrapper(eval_base, fixed_start_cell)
    eval_goal_env = FixedGoalWrapper(eval_start_env, fixed_goal_cell)
    eval_no_goal_env = RemoveGoalWrapper(eval_goal_env)
    eval_visit_count_env = VisitCountWrapper(
        eval_no_goal_env,
        beta=args.beta,
        intrinsic_decay_rate=args.intrinsic_decay_rate,
        count_map_ref=visit_count_env.visit_counts,
        update_counts=False,
    )
    eval_env = Monitor(eval_visit_count_env, filename=None)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env.seed(seed)
    eval_env.reset()

    run_name = _args_to_run_name(args)
    wandb_eval_callback = WandbEvalLoggingCallback(
        eval_env, args.eval_freq, args.n_eval_episodes, args.use_wandb,
        visit_count_env=visit_count_env, goal_cell=fixed_goal_cell, start_cell=fixed_start_cell, run_name=run_name,
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
