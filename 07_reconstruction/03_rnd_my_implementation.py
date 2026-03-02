"""
Self-contained training script: SAC on PointMaze with custom MyRND and VectorIntrinsicReplayBuffer.
Same overall tasks as 02_rnd_rlexplore.py; FlattenObservation is applied after VisitCountWrapper.
Uses MlpPolicy (Box obs) and ReplayBuffer-based VectorIntrinsicReplayBuffer.
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
    VisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
)
from intrinsic.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
from intrinsic.my_rnd import MyRND

# PointMaze flat obs after RemoveGoal + Flatten: (4,) = [x, y, vx, vy]. RND uses position only.
RND_POS_DIM = 2


def _args_to_run_name(args) -> str:
    """Sanitized string from args for folder/image naming."""
    parts = [
        getattr(args, "env_name", "env").replace("/", "-"),
        f"seed={getattr(args, 'a_seed', 0)}",
        f"goal={getattr(args, 'goal_position', 'top_left')}",
        f"beta={getattr(args, 'beta', 0)}",
        f"n_predictors={getattr(args, 'n_predictors', 5)}",
        f"beta_std={getattr(args, 'beta_std', 0)}",
        f"rnd_obs_norm={getattr(args, 'rnd_obs_norm', False)}",
        f"rnd_distance={getattr(args, 'rnd_distance', 'mse')}",
        f"rnd_input={getattr(args, 'rnd_input', 'position')}",
        f"rnd_output_dim={getattr(args, 'rnd_output_dim', 128)}",
        f"intrinsic_method={getattr(args, 'intrinsic_method', 'rnd')}",
        f"discount_factor={getattr(args, 'discount_factor', 0.99)}",
        f"env_max_episode={getattr(args, 'env_max_episode', 300)}",
    ]
    return "|".join(str(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="Train SAC on PointMaze (07_reconstruction, my RND)")
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3", help="PointMaze env id")
    parser.add_argument("--a_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100_0, help="Training steps")
    parser.add_argument("--eval_freq", type=int, default=5_0, help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--beta", type=float, default=0.01, help="Intrinsic reward coefficient")
    parser.add_argument("--intrinsic_method", default=0, help="intrinsic_method")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--env_max_episode", type=int, default=300, help="Max episode length (steps)")
    parser.add_argument("--goal_position", type=str, default="top_left", choices=["top_left", "bottom_right", "random"], help="Fixed goal corner")
    parser.add_argument("--rnd_obs_norm", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="Use RunningMeanStd observation normalization for RND")
    parser.add_argument("--rnd_distance", type=str, default="mse", choices=["mse", "abs"], help="Distance metric for RND (intrinsic + predictor loss)")
    parser.add_argument("--rnd_input", type=str, default="position", choices=["position", "all"], help="RND input: 'position' (pos only) or 'all' (full obs)")
    parser.add_argument("--rnd_output_dim", type=int, default=128, help="RND predictor/target output dimension")
    parser.add_argument("--n_predictors", type=int, default=5, help="Number of ensemble predictor networks in MyRND")
    parser.add_argument("--beta_std", type=float, default=0.0, help="Scale for std-of-ensemble-distances term in intrinsic reward")
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
    visit_count_env = VisitCountWrapper(no_goal_env)
    flat_env = FlattenObservation(visit_count_env)

    replay_buffer_class = VectorIntrinsicReplayBuffer if args.beta > 0 else None
    my_rnd = None
    if args.beta > 0:
        full_obs_dim = int(np.prod(flat_env.observation_space.shape))
        obs_shape = (full_obs_dim,)
        rnd_obs_slice = (0, RND_POS_DIM) if args.rnd_input == "position" else None
        my_rnd = MyRND(
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
        )
        # Pre-init for RND observation normalization using random samples (position only when obs_slice is set)
        if args.rnd_obs_norm and getattr(my_rnd, "obs_rms", None) is not None:
            n_init = 200  # small number of batches for RMS initialization
            obs_buf = []
            for _ in range(n_init):
                sample = flat_env.observation_space.sample()
                obs_buf.append(np.asarray(sample, dtype=np.float32))
            obs_arr = np.stack(obs_buf, axis=0)
            if my_rnd.obs_slice is not None:
                start, end = my_rnd.obs_slice
                obs_arr = obs_arr[..., start:end]
            my_rnd.obs_rms.update(obs_arr)

        replay_buffer_kwargs = {
            "beta": args.beta,
            "rnd_module": my_rnd,
        }
    else:
        replay_buffer_kwargs = None

    intrinsic_env = ComputeIntrinsicRewardWrapper(flat_env, beta=args.beta, rnd_module=my_rnd)
    monitored_env = Monitor(intrinsic_env, filename=None)
    # Training environment (wrapped and vectorized) used by SAC for learning
    env = DummyVecEnv([lambda: monitored_env])
    # SB3 VecEnv.seed() stores seeds applied at the next reset() (Gymnasium 0.26+); order is correct.
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

    # Separate evaluation environment (wrapped and vectorized) used only for periodic evaluation
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
        count_map_ref=visit_count_env.visit_counts,
        update_counts=False,
    )
    eval_flat_env = FlattenObservation(eval_visit_count_env)
    eval_intrinsic_env = ComputeIntrinsicRewardWrapper(eval_flat_env, beta=args.beta, rnd_module=my_rnd)
    eval_monitored = Monitor(eval_intrinsic_env, filename=None)
    eval_env = DummyVecEnv([lambda: eval_monitored])
    # SB3 VecEnv: seed stored and applied at next reset.
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
