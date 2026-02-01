"""
Self-contained training script: SAC on PointMaze.

This folder (07_reconstruction) is independent of all other project folders.
No imports from 01-06; all logic is local or from standard packages.

No files or folders are written; results are printed to the CLI.
"""
import argparse
import collections

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from utilities.debug import print_or_wandb_log


class WandbEvalLoggingCallback(BaseCallback):
    """Run evaluate_policy at eval_freq and log results to WandB when use_wandb is True."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        use_wandb: bool,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))
        std_length = float(np.std(episode_lengths))
        summary = collections.OrderedDict([
            ("step", self.num_timesteps),
            ("eval/mean_reward", mean_reward),
            ("eval/std_reward", std_reward),
            ("eval/mean_ep_length", mean_length),
            ("eval/std_ep_length", std_length),
        ])
        print_or_wandb_log(
            self.use_wandb,
            summary,
            f"Eval (step {self.num_timesteps})",
        )
        return True


def main():
    parser = argparse.ArgumentParser(description="Train SAC on PointMaze (07_reconstruction, self-contained)")
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3", help="PointMaze env id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100_0, help="Training steps")
    parser.add_argument("--eval_freq", type=int, default=5_0, help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--continuing_task", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, nargs="?", const=True, help="If True, episode continues after reaching goal")
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)

    env = gym.make(args.env_name, continuing_task=args.continuing_task)
    env.reset(seed=args.seed)
    env = Monitor(env, filename=None)
    env = DummyVecEnv([lambda: env])

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=None,
    )

    eval_env = gym.make(args.env_name, continuing_task=args.continuing_task)
    eval_env.reset(seed=args.seed + 1)
    eval_env = Monitor(eval_env, filename=None)
    eval_env = DummyVecEnv([lambda: eval_env])

    wandb_eval_callback = WandbEvalLoggingCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_wandb=args.use_wandb,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=wandb_eval_callback,
    )

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    std_length = float(np.std(episode_lengths))

    comparison_summary = collections.OrderedDict([
        ("eval_mean_reward", mean_reward),
        ("eval_std_reward", std_reward),
        ("eval_mean_length", mean_length),
        ("eval_std_length", std_length),
        ("eval_episodes", args.n_eval_episodes),
        ("total_timesteps", args.total_timesteps),
    ])

    print_or_wandb_log(args.use_wandb, comparison_summary, "Results (final evaluation)")

    if not args.use_wandb:
        print("-------------Program Finished-------------")
    else:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
