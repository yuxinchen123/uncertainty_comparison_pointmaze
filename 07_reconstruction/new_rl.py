"""
Self-contained training script: SAC on PointMaze.

This folder (07_reconstruction) is independent of all other project folders.
No imports from 01-06; all logic is local or from standard packages.

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
from utilities.env_utils import observation_to_grid, get_maze_map


class VisitCountWrapper(gym.Wrapper):
    """
    Wraps PointMaze to track visit counts per grid cell.
    Reads maze map from env. Maps (x,y) to (row,col), increments visit_counts on open cells.
    Optional: add intrinsic reward 1/sqrt(count) when beta > 0.
    """

    def __init__(self, env, beta=0.0):
        super().__init__(env)
        self.beta = beta
        self.maze_map = get_maze_map(env)
        if self.maze_map is None:
            raise ValueError("Could not extract maze_map from environment")
        self.grid_rows, self.grid_cols = self.maze_map.shape
        self.visit_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

    def _state_to_grid(self, obs):
        """Map observation to 0-based (row, col)."""
        return observation_to_grid(obs, self.grid_rows, self.grid_cols)

    def _get_intrinsic_reward(self, row, col):
        """Intrinsic reward = 1/sqrt(visit_count) before this visit; capped at 1.0 for unvisited."""
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            return 0.0
        if self.maze_map[row, col] != 0:
            return 0.0
        count = self.visit_counts[row, col]
        if count <= 0:
            return 1.0
        return min(1.0, 1.0 / np.sqrt(count))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        row, col = self._state_to_grid(obs)
        intrinsic = 0.0
        if self.beta != 0.0:
            intrinsic = self._get_intrinsic_reward(row, col)
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:
                self.visit_counts[row, col] += 1
        total_reward = reward + self.beta * intrinsic
        info["intrinsic_reward"] = intrinsic
        info["extrinsic_reward"] = reward
        return obs, total_reward, terminated, truncated, info

    def get_visit_counts(self):
        return self.visit_counts.copy()


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
    parser.add_argument("--beta", type=float, default=0.0, help="Intrinsic reward coefficient (1/sqrt(visit_count)); 0 = tracking only")
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)

    base_env = gym.make(args.env_name, continuing_task=args.continuing_task)
    base_env.reset(seed=args.seed)
    visit_wrapper = VisitCountWrapper(base_env, beta=args.beta)
    monitored_env = Monitor(visit_wrapper, filename=None)
    env = DummyVecEnv([lambda: monitored_env])

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

    visit_counts = visit_wrapper.get_visit_counts()
    open_cells = (visit_wrapper.maze_map == 0).sum()
    visited_cells = (visit_counts > 0).sum()
    total_visits = int(visit_counts.sum())
    visit_summary = collections.OrderedDict([
        ("visit_counts/total_visits", total_visits),
        ("visit_counts/cells_visited", int(visited_cells)),
        ("visit_counts/open_cells", int(open_cells)),
        ("visit_counts/coverage_pct", 100.0 * visited_cells / max(1, open_cells)),
    ])
    print_or_wandb_log(args.use_wandb, visit_summary, "Visit counts")

    if not args.use_wandb:
        print("-------------Program Finished-------------")
    else:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
