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
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from utilities.debug import print_or_wandb_log
from utilities.env_utils import (
    observation_to_grid,
    get_maze_map,
    select_fixed_cell,
    select_fixed_goal_top_left,
    select_fixed_goal_bottom_right,
)
from utilities.heatmap_utils import create_visit_count_heatmap
from utilities.intrinsic_replay_buffer import DictIntrinsicReplayBuffer


class FixedGoalWrapper(gym.Wrapper):
    """Wraps PointMaze to use a fixed goal on every reset, selected deterministically by seed."""

    def __init__(self, env, goal_cell):
        super().__init__(env)
        self.goal_cell = goal_cell  # (row, col) 0-based

    def reset(self, seed=None, options=None, **kwargs):
        opts = dict(options) if options else {}
        opts["goal_cell"] = [int(self.goal_cell[0]), int(self.goal_cell[1])]
        return self.env.reset(seed=seed, options=opts, **kwargs)


class FixedStartWrapper(gym.Wrapper):
    """Wraps PointMaze to use a fixed start cell on every reset (each seed corresponds to one fixed start)."""

    def __init__(self, env, start_cell):
        super().__init__(env)
        self.start_cell = start_cell  # (row, col) 0-based

    def reset(self, seed=None, options=None, **kwargs):
        opts = dict(options) if options else {}
        opts["reset_cell"] = [int(self.start_cell[0]), int(self.start_cell[1])]
        return self.env.reset(seed=seed, options=opts, **kwargs)


class RemoveGoalWrapper(gym.Wrapper):
    """Removes desired_goal (and achieved_goal) from observation. Use when goal is fixed."""

    def __init__(self, env, remove_keys=None):
        super().__init__(env)
        self.remove_keys = remove_keys or ["desired_goal", "achieved_goal"]
        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = {
                k: v for k, v in env.observation_space.spaces.items()
                if k not in self.remove_keys
            }
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = env.observation_space

    def _filter_obs(self, obs):
        if isinstance(obs, dict):
            return {k: v for k, v in obs.items() if k not in self.remove_keys}
        return obs

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self._filter_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info


class VisitCountWrapper(gym.Wrapper):
    """
    Wraps PointMaze to track visit counts per grid cell.
    Reads maze map from env. Maps (x,y) to (row,col), increments visit_counts on open cells.
    Optional: add intrinsic reward when beta > 0. Decay: intrinsic_decay_rate -2 = 1/sqrt(n), -1 = 1/n.
    For eval: pass count_map_ref=train_env.visit_counts and update_counts=False to reuse train counts
    without modifying them.
    """

    def __init__(self, env, beta=0.0, intrinsic_decay_rate=-0.5, count_map_ref=None, update_counts=True):
        super().__init__(env)
        self.beta = beta
        self.intrinsic_decay_rate = intrinsic_decay_rate
        self.update_counts = update_counts
        self.maze_map = get_maze_map(env)
        if self.maze_map is None:
            raise ValueError("Could not extract maze_map from environment")
        self.grid_rows, self.grid_cols = self.maze_map.shape
        if count_map_ref is not None:
            self.visit_counts = count_map_ref
        else:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

    def _state_to_grid(self, obs):
        """Map observation to 0-based (row, col)."""
        return observation_to_grid(obs, self.grid_rows, self.grid_cols)

    def _get_intrinsic_reward(self, row, col):
        """Intrinsic reward = 1/n^exponent before this visit; exponent = -decay_rate. Capped at 1.0 for unvisited."""
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            return 0.0
        if self.maze_map[row, col] != 0:
            return 0.0
        count = self.visit_counts[row, col]
        if count <= 0:
            return 1.0
        return min(1.0, pow(float(count), self.intrinsic_decay_rate))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        row, col = self._state_to_grid(obs)
        intrinsic = 0.0
        if self.beta != 0.0:
            intrinsic = self._get_intrinsic_reward(row, col)
        if self.update_counts and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:
                self.visit_counts[row, col] += 1
        total_reward = reward + self.beta * intrinsic
        info["intrinsic_reward"] = intrinsic
        info["extrinsic_reward"] = reward
        return obs, total_reward, terminated, truncated, info

    def get_visit_counts(self):
        return self.visit_counts.copy()

    def compute_intrinsic_reward(self, obs):
        """Compute intrinsic reward for obs using current visit counts. For replay buffer sample-time recomputation."""
        row, col = self._state_to_grid(obs)
        return self._get_intrinsic_reward(row, col)


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


class WandbEvalLoggingCallback(BaseCallback):
    """Eval at eval_freq, log to WandB. Log visit-count heatmap at same freq when use_wandb."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, visit_count_env=None, goal_cell=None, start_cell=None, run_name: str = "", verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.visit_count_env = visit_count_env
        self.goal_cell = goal_cell
        self.start_cell = start_cell
        self.run_name = run_name

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        # Heatmap at eval freq
        if self.visit_count_env is not None:
            try:
                import matplotlib.pyplot as plt
                step = self.num_timesteps
                name = f"{self.run_name}|step{step:07d}"
                fig = create_visit_count_heatmap(
                    self.visit_count_env.get_visit_counts(),
                    maze_map=self.visit_count_env.maze_map,
                    title=f"Train Visit Count ({name})",
                    goal_cell=self.goal_cell,
                    start_cell=self.start_cell,
                )
                if self.use_wandb and wandb.run:
                    wandb.log({
                        f"train_visit_count_heatmap/{name}": wandb.Image(fig),
                        "media/heatmap_latest": wandb.Image(fig),
                    }, step=step, commit=True)
                else:
                    img_dir = os.path.join("image", self.run_name)
                    os.makedirs(img_dir, exist_ok=True)
                    fig.savefig(os.path.join(img_dir, f"heatmap_step{step:07d}.png"))
                plt.close(fig)
            except Exception as e:
                if self.verbose > 0:
                    print(f"  Heatmap: {e}")
        # Custom eval loop to collect extrinsic, intrinsic, and total per episode
        episode_extrinsic = []
        episode_intrinsic = []
        episode_total = []
        episode_lengths = []
        for _ in range(self.n_eval_episodes):
            reset_out = self.eval_env.reset()
            obs = reset_out[0] if isinstance(reset_out, (list, tuple)) else reset_out
            done = False
            ep_ext, ep_int, ep_tot = 0.0, 0.0, 0.0
            ep_len = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_tot += float(rewards[0])
                ep_len += 1
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                beta = getattr(self.visit_count_env, "beta", 1.0) if self.visit_count_env is not None else 1.0
                if beta != 0:
                    if "extrinsic_reward" not in info:
                        raise KeyError("eval env step info must contain 'extrinsic_reward' when beta != 0")
                    if "intrinsic_reward" not in info:
                        raise KeyError("eval env step info must contain 'intrinsic_reward' when beta != 0")
                # When beta=0, info may be empty (no VisitCountWrapper): extrinsic = step reward, intrinsic = 0
                extrinsic = float(info.get("extrinsic_reward", rewards[0]))
                intrinsic = float(info.get("intrinsic_reward", 0.0))
                ep_ext += extrinsic
                ep_int += beta * intrinsic
                done = bool(dones[0])
            episode_extrinsic.append(ep_ext)
            episode_intrinsic.append(ep_int)
            episode_total.append(ep_tot)
            episode_lengths.append(ep_len)

        mean_extrinsic = float(np.mean(episode_extrinsic))
        mean_intrinsic = float(np.mean(episode_intrinsic))
        mean_total = float(np.mean(episode_total))
        mean_length = float(np.mean(episode_lengths))
        summary = collections.OrderedDict([
            ("step", self.num_timesteps),
            ("eval/mean_extrinsic_reward", mean_extrinsic),
            ("eval/mean_intrinsic_reward", mean_intrinsic),
            ("eval/mean_total_reward", mean_total),
            ("eval/mean_ep_length", mean_length),
        ])
        if self.visit_count_env is not None:
            visit_counts = self.visit_count_env.get_visit_counts()
            open_cells = (self.visit_count_env.maze_map == 0).sum()
            visited_cells = (visit_counts > 0).sum()
            total_visits = int(visit_counts.sum())
            summary["visit_counts/total_visits"] = total_visits
            summary["visit_counts/cells_visited"] = int(visited_cells)
            summary["visit_counts/open_cells"] = int(open_cells)
            summary["visit_counts/coverage_pct"] = 100.0 * visited_cells / max(1, open_cells)
        print_or_wandb_log(
            self.use_wandb,
            summary,
            f"Eval (step {self.num_timesteps})",
        )
        return True


def main():
    parser = argparse.ArgumentParser(description="Train SAC on PointMaze (07_reconstruction, self-contained)")
    parser.add_argument("--env_name", type=str, default="PointMaze_Large-v3", help="PointMaze env id")
    parser.add_argument("--a_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=100_0, help="Training steps")
    parser.add_argument("--eval_freq", type=int, default=5_0, help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--continuing_task", type=lambda x: x.lower() in ("true", "1", "yes"), default=False, nargs="?", const=True, help="If True, episode continues after reaching goal")
    parser.add_argument("--use_wandb", default=False, type=lambda x: x.lower() in ["true", "1", "yes"])
    parser.add_argument("--beta", type=float, default=0.01, help="Intrinsic reward coefficient; 0 = tracking only")
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

    base_env = gym.make(args.env_name, continuing_task=args.continuing_task, max_episode_steps=args.env_max_episode)
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
    visit_count_env = VisitCountWrapper(no_goal_env, beta=args.beta, intrinsic_decay_rate=args.intrinsic_decay_rate)
    monitored_env = Monitor(visit_count_env, filename=None)
    env = DummyVecEnv([lambda: monitored_env])
    env.seed(seed)
    env.reset()

    replay_buffer_class = DictIntrinsicReplayBuffer if args.beta > 0 else None
    replay_buffer_kwargs = (
        {"intrinsic_reward_fn": visit_count_env.compute_intrinsic_reward, "beta": args.beta}
        if args.beta > 0
        else None
    )

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
    eval_base = gym.make(args.env_name, continuing_task=args.continuing_task, max_episode_steps=args.env_max_episode)
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

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=wandb_eval_callback,
    )

    if not args.use_wandb:
        print("-------------Program Finished-------------")
    else:
        wandb.finish()


if __name__ == "__main__":
    main()
