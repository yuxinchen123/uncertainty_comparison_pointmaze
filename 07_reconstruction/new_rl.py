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
from stable_baselines3.common.callbacks import BaseCallback

from utilities.debug import print_or_wandb_log
from env_wrapper.point_maze_utils import (
    select_fixed_cell,
    select_fixed_goal_top_left,
    select_fixed_goal_bottom_right,
)
from utilities.heatmap_utils import create_visit_count_heatmap
from utilities.intrinsic_replay_buffer import DictIntrinsicReplayBuffer

from env_wrapper.point_maze_wrappers import (
    FixedGoalWrapper,
    FixedStartWrapper,
    RemoveGoalWrapper,
    VisitCountWrapper,
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


class TrainEpisodeStatsCallback(BaseCallback):
    """
    Logs average episode reward (extrinsic, intrinsic, total) and episode length
    from the training env at the same frequency as eval.
    Uses only the past n_eval_episodes training episodes for each log.
    Assumes Monitor wrapper is present. Tracks extrinsic/intrinsic from info dict.
    """

    def __init__(self, train_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, beta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.beta = beta
        self._monitor = None
        self._ep_extrinsic = 0.0
        self._ep_intrinsic = 0.0
        self._episode_extrinsics = []
        self._episode_intrinsics = []

    def _get_monitor(self):
        """Locate Monitor wrapper (assumed to exist)."""
        if self._monitor is not None:
            return self._monitor
        env = self.train_env.envs[0]
        while hasattr(env, "env"):
            if isinstance(env, Monitor):
                self._monitor = env
                return self._monitor
            env = env.env
        raise RuntimeError("Monitor wrapper not found in training env")

    def _on_step(self) -> bool:
        # Accumulate extrinsic and intrinsic from info on every step
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])
        if len(infos) > 0:
            info = infos[0]
            ext = float(info.get("extrinsic_reward", 0.0))
            intr_raw = float(info.get("intrinsic_reward", 0.0))
            intr = self.beta * intr_raw
            self._ep_extrinsic += ext
            self._ep_intrinsic += intr
            if dones[0]:
                self._episode_extrinsics.append(self._ep_extrinsic)
                self._episode_intrinsics.append(self._ep_intrinsic)
                self._ep_extrinsic = 0.0
                self._ep_intrinsic = 0.0

        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        monitor = self._get_monitor()
        rewards = monitor.get_episode_rewards()
        lengths = monitor.get_episode_lengths()
        n_completed = len(rewards)
        if n_completed == 0:
            return True

        n_window = min(self.n_eval_episodes, n_completed)
        window_rewards = rewards[-n_window:]
        window_lengths = lengths[-n_window:]
        window_extrinsics = self._episode_extrinsics[-n_window:]
        window_intrinsics = self._episode_intrinsics[-n_window:]

        mean_total = float(np.mean(window_rewards))
        mean_extrinsic = float(np.mean(window_extrinsics)) if window_extrinsics else 0.0
        mean_intrinsic = float(np.mean(window_intrinsics)) if window_intrinsics else 0.0
        mean_length = float(np.mean(window_lengths))

        summary = collections.OrderedDict([
            ("step", self.num_timesteps),
            ("train/mean_extrinsic_reward", mean_extrinsic),
            ("train/mean_intrinsic_reward", mean_intrinsic),
            ("train/mean_total_reward", mean_total),
            ("train/mean_episode_length", mean_length),
            ("train/n_episodes_averaged", n_window),
        ])
        print_or_wandb_log(
            self.use_wandb,
            summary,
            f"Train episode stats (step {self.num_timesteps})",
        )
        return True


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
