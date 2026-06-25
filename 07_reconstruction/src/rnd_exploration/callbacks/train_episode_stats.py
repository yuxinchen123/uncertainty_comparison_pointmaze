"""Train episode stats callback: log mean extrinsic/intrinsic/total reward and length."""
import collections

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rnd_exploration.common.debug import print_or_wandb_log


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
