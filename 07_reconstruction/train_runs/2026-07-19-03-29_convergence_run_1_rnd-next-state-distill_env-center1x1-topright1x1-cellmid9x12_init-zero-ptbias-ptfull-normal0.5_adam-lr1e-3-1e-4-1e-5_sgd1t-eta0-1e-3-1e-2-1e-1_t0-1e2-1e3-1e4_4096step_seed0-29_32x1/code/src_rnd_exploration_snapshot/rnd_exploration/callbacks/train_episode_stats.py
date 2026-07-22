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

    def __init__(self, train_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, beta: float = 0.0, log_to_wandb: bool = None, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        # log_to_wandb: send train stats to wandb only in full mode (see WandbEvalLoggingCallback)
        self.log_to_wandb = use_wandb if log_to_wandb is None else log_to_wandb
        self.history = []            # per-eval-cadence mean over the past n_eval_episodes (the live summary)
        # NEW (run-4 logging convention, run-id-and-logging.md): one entry per COMPLETED training episode, so
        # the full first-to-last training-episode trajectory is saved (episode-level, not step-level). The
        # per-eval training curve = mean over the past n_eval_episodes of these, computed in the analysis.
        self.episode_history = []
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
                # record this completed episode (episode-level log): the Monitor holds its total reward +
                # length; we hold the accumulated extrinsic/intrinsic. before: episode just ended; after: one
                # row appended with the step it ended at -> the full episode trajectory is reconstructable.
                monitor = self._get_monitor()
                self.episode_history.append({
                    "step": int(self.num_timesteps),
                    "train/extrinsic_reward": float(self._ep_extrinsic),
                    "train/intrinsic_reward": float(self._ep_intrinsic),
                    "train/total_reward": float(monitor.get_episode_rewards()[-1]),
                    "train/episode_length": int(monitor.get_episode_lengths()[-1]),
                })
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
        self.history.append(dict(summary))
        # wandb mode still logs every snapshot; local mode prints the block only when verbose>0
        # (default silent — the per-run JSON carries the full history, the console adds nothing)
        if self.log_to_wandb or self.verbose > 0:
            print_or_wandb_log(
                self.log_to_wandb,
                summary,
                f"Train episode stats (step {self.num_timesteps})",
            )
        return True
