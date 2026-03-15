"""WandB eval logging callback: eval at eval_freq, log metrics and visit-count heatmap."""
import os
import collections

import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback

from utilities.debug import print_or_wandb_log
from utilities.heatmap_utils import create_visit_count_heatmap


class WandbEvalLoggingCallback(BaseCallback):
    """Eval at eval_freq, log to WandB. Log visit-count heatmap at same freq when use_wandb."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, visit_count_env=None, goal_cell=None, start_cell=None, run_name: str = "", beta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.use_wandb = use_wandb
        self.visit_count_env = visit_count_env
        self.goal_cell = goal_cell
        self.start_cell = start_cell
        self.run_name = run_name
        self.beta = beta

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
                if self.beta != 0:
                    if "extrinsic_reward" not in info:
                        raise KeyError("eval env step info must contain 'extrinsic_reward' when beta != 0")
                    if "intrinsic_reward" not in info:
                        raise KeyError("eval env step info must contain 'intrinsic_reward' when beta != 0")
                # When beta=0, info may be empty (no visit-count wrapper): extrinsic = step reward, intrinsic = 0
                extrinsic = float(info.get("extrinsic_reward", rewards[0]))
                intrinsic = float(info.get("intrinsic_reward", 0.0))
                ep_ext += extrinsic
                ep_int += self.beta * intrinsic
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
