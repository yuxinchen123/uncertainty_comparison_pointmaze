"""WandB eval logging callback: eval at eval_freq, log metrics and visit-count heatmap."""
import os
import collections

import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback

from rnd_exploration.common.debug import print_or_wandb_log
from rnd_exploration.common.heatmap_utils import create_visit_count_heatmap


class WandbEvalLoggingCallback(BaseCallback):
    """Eval at eval_freq, log to WandB. Log visit-count heatmap at same freq when use_wandb."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, use_wandb: bool, visit_count_env=None, goal_cell=None, start_cell=None, run_name: str = "", beta: float = 0.0, total_timesteps: int = None, n_eval_episodes_final: int = 100, eval_standalone: bool = True, log_to_wandb: bool = None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.total_timesteps = total_timesteps
        self.n_eval_episodes_final = n_eval_episodes_final
        # eval_standalone gates the expensive deterministic rollout. When False (the run-3.1.1 standard),
        # the rollout is skipped and eval_history holds only step + the cheap visit-count coverage; the
        # training-episode reward (TrainEpisodeStatsCallback) is used for the curves instead.
        self.eval_standalone = eval_standalone
        self.use_wandb = use_wandb
        # log_to_wandb decouples "wandb.init ran (params fetched)" from "metrics sent to wandb cloud".
        # Default = use_wandb (preserves the two original modes). The new wandb_param_only mode sets
        # use_wandb=True (init for sweep params) but log_to_wandb=False (metrics local only).
        self.log_to_wandb = use_wandb if log_to_wandb is None else log_to_wandb
        self.history = []  # per-eval summaries captured for local logging (used by both modes)
        self.visit_count_env = visit_count_env
        self.goal_cell = goal_cell
        self.start_cell = start_cell
        self.run_name = run_name
        self.beta = beta

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        # Heatmap is a wandb-only visualization: compute + upload it only when logging to wandb. Local
        # work-queue runs (no wandb) skip it entirely -- the visit_counts/* metrics already capture
        # coverage in the per-run JSON, and a per-eval PNG write would break the group-style, single
        # write-at-end logging design (and would clutter image/ with thousands of files across 512 workers).
        if self.visit_count_env is not None and self.log_to_wandb and wandb.run:
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
                # upload the heatmap image to wandb; no local PNG is written
                wandb.log({
                    f"train_visit_count_heatmap/{name}": wandb.Image(fig),
                    "media/heatmap_latest": wandb.Image(fig),
                }, step=step, commit=True)
                plt.close(fig)
            except Exception as e:
                if self.verbose > 0:
                    print(f"  Heatmap: {e}")
        # The eval summary always carries the step; the eval/* reward keys are added only when the
        # standalone rollout runs. The visit-count coverage (below) is cheap and logged either way.
        summary = collections.OrderedDict([("step", self.num_timesteps)])
        # Standalone eval rollout (OFF by default): collect extrinsic, intrinsic, and total per
        # deterministic episode. run-4 convention: NO final-eval special case -- the final eval uses the
        # same n_eval_episodes as every other eval (the n_eval_episodes_final knob is retired).
        if self.eval_standalone:
            n_ep = self.n_eval_episodes
            episode_extrinsic = []
            episode_intrinsic = []
            episode_total = []
            episode_lengths = []
            for _ in range(n_ep):
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

            # reward summary keys (only present when the rollout ran)
            summary["eval/n_eval_episodes"] = n_ep
            summary["eval/mean_extrinsic_reward"] = float(np.mean(episode_extrinsic))
            summary["eval/mean_intrinsic_reward"] = float(np.mean(episode_intrinsic))
            summary["eval/mean_total_reward"] = float(np.mean(episode_total))
            summary["eval/mean_ep_length"] = float(np.mean(episode_lengths))
        if self.visit_count_env is not None:
            visit_counts = self.visit_count_env.get_visit_counts()
            open_cells = (self.visit_count_env.maze_map == 0).sum()
            visited_cells = (visit_counts > 0).sum()
            total_visits = int(visit_counts.sum())
            summary["visit_counts/total_visits"] = total_visits
            summary["visit_counts/cells_visited"] = int(visited_cells)
            summary["visit_counts/open_cells"] = int(open_cells)
            summary["visit_counts/coverage_pct"] = 100.0 * visited_cells / max(1, open_cells)
        # capture every eval summary locally (both modes), then send to wandb only in full mode
        self.history.append(dict(summary))
        # wandb mode still logs every snapshot; local mode prints the block only when verbose>0
        # (default silent — the per-run JSON carries the full history, the console adds nothing)
        if self.log_to_wandb or self.verbose > 0:
            print_or_wandb_log(
                self.log_to_wandb,
                summary,
                f"Eval (step {self.num_timesteps})",
            )
        return True
