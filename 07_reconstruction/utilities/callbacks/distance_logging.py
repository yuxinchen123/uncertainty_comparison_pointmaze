"""Callback to log distance between current algorithm intrinsic vector and GT (position_velocity visit counts)."""
from typing import Any

import wandb

from stable_baselines3.common.callbacks import BaseCallback

from utilities.debug import print_or_wandb_log
from distance_to_GT.algorithm_vector import compute_intrinsic_vector_distance


class DistanceLoggingCallback(BaseCallback):
    """
    At eval_freq, compute distance between the current algorithm's intrinsic vector
    and GT (position_velocity visit counts). Logs metrics for algorithms in
    ALGORITHMS_NO_ACTION (no_exploration, gt_position, gt_position_velocity, RND state/next_state, etc.).
    Logs to wandb or prints via align_print_dic.
    """

    def __init__(
        self,
        algorithm: str,
        intrinsic_reward_model: Any,
        visit_count_env_position_velocity: Any,
        eval_freq: int,
        use_wandb: bool,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.algorithm = algorithm
        self.intrinsic_reward_model = intrinsic_reward_model
        self.visit_count_env_position_velocity = visit_count_env_position_velocity
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        if self.use_wandb and wandb.run is not None:
            wandb.define_metric("step")
            wandb.define_metric("distance_to_gt/*", step_metric="step")

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        maze_map = self.visit_count_env_position_velocity.maze_map
        metrics = compute_intrinsic_vector_distance(
            self.algorithm,
            maze_map,
            self.intrinsic_reward_model,
            self.visit_count_env_position_velocity,
        )
        if metrics is None:
            return True
        metrics["step"] = self.num_timesteps
        print_or_wandb_log(
            self.use_wandb,
            metrics,
            "Distance To Ground Truth"
        )
        return True
