"""Callback to log distance between current algorithm intrinsic vector and GT (position_velocity visit counts)."""
import wandb

from stable_baselines3.common.callbacks import BaseCallback

from utilities.debug import align_print_dic
from distance_to_GT.algorithm_vector import compute_intrinsic_vector_distance


class DistanceLoggingCallback(BaseCallback):
    """At eval_freq, compute distance-to-GT for algorithms that don't need action; log as dict."""

    def __init__(
        self,
        algorithm: str,
        intrinsic_reward_model,
        visit_count_env_position_velocity,
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
        if self.use_wandb:
            wandb.log(metrics, step=self.num_timesteps)
        else:
            align_print_dic(metrics, "Distance To Ground Truth")
        return True
