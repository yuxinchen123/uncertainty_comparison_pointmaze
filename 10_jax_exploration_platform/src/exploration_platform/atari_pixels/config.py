"""The original CleanRL Atari configuration, separate from the PointMaze defaults."""

from dataclasses import dataclass


CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"


@dataclass(frozen=True)
class PixelPPOConfig:
    """All settings of one compiled group of independent pixel policies."""

    n_copies: int = 1
    n_envs: int = 128
    num_steps: int = 128
    action_count: int = 18
    total_timesteps: int = 2_000_000_000
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-5
    anneal_learning_rate: bool = True
    gamma_ext: float = 0.999
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.1
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    intrinsic_coefficient: float = 1.0
    extrinsic_coefficient: float = 2.0
    predictor_update_proportion: float = 0.25
    observation_warmup_iterations: int = 50
    per_copy_learning_rates: tuple[float, ...] = ()
    per_copy_intrinsic_coefficients: tuple[float, ...] = ()
    rollout_unroll: int = 1
    update_unroll: int = 1
    cache_target_features: bool = False
    hoist_rollout_bonus: bool = False
    donate_state: bool = False
    network_layout: str = "nchw"
    dense_batching: str = "matmul"
    convolution_batching: str = "vmap"
    flat_parameters: bool = False
    fuse_optimizer: bool = False
    pre_gather_minibatches: bool = False
    hoist_critic: bool = False
    hoist_rollout_log_probabilities: bool = False
    rollout_observation_storage: str = "float32"
    associative_gae: bool = False
    associative_reward_filter: bool = False
    moment_reduction: str = "two_pass"
    # These preserve literal upstream behavior rather than the PointMaze corrections.
    upstream_reward_filter_axis: bool = True
    warmup_keeps_initial_observation: bool = True

    def __post_init__(self):
        """Reject shapes and settings that cannot form the specified PPO batches."""
        # A minibatch has at least two rows because upstream uses sample standard deviation.
        counts = (self.n_copies, self.n_envs, self.num_steps, self.action_count,
                  self.update_epochs, self.num_minibatches, self.total_timesteps)
        if min(counts) < 1:
            raise ValueError("copy, environment, rollout, action, and update counts must be positive")
        if self.batch_size % self.num_minibatches or self.minibatch_size < 2:
            raise ValueError("the rollout batch must divide into minibatches of at least two rows")
        if self.total_iterations < 1:
            raise ValueError("the transition budget must contain at least one complete rollout")
        if not 0.0 <= self.predictor_update_proportion <= 1.0:
            raise ValueError("predictor_update_proportion must be in [0, 1]")
        if self.rollout_unroll < 1 or self.update_unroll < 1:
            raise ValueError("scan unroll factors must be positive")
        if self.observation_warmup_iterations < 0:
            raise ValueError("observation_warmup_iterations must be nonnegative")
        for values in (self.per_copy_learning_rates, self.per_copy_intrinsic_coefficients):
            if values and len(values) != self.n_copies:
                raise ValueError("each per-copy setting must contain exactly n_copies values")
        choices = {"network_layout": ("nchw", "nhwc"),
                   "dense_batching": ("matmul", "einsum"),
                   "convolution_batching": ("vmap", "grouped"),
                   "rollout_observation_storage": ("float32", "uint8"),
                   "moment_reduction": ("two_pass", "raw_moments")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")

    @property
    def batch_size(self):
        """Transitions belonging to one policy in one rollout."""
        return self.n_envs * self.num_steps

    @property
    def minibatch_size(self):
        """Rows belonging to one policy in one optimizer step."""
        return self.batch_size // self.num_minibatches

    @property
    def total_iterations(self):
        """Complete rollouts, using the original floor-to-whole-updates convention."""
        return self.total_timesteps // self.batch_size
