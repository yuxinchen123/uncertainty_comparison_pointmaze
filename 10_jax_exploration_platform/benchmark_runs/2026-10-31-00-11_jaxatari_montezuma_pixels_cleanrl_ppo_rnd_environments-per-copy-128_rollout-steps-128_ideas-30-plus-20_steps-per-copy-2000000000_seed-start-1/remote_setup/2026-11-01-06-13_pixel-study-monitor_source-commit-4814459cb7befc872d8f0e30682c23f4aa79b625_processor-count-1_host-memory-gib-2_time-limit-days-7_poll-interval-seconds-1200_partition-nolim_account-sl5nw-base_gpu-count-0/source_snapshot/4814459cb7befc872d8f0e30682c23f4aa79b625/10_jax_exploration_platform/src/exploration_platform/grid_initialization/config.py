"""Configuration for exact count bonuses and independent grid policies."""

from dataclasses import dataclass
from collections.abc import Mapping
import math


SCALAR_NAMES = frozenset({"learning_rate", "intrinsic_coefficient", "extrinsic_coefficient",
    "entropy_coefficient", "value_coefficient", "clip_coefficient", "gamma_ext",
    "gamma_int", "gae_lambda", "max_gradient_norm", "adam_epsilon"})


@dataclass(frozen=True)
class GridPPOConfig:
    """Separate fixed program shapes from independently swept PPO scalars."""

    n_copies: int = 64
    policy_kind: str = "tabular"
    hidden_width: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    total_timesteps: int = 1_048_576
    episode_length: int = 1024
    start_state: int = 0
    goal_state: int = -1
    learning_rate: float = 2 ** -8
    intrinsic_coefficient: float = 1.0
    extrinsic_coefficient: float = 2.0
    entropy_coefficient: float = 2 ** -10
    value_coefficient: float = 0.5
    clip_coefficient: float = 0.1
    gamma_ext: float = 0.999
    gamma_int: float = 0.99
    gae_lambda: float = 0.95
    max_gradient_norm: float = 0.5
    adam_epsilon: float = 1e-5
    anneal_learning_rate: bool = True
    normalize_intrinsic_rewards: bool = False
    upstream_reward_filter_axis: bool = True
    associative_reward_filter: bool = False
    moment_reduction: str = "two_pass"
    per_copy_scalar_settings: tuple | Mapping = ()
    rollout_unroll: int = 1
    update_unroll: int = 1
    precompute_all_states: bool = False
    all_states_minibatch: bool = False
    hoist_values: bool = False
    hoist_log_probabilities: bool = False
    hoist_bonus: bool = False
    batch_random_keys: bool = False
    flat_parameters: bool = False
    flatten_update_scan: bool = False
    pre_gather_minibatches: bool = False
    donate_state: bool = False
    associative_gae: bool = False
    rsqrt_bonus: bool = False
    fuse_optimizer: bool = False
    flat_state_gather: bool = False

    def __post_init__(self):
        """Reject invalid shapes and normalize per-copy settings into immutable values."""
        # A single policy owns one environment, so num_steps is its entire rollout batch.
        if self.policy_kind not in {"tabular", "neural"}:
            raise ValueError("policy_kind must be tabular or neural")
        if self.moment_reduction not in {"two_pass", "raw_moments"}:
            raise ValueError("moment_reduction must be two_pass or raw_moments")
        if min(self.n_copies, self.hidden_width, self.num_steps, self.update_epochs,
               self.num_minibatches, self.total_timesteps, self.episode_length,
               self.rollout_unroll, self.update_unroll) < 1:
            raise ValueError("counts, lengths and unroll factors must be positive")
        if self.num_steps % self.num_minibatches or self.minibatch_size < 2:
            raise ValueError("rollouts must divide into minibatches containing at least two transitions")
        if self.total_timesteps % self.num_steps:
            raise ValueError("total_timesteps must contain an integer number of rollouts")
        # Before: mapping of lists; after: sorted immutable name/vector tuples.
        items = self.per_copy_scalar_settings.items() if isinstance(
            self.per_copy_scalar_settings, Mapping) else self.per_copy_scalar_settings
        settings, seen = [], set()
        for name, supplied in items:
            if name not in SCALAR_NAMES or name in seen:
                raise ValueError(f"unknown or repeated per-copy scalar: {name}")
            values = tuple(float(value) for value in supplied)
            if len(values) != self.n_copies:
                raise ValueError(f"{name} requires one value per independent copy")
            settings.append((name, values))
            seen.add(name)
        object.__setattr__(self, "per_copy_scalar_settings", tuple(sorted(settings)))
        for name in SCALAR_NAMES:
            values = self.per_copy_values(name) or (getattr(self, name),)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must be finite")
            if name in {"gamma_ext", "gamma_int", "gae_lambda"}:
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError(f"{name} must be between zero and one")
            elif any(value < 0 for value in values) or (
                    name == "adam_epsilon" and any(value == 0 for value in values)):
                raise ValueError(f"invalid nonnegative scalar: {name}")

    def per_copy_values(self, name):
        """Return an explicitly swept vector or leave a shared scalar unchanged."""
        for supplied_name, values in self.per_copy_scalar_settings:
            if supplied_name == name:
                return values
        return None

    @property
    def minibatch_size(self):
        """Count the transitions in each independent policy's minibatch."""
        return self.num_steps // self.num_minibatches

    @property
    def total_iterations(self):
        """Count full rollouts in the per-policy transition budget."""
        return self.total_timesteps // self.num_steps

    @property
    def n_envs(self):
        """Keep exact arrival counts unambiguous with one environment per policy."""
        return 1

    @property
    def batch_size(self):
        """Expose the original CleanRL normalization function's expected batch size."""
        return self.num_steps

