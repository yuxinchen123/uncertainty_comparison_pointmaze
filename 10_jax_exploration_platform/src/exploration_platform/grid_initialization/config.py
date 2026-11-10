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
    # Defaults are CleanRL PPO-RND's scalar settings, as the audit report recommends for the
    # straight and snake grids (GPT_session/ChatGPT_Research-audit-report_20261107-1858.pdf,
    # pages 14-15). Before 2026-11-07: learning rate 2^-8 and entropy coefficient 2^-10.
    learning_rate: float = 1e-4
    intrinsic_coefficient: float = 1.0
    extrinsic_coefficient: float = 2.0
    entropy_coefficient: float = 1e-3
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
    # "count": frozen field divided by the square root of visits; "online_rnd": the predictor
    # error of a CleanRL-style random network distillation trained during PPO.
    intrinsic_source: str = "count"
    # Whiten the online RND inputs with running statistics (CleanRL observation normalization).
    normalize_rnd_inputs: bool = False
    predictor_update_proportion: float = 0.25
    # Count rule exponent: bonus = field x visits^(-exponent); 0.5 keeps the exact 1/sqrt path.
    count_decay_exponent: float = 0.5
    # Online RND input: "coordinates" (coordinate-RND) or "one_hot" (tabular-RND). The policy always reads coordinates.
    rnd_input: str = "coordinates"
    # Online RND bonus readout: CleanRL's half summed squared error, or its square root (the error length);
    # the "initial_ratio" forms divide by the same readout of the untrained predictor in each cell.
    rnd_readout: str = "half_squared"
    # Scale the target's features to unit length in every cell.
    rnd_unit_length_target: bool = False
    # Predictor loss: CleanRL's mean squared feature error, the power (2/q) E^(q/2) of the summed squared
    # error E, or a regression to the predictor's own output moved so that the error length goes r -> r/sqrt(1+r^2).
    rnd_loss: str = "cleanrl"
    rnd_loss_power: float = 2.0
    # "shared_adam": the predictor joins the policy's Adam and joint clip (CleanRL); otherwise it has its own
    # optimizer with no clip and its own learning rate, constant or rate/(1 + t/rnd_learning_rate_time_scale).
    rnd_optimizer: str = "shared_adam"
    rnd_learning_rate: float = 1e-4
    rnd_learning_rate_time_scale: float = 0.0
    # "all" layers of the predictor train, or only its input weights (one row per cell for one-hot inputs).
    rnd_trainable: str = "all"
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
        if self.intrinsic_source not in {"count", "online_rnd"}:
            raise ValueError("intrinsic_source must be count or online_rnd")
        if self.normalize_rnd_inputs and self.intrinsic_source != "online_rnd":
            raise ValueError("RND input normalization acts only on online RND; frozen fields are built normalized")
        if not math.isfinite(self.count_decay_exponent) or self.count_decay_exponent <= 0:
            raise ValueError("count_decay_exponent must be positive")
        if self.count_decay_exponent != 0.5 and self.rsqrt_bonus:
            raise ValueError("rsqrt_bonus implements only the exponent 0.5")
        choices = {"rnd_input": {"coordinates", "one_hot"},
                   "rnd_readout": {"half_squared", "length", "initial_ratio_half_squared", "initial_ratio_length"},
                   "rnd_loss": {"cleanrl", "power", "shrink_target"},
                   "rnd_optimizer": {"shared_adam", "adam", "sgd", "adagrad"}, "rnd_trainable": {"all", "input_layer"}}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        changed = [name for name, default in (("rnd_input", "coordinates"), ("rnd_readout", "half_squared"),
                   ("rnd_unit_length_target", False), ("rnd_loss", "cleanrl"), ("rnd_optimizer", "shared_adam"),
                   ("rnd_trainable", "all")) if getattr(self, name) != default]
        if changed and self.intrinsic_source != "online_rnd":
            raise ValueError(f"online RND settings were given to a count-bonus configuration: {changed}")
        if self.rnd_input == "one_hot" and self.normalize_rnd_inputs:
            raise ValueError("one-hot RND inputs are never whitened")
        if self.rnd_loss == "power" and self.rnd_loss_power < 2:
            raise ValueError("rnd_loss_power must be at least 2")
        if self.rnd_learning_rate <= 0 or self.rnd_learning_rate_time_scale < 0:
            raise ValueError("rnd_learning_rate must be positive and its time scale nonnegative")
        if self.intrinsic_source == "online_rnd":
            if not 0 < self.predictor_update_proportion <= 1:
                raise ValueError("predictor_update_proportion must be in (0, 1]")
            unsupported = [name for name in ("flatten_update_scan", "pre_gather_minibatches",
                           "all_states_minibatch", "hoist_bonus", "flat_parameters") if getattr(self, name)]
            if unsupported:
                raise ValueError(f"online RND is not implemented with: {unsupported}")
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

