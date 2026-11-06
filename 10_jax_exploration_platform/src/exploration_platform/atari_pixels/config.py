"""The original CleanRL Atari configuration, separate from the PointMaze defaults."""

from dataclasses import dataclass
from collections.abc import Mapping
import math


CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
SCALAR_SETTING_NAMES = frozenset({
    "learning_rate", "intrinsic_coefficient", "extrinsic_coefficient",
    "entropy_coefficient", "value_coefficient", "clip_coefficient",
    "gamma_ext", "gamma_int", "gae_lambda", "predictor_update_proportion",
    "max_gradient_norm",
    "adam_epsilon",
})


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
    per_copy_scalar_settings: tuple[tuple[str, tuple[float, ...]], ...] | Mapping = ()
    rollout_unroll: int = 1
    update_unroll: int = 1
    cache_target_features: bool = False
    hoist_rollout_bonus: bool = False
    donate_state: bool = False
    network_layout: str = "nchw"
    dense_batching: str = "matmul"
    convolution_batching: str = "vmap"
    convolution_algorithm: str = "compiler"
    # XLA options for the compiled training iteration only, as (name, value) pairs. They may
    # change how kernels are launched, never the arithmetic.
    iteration_compiler_options: tuple[tuple[str, str], ...] = ()
    paired_rnd_rollout_trunks: bool = False
    # One algorithm per convolution layer (first, second, third); empty uses convolution_algorithm.
    layer_convolution_algorithms: tuple[str, ...] = ()
    flat_parameters: bool = False
    fuse_optimizer: bool = False
    pre_gather_minibatches: bool = False
    hoist_critic: bool = False
    hoist_rollout_log_probabilities: bool = False
    rollout_observation_storage: str = "float32"
    associative_gae: bool = False
    associative_reward_filter: bool = False
    moment_reduction: str = "two_pass"
    fuse_minibatch_image_preparation: bool = False
    recompute_network_activations: bool = False
    fuse_network_bias_and_activation: bool = False
    flatten_update_scan: bool = False
    batch_quantity_keyed_random_generation: bool = False
    rollout_frame_storage: str = "stacked_frames"
    # These preserve literal upstream behavior rather than the PointMaze corrections.
    upstream_reward_filter_axis: bool = True
    warmup_keeps_initial_observation: bool = True

    def __post_init__(self):
        """Reject shapes and settings that cannot form the specified PPO batches."""
        # JSON changes tuples to lists; normalize the two existing vector inputs as well.
        object.__setattr__(self, "per_copy_learning_rates", tuple(float(value) for value in self.per_copy_learning_rates))
        object.__setattr__(self, "per_copy_intrinsic_coefficients", tuple(
            float(value) for value in self.per_copy_intrinsic_coefficients))
        # Before: a JSON/Python mapping; after: sorted immutable (name, values) pairs.
        items = (self.per_copy_scalar_settings.items() if isinstance(self.per_copy_scalar_settings, Mapping)
                 else self.per_copy_scalar_settings)
        canonical, seen = [], set()
        for name, values in items:
            if name not in SCALAR_SETTING_NAMES:
                if name in self.__dataclass_fields__:
                    raise ValueError(f"per-copy structural or compilation differences are unsupported: {name}")
                raise ValueError(f"unsupported per-copy scalar setting: {name}")
            if name in seen:
                raise ValueError(f"duplicate per-copy scalar setting: {name}")
            seen.add(name)
            values = tuple(float(value) for value in values)
            if len(values) != self.n_copies or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} requires n_copies finite numeric values")
            # JAX GPU arithmetic may turn subnormal inputs into zero, including Adam's divisor.
            if any(abs(value) > float.fromhex("0x1.fffffep+127") or
                   (value != 0 and abs(value) < 2 ** -126) for value in values):
                raise ValueError(f"per-copy {name} must fit the normal float32 range or equal zero")
            if name in {"gamma_ext", "gamma_int", "gae_lambda", "predictor_update_proportion"}:
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError(f"per-copy {name} must be in [0, 1]")
            if name in {"learning_rate", "entropy_coefficient", "value_coefficient",
                        "clip_coefficient", "max_gradient_norm"} and any(value < 0 for value in values):
                raise ValueError(f"per-copy {name} must be nonnegative")
            if name == "adam_epsilon" and any(value <= 0 for value in values):
                raise ValueError("per-copy adam_epsilon must be positive")
            canonical.append((name, values))
        object.__setattr__(self, "per_copy_scalar_settings", tuple(sorted(canonical)))
        if self.per_copy_learning_rates and "learning_rate" in seen:
            raise ValueError("configure learning_rate through only one per-copy interface")
        if self.per_copy_intrinsic_coefficients and "intrinsic_coefficient" in seen:
            raise ValueError("configure intrinsic_coefficient through only one per-copy interface")

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
                   "convolution_batching": ("vmap", "grouped", "sequential", "unrolled"),
                   "convolution_algorithm": ("compiler", "patch_weight_gradient", "patch_matrix",
                                             "space_to_depth"),
                   "rollout_observation_storage": ("float32", "uint8"),
                   "rollout_frame_storage": ("stacked_frames", "indexed_frames"),
                   "moment_reduction": ("two_pass", "raw_moments")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.layer_convolution_algorithms and (
                len(self.layer_convolution_algorithms) != 3
                or not set(self.layer_convolution_algorithms) <= set(choices["convolution_algorithm"])):
            raise ValueError("layer_convolution_algorithms needs three known algorithms")

    def per_copy_values(self, name):
        """Return explicitly supplied values, leaving unspecified settings as original scalars."""
        for setting, values in self.per_copy_scalar_settings:
            if setting == name:
                return values
        if name == "learning_rate" and self.per_copy_learning_rates:
            return self.per_copy_learning_rates
        if name == "intrinsic_coefficient" and self.per_copy_intrinsic_coefficients:
            return self.per_copy_intrinsic_coefficients
        return None

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
