"""Create archived initial bonus fields with independent named random quantities."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .environments import GridWorld


ADVERSARIAL_FIELD_KINDS = (
    "high_start_cell", "high_start_region", "high_interior_pair", "smooth_interior_peak",
    "decreasing_exponential", "low_middle_interval", "repeated_low_intervals",
    "repeated_smooth_peaks", "alternating_high_low", "decreasing_steps",
)
FIELD_KINDS = ("uniform", "none", "random_log_uniform", *ADVERSARIAL_FIELD_KINDS, "random_network")


@dataclass(frozen=True)
class BonusField:
    """Retain the original field, its mean-one comparison, and its generating settings."""

    raw: np.ndarray
    normalized: np.ndarray
    metadata: dict


def derive_quantity_seed(base_seed: int, *quantity_names) -> int:
    """Derive a stable 32-bit seed independently for each named random quantity."""
    # Before: (7, 'target', 'layer_0'); after: one reproducible unsigned integer.
    identifier = "::".join(str(value) for value in (base_seed, "grid_initialization", *quantity_names))
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[-4:], "big")


def _archive_field(raw, metadata: dict, allow_zero: bool = False) -> BonusField:
    """Validate field values and keep a separately normalized copy."""
    # Nonzero fields must stay positive without clipping or adding artificial bonuses.
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Initial bonus fields must be finite one-dimensional arrays.")
    if allow_zero:
        if np.any(values != 0):
            raise ValueError("The no-bonus field must contain only zeros.")
        normalized = values.copy()
    else:
        if np.any(values <= 0):
            raise ValueError("Every initial bonus must be strictly positive.")
        normalized = (values.astype(np.float64) / values.astype(np.float64).mean()).astype(np.float32)
        if np.any(normalized <= 0):
            raise ValueError("Mean normalization underflowed an initial bonus to zero.")

    # Before: an unscaled field; after: both arrays plus their recorded scale and range.
    details = {**metadata, "raw_mean": float(values.astype(np.float64).mean()),
               "raw_minimum": float(values.min()), "raw_maximum": float(values.max()),
               "normalized_mean": float(normalized.astype(np.float64).mean()),
               "normalized_minimum": float(normalized.min()),
               "normalized_maximum": float(normalized.max()),
               "normalization": "none" if allow_zero else "mean over accessible states"}
    return BonusField(values, normalized, details)


def make_initial_bonus_field(kind: str, geometry: GridWorld, seed: int = 0,
                             severity_exponent: int = 4, width_exponent: int = 2,
                             period_exponent: int = 3, step_count_exponent: int = 2) -> BonusField:
    """Make one uniform, random, adversarial, or untrained-network bonus field."""
    # Neural initialization is separate so importing this module does not load JAX.
    if kind == "random_network":
        return make_random_network_bonus_field(geometry, seed=seed)
    if kind not in FIELD_KINDS:
        raise ValueError(f"Unknown initial bonus field: {kind!r}.")
    if not isinstance(severity_exponent, (int, np.integer)) or not 0 <= severity_exponent <= 30:
        raise ValueError("Severity exponent must be an integer from zero through thirty.")
    for name, value in (("width", width_exponent), ("period", period_exponent),
                        ("step count", step_count_exponent)):
        if not isinstance(value, (int, np.integer)) or not 0 <= value <= 30:
            raise ValueError(f"The {name} exponent must be an integer from zero through thirty.")

    # Before: state identifiers 0..S-1; after: normalized distance x in [0,1].
    count = geometry.state_count
    index = np.arange(count, dtype=np.int32)
    distance = index.astype(np.float64) / (count - 1)
    amplitude, minimum = 2.0 ** severity_exponent, 2.0 ** (-severity_exponent)
    width, period, steps = 2 ** width_exponent, 2 ** period_exponent, 2 ** step_count_exponent
    interior = (count - 1) // 4
    metadata = {"kind": kind, "seed": int(seed), "severity_exponent": int(severity_exponent),
                "width_exponent": int(width_exponent), "period_exponent": int(period_exponent),
                "step_count_exponent": int(step_count_exponent), "state_count": count,
                "interior_peak_state": interior, "seed_scheme": "sha256_named_quantities_v1"}

    # Each named candidate changes only the initial field; every later count decays identically.
    if kind == "uniform":
        raw = np.ones(count)
    elif kind == "none":
        return _archive_field(np.zeros(count), metadata, allow_zero=True)
    elif kind == "random_log_uniform":
        generator = np.random.default_rng(derive_quantity_seed(seed, "random_log_uniform", "state_values"))
        raw = np.exp2(generator.uniform(-severity_exponent, severity_exponent, count))
    elif kind == "high_start_cell":
        raw = 1 + (amplitude - 1) * (index == 0)
    elif kind == "high_start_region":
        if width >= count:
            raise ValueError("The high start region must be shorter than the path.")
        raw = 1 + (amplitude - 1) * (index < width)
    elif kind == "high_interior_pair":
        raw = 1 + (amplitude - 1) * ((index == interior) | (index == interior + 1))
    elif kind == "smooth_interior_peak":
        raw = minimum + (1 - minimum) * np.exp2(-((index - interior) / width) ** 2)
    elif kind == "decreasing_exponential":
        raw = np.exp2(-severity_exponent * distance)
    elif kind == "low_middle_interval":
        if interior + width >= count:
            raise ValueError("The low middle interval must end before the goal.")
        raw = np.where((index >= interior) & (index < interior + width), minimum, 1)
    elif kind == "repeated_low_intervals":
        if 2 * width >= count:
            raise ValueError("Repeated low intervals require at least one low interval on the path.")
        raw = np.where((index // width) % 4 == 2, minimum, 1)
    elif kind == "repeated_smooth_peaks":
        raw = minimum + (1 - minimum) * (1 + np.cos(2 * math.pi * index / period)) / 2
    elif kind == "alternating_high_low":
        raw = np.where(index % 2 == 0, 1, minimum)
    else:
        raw = np.exp2(-severity_exponent * np.floor(steps * distance) / steps)
    return _archive_field(raw, metadata)


def make_random_network_bonus_field(geometry: GridWorld, seed: int = 0,
                                    constant_coordinate: bool = False) -> BonusField:
    """Evaluate independent untrained coordinate networks once and archive their squared errors."""
    # This function initializes JAX arrays and must be called on an allocated compute node.
    import jax
    import jax.numpy as jnp

    # Before: [S,2] centered coordinates; optional control appends a constant to give [S,3].
    inputs = geometry.coordinates
    if constant_coordinate:
        inputs = np.column_stack((inputs, np.ones(geometry.state_count, dtype=np.float32)))
    observations = jnp.asarray(inputs, dtype=jnp.float32)
    architectures = {"target": (32, 64, 64, 512), "predictor": (32, 64, 64, 512, 512, 512)}

    def evaluate_network(network_name: str):
        """Initialize each layer independently and evaluate one fixed network."""
        # Orthogonal weights and zero biases follow the CleanRL random-network initializer.
        hidden = observations
        for layer, output_width in enumerate(architectures[network_name]):
            layer_seed = derive_quantity_seed(seed, "random_network", network_name, f"layer_{layer}")
            weights = jax.nn.initializers.orthogonal(math.sqrt(2))(
                jax.random.PRNGKey(layer_seed), (hidden.shape[-1], output_width), jnp.float32)
            hidden = hidden @ weights
            if layer < 3:
                hidden = jax.nn.leaky_relu(hidden, negative_slope=0.01)
            elif layer < len(architectures[network_name]) - 1:
                hidden = jax.nn.relu(hidden)
        return hidden

    # Before: target/predictor features [S,512]; after: one fixed nonnegative error per state.
    target = evaluate_network("target")
    predictor = evaluate_network("predictor")
    raw = np.asarray(0.5 * jnp.sum(jnp.square(predictor - target), axis=-1), dtype=np.float32)
    metadata = {"kind": "random_network", "seed": int(seed), "state_count": geometry.state_count,
                "coordinate_transform": "centered and isotropically scaled",
                "constant_coordinate": bool(constant_coordinate),
                "target_widths": [inputs.shape[1], *architectures["target"]],
                "predictor_widths": [inputs.shape[1], *architectures["predictor"]],
                "first_three_activations": "leaky rectified linear with negative slope 0.01",
                "predictor_extra_activations": "rectified linear",
                "output_activation": "linear", "weight_initialization": "orthogonal",
                "weight_scale": math.sqrt(2), "bias_initialization": "zero",
                "error_reduction": "one half times sum of squared feature errors, as in CleanRL",
                "seed_scheme": "sha256_named_quantities_v1", "trained": False}
    return _archive_field(raw, metadata)
