"""Grid random network distillation: the frozen-field networks, trained online as in CleanRL PPO-RND.

The target 2->32->64->64->512 and predictor 2->32->64->64->512->512->512 are the coordinate
networks that build the frozen random-network field (audit report, page 29). Their weights come
from the same named seeds as `fields.make_random_network_bonus_field`, generated on the processor
so every graphics-card class receives identical numbers; biases start at zero, as in CleanRL.
"""

from dataclasses import dataclass
import math

import numpy as np

from .environments import GridWorld
from .fields import _archive_field, derive_quantity_seed

WIDTHS = {"target": (32, 64, 64, 512), "predictor": (32, 64, 64, 512, 512, 512)}


@dataclass(frozen=True)
class NetworkOptions:
    """How one study builds its RND networks; the defaults are the coordinate networks above."""

    # "coordinates": 2-d cell coordinates (coordinate-RND); "one_hot": one cell indicator per state (tabular-RND).
    input_kind: str = "coordinates"
    first_width: int = 32
    bias_std: float = 0.0
    zero_predictor_output: bool = False
    # "own": the 6-layer CleanRL predictor; "target_copy": the target's layers after the input layer, frozen.
    predictor_trunk: str = "own"

    def __post_init__(self):
        """Reject construction options the trainer does not implement."""
        if self.input_kind not in {"coordinates", "one_hot"}:
            raise ValueError("input_kind must be coordinates or one_hot")
        if self.predictor_trunk not in {"own", "target_copy"}:
            raise ValueError("predictor_trunk must be own or target_copy")
        if self.first_width < 1 or self.bias_std < 0:
            raise ValueError("first_width must be positive and bias_std nonnegative")

    @property
    def is_default(self):
        """True for the coordinate networks of the 2026-11-07 study."""
        return self == NetworkOptions()
LEAKY_LAYERS = 3
LEAKY_SLOPE = 0.01
WARMUP_ROLLOUTS = 50
WARMUP_STEPS = 128
CLIP = 5.0


def network_weights(field_seed: int, input_width: int = 2):
    """Generate one field seed's target and predictor weights on the processor."""
    # Before: field seed 3; after: {"target": [W0 [2,32], ...], "predictor": [...]} as float32.
    import jax
    import jax.numpy as jnp
    with jax.default_device(jax.devices("cpu")[0]):
        networks = {}
        for name, widths in WIDTHS.items():
            layers, width = [], input_width
            for layer, output in enumerate(widths):
                key = jax.random.PRNGKey(derive_quantity_seed(field_seed, "random_network", name, f"layer_{layer}"))
                weight = jax.nn.initializers.orthogonal(math.sqrt(2))(key, (width, output), jnp.float32)
                layers.append(np.asarray(weight, np.float32))
                width = output
            networks[name] = layers
    return networks


def input_table(geometry: GridWorld, options: NetworkOptions):
    """Return every cell's RND input: its coordinates or its one-hot indicator."""
    # Before: 32 cells; after: [32,2] coordinates, or the [32,32] identity for one-hot inputs.
    if options.input_kind == "one_hot":
        return np.eye(geometry.state_count, dtype=np.float32)
    return np.asarray(geometry.coordinates, np.float32)


def network_layers(field_seed: int, options: NetworkOptions, input_width: int):
    """Generate one field seed's target and predictor weights and biases under the options."""
    # The default options give exactly network_weights(field_seed) with zero biases.
    import jax
    import jax.numpy as jnp
    if options.is_default:
        weights = network_weights(field_seed, input_width)
    else:
        widths = {name: (options.first_width, *values[1:]) for name, values in WIDTHS.items()}
        weights = {}
        with jax.default_device(jax.devices("cpu")[0]):
            for name, sizes in widths.items():
                layers, width = [], input_width
                for layer, output in enumerate(sizes):
                    # New shapes get new identifiers, so the coordinate networks keep their numbers.
                    suffix = ("_one_hot" if options.input_kind == "one_hot" and layer == 0 else "") + (
                        f"_first_width_{options.first_width}" if options.first_width != 32 and layer < 2 else "")
                    key = jax.random.PRNGKey(derive_quantity_seed(field_seed, "random_network", name, f"layer_{layer}{suffix}"))
                    layers.append(np.asarray(jax.nn.initializers.orthogonal(math.sqrt(2))(
                        key, (width, output), jnp.float32), np.float32))
                    width = output
                weights[name] = layers
    biases = {}
    for name, layers in weights.items():
        rows = []
        for layer, weight in enumerate(layers):
            if options.bias_std > 0:
                generator = np.random.default_rng(derive_quantity_seed(
                    field_seed, "random_network_bias", name, f"layer_{layer}"))
                rows.append((options.bias_std * generator.standard_normal(weight.shape[1])).astype(np.float32))
            else:
                rows.append(np.zeros(weight.shape[1], np.float32))
        biases[name] = rows
    # The trunk copy reuses the target after its input layer; the zero output starts every error at |f(x)|^2.
    if options.predictor_trunk == "target_copy":
        weights["predictor"] = [weights["predictor"][0]] + [weight.copy() for weight in weights["target"][1:]]
        biases["predictor"] = [biases["predictor"][0]] + [bias.copy() for bias in biases["target"][1:]]
    if options.zero_predictor_output:
        weights["predictor"][-1] = np.zeros_like(weights["predictor"][-1])
        biases["predictor"][-1] = np.zeros_like(biases["predictor"][-1])
    return {name: [(weights[name][layer], biases[name][layer]) for layer in range(len(weights[name]))]
            for name in weights}


def numpy_forward(layers, inputs):
    """Evaluate one network in float64 with the CleanRL activation order; layers are weights or (weight, bias)."""
    values = np.asarray(inputs, np.float64)
    for layer, weight in enumerate(layers):
        if isinstance(weight, tuple):
            weight, bias = weight
            values = values @ weight.astype(np.float64) + bias.astype(np.float64)
        else:
            values = values @ weight.astype(np.float64)
        if layer < LEAKY_LAYERS:
            values = np.where(values > 0, values, LEAKY_SLOPE * values)
        elif layer < len(layers) - 1:
            values = np.maximum(values, 0.0)
    return values


def normalize_coordinates(coordinates, mean, variance):
    """Whiten coordinates with running statistics and clip to [-5, 5], as CleanRL does."""
    # Before: [S,2] coordinates; after: [S,2] whitened, clipped inputs.
    values = (np.asarray(coordinates, np.float64) - mean) / np.sqrt(variance)
    return np.clip(values, -CLIP, CLIP)


def warmup_statistics(geometry: GridWorld, episode_length: int, field_seed: int):
    """Start the input statistics from 50 random-action rollouts of 128 steps, as CleanRL does."""
    # The warm-up is seeded by the field seed, so the policies that share a field share its
    # statistics; its steps are never counted as training transitions.
    generator = np.random.default_rng(derive_quantity_seed(field_seed, "rnd_observation_warmup", "actions"))
    table, coordinates = geometry.transition_table, np.asarray(geometry.coordinates, np.float64)
    goal = geometry.goal_state
    mean, variance, count = np.zeros(2), np.ones(2), 1e-4
    position, elapsed = 0, 0
    for _ in range(WARMUP_ROLLOUTS):
        observations = np.empty((WARMUP_STEPS, 2))
        for step, action in enumerate(generator.integers(0, 4, WARMUP_STEPS)):
            # Each recorded observation is the one after the move, the start cell after a reset.
            arrived = int(table[position, action])
            elapsed += 1
            done = arrived == goal or elapsed >= episode_length
            position, elapsed = (0, 0) if done else (arrived, elapsed)
            observations[step] = coordinates[position]
        # gym RunningMeanStd's parallel update, once per rollout of 128 observations.
        batch_mean, batch_variance, batch_count = observations.mean(0), observations.var(0), WARMUP_STEPS
        delta, total = batch_mean - mean, count + batch_count
        mean = mean + delta * batch_count / total
        variance = (variance * count + batch_variance * batch_count + delta ** 2 * count * batch_count / total) / total
        count = total
    return {"mean": mean, "variance": variance, "count": float(count)}


def frozen_network_field(geometry: GridWorld, field_seed: int, statistics=None, options=NetworkOptions()):
    """Return the frozen random-network field on coordinates (raw or whitened) or one-hot inputs, mean-normalized."""
    if options.input_kind == "one_hot" and statistics is not None:
        raise ValueError("one-hot RND inputs are never whitened")
    table = input_table(geometry, options)
    networks = network_layers(field_seed, options, table.shape[1])
    inputs = (table if statistics is None else
              normalize_coordinates(geometry.coordinates, statistics["mean"], statistics["variance"]))
    errors = 0.5 * np.sum((numpy_forward(networks["predictor"], inputs)
                           - numpy_forward(networks["target"], inputs)) ** 2, axis=-1)
    metadata = {"kind": "random_network", "seed": int(field_seed), "state_count": geometry.state_count,
                "generator": "online_rnd.frozen_network_field (processor weights, float64 evaluation)",
                "target_widths": [table.shape[1], *[layer[0].shape[1] for layer in networks["target"]]],
                "predictor_widths": [table.shape[1], *[layer[0].shape[1] for layer in networks["predictor"]]],
                "input_kind": options.input_kind, "network_options": options.__dict__,
                "weight_initialization": "orthogonal, gain sqrt(2)",
                "bias_initialization": "zero" if options.bias_std == 0 else f"normal, standard deviation {options.bias_std}",
                "input_normalization": "none" if statistics is None else
                "whitened by random-action warm-up statistics, clipped to [-5, 5]",
                "input_statistics": None if statistics is None else
                {key: np.asarray(value).tolist() for key, value in statistics.items()},
                "error_reduction": "one half times sum of squared feature errors, as in CleanRL",
                "seed_scheme": "sha256_named_quantities_v1", "trained": False}
    return _archive_field(errors, metadata)


def stack_networks(field_seeds, options=NetworkOptions(), input_width=2):
    """Stack per-copy network weights and biases for the copies' field seeds."""
    # Before: field seeds [0,0,1]; after: {"target": ({"weight": [3,2,32], "bias": [3,32]}, ...)}.
    cache = {int(seed): network_layers(int(seed), options, input_width) for seed in set(int(seed) for seed in field_seeds)}
    stacked = {}
    for name in WIDTHS:
        stacked[name] = tuple(
            {"weight": np.stack([cache[int(seed)][name][layer][0] for seed in field_seeds]),
             "bias": np.stack([cache[int(seed)][name][layer][1] for seed in field_seeds])}
            for layer in range(len(cache[int(field_seeds[0])][name])))
    return stacked


def jax_forward(layers, inputs):
    """Evaluate per-copy networks on [copies, rows, 2] inputs at full float32 precision."""
    import jax
    import jax.numpy as jnp
    values = inputs
    for layer, weights in enumerate(layers):
        values = jnp.matmul(values, weights["weight"], precision=jax.lax.Precision.HIGHEST) + weights["bias"][:, None]
        if layer < LEAKY_LAYERS:
            values = jax.nn.leaky_relu(values, negative_slope=LEAKY_SLOPE)
        elif layer < len(layers) - 1:
            values = jax.nn.relu(values)
    return values


def jax_features(layers, inputs):
    """Evaluate per-copy networks up to the input of their output layer: the features that layer reads."""
    # Before: inputs [copies, rows, D]; after: activations [copies, rows, width of the last hidden layer].
    import jax
    import jax.numpy as jnp
    values = inputs
    for layer, weights in enumerate(layers[:-1]):
        values = jnp.matmul(values, weights["weight"], precision=jax.lax.Precision.HIGHEST) + weights["bias"][:, None]
        values = jax.nn.leaky_relu(values, negative_slope=LEAKY_SLOPE) if layer < LEAKY_LAYERS else jax.nn.relu(values)
    return values


def jax_errors(predictor, target, inputs):
    """Return the CleanRL rollout bonus: half the summed squared feature error per row."""
    import jax.numpy as jnp
    return 0.5 * jnp.sum((jax_forward(predictor, inputs) - jax_forward(target, inputs)) ** 2, axis=-1)
