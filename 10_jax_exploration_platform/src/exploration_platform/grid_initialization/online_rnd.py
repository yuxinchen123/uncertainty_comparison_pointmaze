"""Grid random network distillation: the frozen-field networks, trained online as in CleanRL PPO-RND.

The target 2->32->64->64->512 and predictor 2->32->64->64->512->512->512 are the coordinate
networks that build the frozen random-network field (audit report, page 29). Their weights come
from the same named seeds as `fields.make_random_network_bonus_field`, generated on the processor
so every graphics-card class receives identical numbers; biases start at zero, as in CleanRL.
"""

import math

import numpy as np

from .environments import GridWorld
from .fields import _archive_field, derive_quantity_seed

WIDTHS = {"target": (32, 64, 64, 512), "predictor": (32, 64, 64, 512, 512, 512)}
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


def numpy_forward(layers, inputs):
    """Evaluate one network in float64 with the CleanRL activation order."""
    values = np.asarray(inputs, np.float64)
    for layer, weight in enumerate(layers):
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


def frozen_network_field(geometry: GridWorld, field_seed: int, statistics=None):
    """Return the frozen random-network field on raw or whitened coordinates, mean-normalized."""
    networks = network_weights(field_seed)
    inputs = (geometry.coordinates if statistics is None else
              normalize_coordinates(geometry.coordinates, statistics["mean"], statistics["variance"]))
    errors = 0.5 * np.sum((numpy_forward(networks["predictor"], inputs)
                           - numpy_forward(networks["target"], inputs)) ** 2, axis=-1)
    metadata = {"kind": "random_network", "seed": int(field_seed), "state_count": geometry.state_count,
                "generator": "online_rnd.frozen_network_field (processor weights, float64 evaluation)",
                "target_widths": [2, *WIDTHS["target"]], "predictor_widths": [2, *WIDTHS["predictor"]],
                "weight_initialization": "orthogonal, gain sqrt(2)", "bias_initialization": "zero",
                "input_normalization": "none" if statistics is None else
                "whitened by random-action warm-up statistics, clipped to [-5, 5]",
                "input_statistics": None if statistics is None else
                {key: np.asarray(value).tolist() for key, value in statistics.items()},
                "error_reduction": "one half times sum of squared feature errors, as in CleanRL",
                "seed_scheme": "sha256_named_quantities_v1", "trained": False}
    return _archive_field(errors, metadata)


def stack_networks(field_seeds):
    """Stack per-copy network weights, with zero biases, for the copies' field seeds."""
    # Before: field seeds [0,0,1]; after: {"target": ({"weight": [3,2,32], "bias": [3,32]}, ...)}.
    cache = {int(seed): network_weights(int(seed)) for seed in set(int(seed) for seed in field_seeds)}
    stacked = {}
    for name, widths in WIDTHS.items():
        stacked[name] = tuple(
            {"weight": np.stack([cache[int(seed)][name][layer] for seed in field_seeds]),
             "bias": np.zeros((len(field_seeds), output), np.float32)}
            for layer, output in enumerate(widths))
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


def jax_errors(predictor, target, inputs):
    """Return the CleanRL rollout bonus: half the summed squared feature error per row."""
    import jax.numpy as jnp
    return 0.5 * jnp.sum((jax_forward(predictor, inputs) - jax_forward(target, inputs)) ** 2, axis=-1)
