"""Checkpointable learner state and independent running statistics."""

from typing import Any, NamedTuple
import math

import jax
import jax.numpy as jnp


class RunningStatistics(NamedTuple):
    """Mean, variance and sample count, each with an independent leading policy axis."""
    mean: Any
    variance: Any
    count: Any


class AdamState(NamedTuple):
    """Float32 first/second moments and the number of completed optimizer steps."""
    first: Any
    second: Any
    step: Any


class PixelTrainState(NamedTuple):
    """Every mutable quantity needed to resume an independent group exactly."""
    environment: Any
    observations: Any
    done: Any
    parameters: Any
    target: Any
    optimizer: AdamState
    observation_statistics: RunningStatistics
    reward_statistics: RunningStatistics
    reward_filter: Any
    iteration: Any
    copy_seeds: Any


class FlatParameterLayout:
    """Pack trainable tensors during an update while preserving the public state structure."""

    def __init__(self, parameters):
        """Record only tensor shapes and the static tree, never parameter values."""
        leaves, self.structure = jax.tree.flatten(parameters)
        self.shapes = tuple(value.shape[1:] for value in leaves)
        self.sizes = tuple(math.prod(shape) for shape in self.shapes)
        self.copies = leaves[0].shape[0]

    def pack(self, parameters):
        """Concatenate each policy's own tensors into one row of a float32 buffer."""
        # Before: kernels [C,8,8,4,32], biases [C,32], ...; after: one [C,P] buffer.
        return jnp.concatenate([value.reshape(self.copies, -1)
                                for value in jax.tree.leaves(parameters)], axis=1)

    def unpack(self, values):
        """Restore named tensors as views of consecutive slices of the flat policy rows."""
        leaves, start = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            leaves.append(values[:, start:start + size].reshape((self.copies,) + shape))
            start += size
        return jax.tree.unflatten(self.structure, leaves)


def initialize_statistics(n_copies, shape):
    """Use gym RunningMeanStd's zero/one initialization and epsilon count."""
    return RunningStatistics(jnp.zeros((n_copies,) + shape, jnp.float64),
                             jnp.ones((n_copies,) + shape, jnp.float64),
                             jnp.full((n_copies,), 1e-4, jnp.float64))


def update_moments(statistics, batch_mean, batch_variance, batch_count):
    """Apply the original parallel-variance identity independently for every policy."""
    # Before: count [C], image means [C,1,84,84]; after: broadcast only inside each policy.
    count = jnp.broadcast_to(jnp.asarray(batch_count, jnp.float64), statistics.count.shape)
    total = statistics.count + count
    broadcast_shape = (count.shape[0],) + (1,) * (statistics.mean.ndim - 1)
    old_count = statistics.count.reshape(broadcast_shape)
    batch_count_array, total_array = count.reshape(broadcast_shape), total.reshape(broadcast_shape)
    delta = batch_mean - statistics.mean
    mean = statistics.mean + delta * batch_count_array / total_array
    variance_sum = (statistics.variance * old_count + batch_variance * batch_count_array
                    + delta ** 2 * old_count * batch_count_array / total_array)
    return RunningStatistics(mean, variance_sum / total_array, total)


def update_observation_statistics(statistics, frames, moment_reduction="two_pass"):
    """Update per-pixel moments from [copies, rows, 1, 84, 84] uint8 frames."""
    values = frames.astype(jnp.float64)
    mean = values.mean(axis=1)
    variance = (values.var(axis=1) if moment_reduction == "two_pass"
                else jnp.maximum((values ** 2).mean(axis=1) - mean ** 2, 0.0))
    return update_moments(statistics, mean, variance, frames.shape[1])


def normalize_frames(frames, statistics):
    """Whiten with float64 statistics, clip to [-5,5], then feed float32 to RND."""
    values = (frames.astype(jnp.float64) - statistics.mean[:, None]) / jnp.sqrt(statistics.variance[:, None])
    return jnp.clip(values, -5.0, 5.0).astype(jnp.float32)


def initialize_adam(parameters):
    """Create float32 moments for trainable parameters only, excluding the frozen target."""
    zeros = jax.tree.map(jnp.zeros_like, parameters)
    return AdamState(zeros, jax.tree.map(jnp.zeros_like, parameters), jnp.zeros((), jnp.int32))


def adam_step(parameters, gradients, optimizer, learning_rates, max_norm, epsilon,
              fuse_optimizer=False):
    """Apply joint policy/predictor clipping per copy, then PyTorch-style Adam."""
    # Packing within each step is measured separately from keeping the whole update flat.
    if fuse_optimizer:
        layout = FlatParameterLayout(parameters)
        flat_optimizer = AdamState(layout.pack(optimizer.first), layout.pack(optimizer.second), optimizer.step)
        flat_parameters, flat_optimizer, norms = adam_step(
            layout.pack(parameters), layout.pack(gradients), flat_optimizer,
            learning_rates, max_norm, epsilon)
        return layout.unpack(flat_parameters), AdamState(layout.unpack(flat_optimizer.first),
            layout.unpack(flat_optimizer.second), flat_optimizer.step), norms

    # A large gradient in one policy cannot reduce another policy's gradient scale.
    copies = learning_rates.shape[0]
    squared_norm = sum((gradient.reshape(copies, -1) ** 2).sum(axis=1)
                       for gradient in jax.tree.leaves(gradients))
    norms = jnp.sqrt(squared_norm)
    scales = jnp.minimum(jnp.asarray(max_norm, jnp.float32) / (norms + 1e-6), 1.0)
    gradients = jax.tree.map(lambda value: value * scales.reshape(
        (copies,) + (1,) * (value.ndim - 1)), gradients)
    step = optimizer.step + 1
    first = jax.tree.map(lambda old, gradient: 0.9 * old + 0.1 * gradient, optimizer.first, gradients)
    second = jax.tree.map(lambda old, gradient: 0.999 * old + 0.001 * gradient ** 2,
                          optimizer.second, gradients)
    correction1 = 1.0 - jnp.asarray(0.9, jnp.float32) ** step
    correction2 = jnp.sqrt(1.0 - jnp.asarray(0.999, jnp.float32) ** step)

    def update(parameter, moment, variance):
        """Broadcast only the policy's learning rate over its own parameters."""
        rate = learning_rates.reshape((copies,) + (1,) * (parameter.ndim - 1))
        return parameter - (rate / correction1) * moment / (jnp.sqrt(variance) / correction2 + epsilon)

    return jax.tree.map(update, parameters, first, second), AdamState(first, second, step), norms
