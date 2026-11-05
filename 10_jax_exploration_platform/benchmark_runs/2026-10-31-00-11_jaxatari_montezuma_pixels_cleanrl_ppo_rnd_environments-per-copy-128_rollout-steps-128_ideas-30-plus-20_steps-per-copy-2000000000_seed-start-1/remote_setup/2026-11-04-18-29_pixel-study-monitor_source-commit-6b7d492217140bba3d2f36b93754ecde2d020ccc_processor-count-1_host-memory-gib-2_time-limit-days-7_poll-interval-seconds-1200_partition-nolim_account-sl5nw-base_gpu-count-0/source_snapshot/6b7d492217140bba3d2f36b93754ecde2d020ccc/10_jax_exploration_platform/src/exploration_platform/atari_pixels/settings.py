"""Resolve explicit per-policy scalars without changing the default CleanRL arithmetic."""

import math

import jax.numpy as jnp


def scalar_setting(configuration, name, shape=None):
    """Keep an unspecified scalar as a Python constant; broadcast only explicit policy vectors."""
    values = configuration.per_copy_values(name)
    if values is None:
        return getattr(configuration, name)
    result = jnp.asarray(values, jnp.float32)
    return result.reshape(shape) if shape is not None else result


def scalar_product(configuration, names, shape=None):
    """Round a product of configured scalar constants once, as the original expression does."""
    explicit = [configuration.per_copy_values(name) for name in names]
    if all(values is None for values in explicit):
        return math.prod(getattr(configuration, name) for name in names)
    # Multiplying the Python constants before creating a float32 vector preserves scalar rounding.
    values = tuple(math.prod(values[index] if values is not None else getattr(configuration, name)
                             for name, values in zip(names, explicit))
                   for index in range(configuration.n_copies))
    result = jnp.asarray(values, jnp.float32)
    return result.reshape(shape) if shape is not None else result
