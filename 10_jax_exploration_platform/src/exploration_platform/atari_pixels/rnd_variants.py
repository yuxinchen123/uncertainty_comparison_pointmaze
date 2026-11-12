"""Opt-in count-free neural RND variants for pixel Montezuma.

The default CleanRL image bonus and MSE objective are unchanged.  These
functions operate only on network features, never visit counts or room IDs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def target_features(target_output, *, uniform_start=False, epsilon=1e-8):
    """Normalize the frozen target vector per image, only for the U ablation.

    With a zero predictor output head, the initial half-squared-L2 bonus is
    1/2 for every frame with a nonzero frozen target feature vector.
    """
    if not uniform_start:
        return target_output
    norm = jnp.sqrt(jnp.sum(jnp.square(target_output), axis=-1, keepdims=True))
    return target_output / jnp.maximum(norm, epsilon)


def predictor_error(prediction, target, *, mode="mse", exponent=2.0,
                    decay_exponent=1.0, epsilon=1e-8):
    """Return per-image predictor objectives with exact default MSE semantics.

    mse:
       mean over output features of squared prediction error (CleanRL).
    power:
       ||target - prediction||_2^(2*exponent) / exponent,
       an explicit *sum*-norm objective, to be tuned with predictor-only SGD.
    shrink:
       Regression toward the detached residual-adaptive target
       target - q*(target-prediction), with
       q=(1+||r||_2^(2/decay_exponent))^(-decay_exponent/2).
       This is count-free.  Its analytical decay law requires exact,
       independent-state regression and is NOT assumed for a CNN.
    """
    error = target - prediction
    if mode == "mse":
        return jnp.mean(jnp.square(error), axis=-1)
    squared = jnp.sum(jnp.square(error), axis=-1)
    if mode == "power":
        return (jnp.maximum(squared, epsilon) ** exponent) / exponent
    if mode == "shrink":
        # jnp.maximum prevents derivative singularities when decay_exponent<1.
        q = (1.0 + jnp.maximum(squared, epsilon) ** (1.0 / decay_exponent)) ** (
            -0.5 * decay_exponent)
        regression_target = jax.lax.stop_gradient(target - q[..., None] * error)
        return jnp.mean(jnp.square(prediction - regression_target), axis=-1)
    raise ValueError(f"Unknown neural RND predictor loss mode: {mode!r}")


def initial_bonus(prediction, target):
    """Raw CleanRL-style half sum of squared feature errors."""
    return 0.5 * jnp.sum(jnp.square(target - prediction), axis=-1)
