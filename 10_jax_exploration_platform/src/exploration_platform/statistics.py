"""Per-copy running mean and variance, held in float64 (algorithm specification 4.2).

Two parts of the platform keep such statistics and neither owns the other: the agent normalises
the intrinsic return stream, and the random-network-distillation bonus whitens its own input. So
the formula lives here, in a module that depends on nothing else.
"""
from typing import NamedTuple

import jax.numpy as jnp


class RMSState(NamedTuple):
    """Per-copy running mean/variance/count, float64 (gymnasium parallel-variance formula)."""
    mean: jnp.ndarray   # [C, dim]
    var: jnp.ndarray    # [C, dim]
    count: jnp.ndarray  # [C, 1]


def rms_init(n_copies: int, dim: int) -> RMSState:
    """Fresh statistics: mean 0, var 1, count 1e-4 (gymnasium epsilon)."""
    return RMSState(jnp.zeros((n_copies, dim), jnp.float64),
                    jnp.ones((n_copies, dim), jnp.float64),
                    jnp.full((n_copies, 1), 1e-4, jnp.float64))


def rms_update(rms: RMSState, batch, batch_stats_f32: bool = False) -> RMSState:
    """Parallel-variance update per copy; batch [C, B, dim] float32, population variance.

    The accumulators are always float64. batch_stats_f32 controls whether the batch's own mean
    and variance are reduced in float32 first and then promoted, instead of promoting the whole
    [C, B, dim] batch — measured as a round-2 experiment in 09_parallelization, since the
    promotion is the only float64 work of any size in the compiled iteration.
    """
    # before: batch [C, B, dim] float32; after: batch_mean/batch_var [C, dim] float64
    if batch_stats_f32:
        batch_mean = batch.mean(axis=1).astype(jnp.float64)
        batch_var = batch.var(axis=1, ddof=0).astype(jnp.float64)
        batch_count = float(batch.shape[1])
    else:
        b = batch.astype(jnp.float64)
        batch_mean = b.mean(axis=1)
        batch_var = b.var(axis=1, ddof=0)
        batch_count = float(b.shape[1])

    # merge the batch into the accumulator (Chan's parallel variance)
    delta = batch_mean - rms.mean
    tot = rms.count + batch_count
    new_mean = rms.mean + delta * batch_count / tot
    m2 = rms.var * rms.count + batch_var * batch_count + delta ** 2 * rms.count * batch_count / tot
    return RMSState(new_mean, m2 / tot, tot)
