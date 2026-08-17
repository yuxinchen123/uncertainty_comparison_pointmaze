"""The two random-network-distillation networks and the whitening their input goes through.

The target is drawn once and never trained; the predictor is trained to reproduce its output, so
the squared error between them is large where the agent has not been. Both take the whitened
next observation, [C, M, 4], and return [C, M, feature_dim].
"""
import jax
import jax.numpy as jnp

from ... import F32
from ...networks import stacked_orthogonal


def init_target(rnd_cfg, n_copies: int, base_seed: int, copy_seed_index):
    """The fixed target network: two layers, never trained, one draw per copy."""
    return stacked_orthogonal(
        "rnd_target",
        [(rnd_cfg.hidden, 4, 2 ** 0.5), (rnd_cfg.feature_dim, rnd_cfg.hidden, 2 ** 0.5)],
        n_copies, base_seed, copy_seed_index)


def init_predictor(rnd_cfg, n_copies: int, base_seed: int, copy_seed_index):
    """The trained predictor: three layers, one draw per copy."""
    return stacked_orthogonal(
        "rnd_predictor",
        [(rnd_cfg.hidden, 4, 2 ** 0.5), (rnd_cfg.feature_dim, rnd_cfg.hidden, 2 ** 0.5),
         (rnd_cfg.feature_dim, rnd_cfg.feature_dim, 2 ** 0.5)],
        n_copies, base_seed, copy_seed_index)


def features(target, predictor, x):
    """(target features, predictor features), each [C, M, feature_dim].

    The target's output carries no gradient by construction, so the predictor's loss can never
    move the thing it is chasing.
    """
    th = jax.nn.relu(jnp.matmul(x, target["W0"]) + target["b0"][:, None, :])
    tf = jnp.matmul(th, target["W1"]) + target["b1"][:, None, :]
    ph = jax.nn.relu(jnp.matmul(x, predictor["W0"]) + predictor["b0"][:, None, :])
    ph = jax.nn.relu(jnp.matmul(ph, predictor["W1"]) + predictor["b1"][:, None, :])
    pf = jnp.matmul(ph, predictor["W2"]) + predictor["b2"][:, None, :]
    return jax.lax.stop_gradient(tf), pf


def whiten(obs, obs_rms):
    """Centre and scale the network input by the running statistics, clipped to +-5.

    The statistics are float64; they are cast to float32 first, which is what the PyTorch twin
    this implementation was checked against does.
    """
    mean = obs_rms.mean.astype(F32)[:, None, :]
    std = jnp.sqrt(obs_rms.var + 1e-8).astype(F32)[:, None, :]
    return jnp.clip((obs - mean) / std, -5.0, 5.0)
