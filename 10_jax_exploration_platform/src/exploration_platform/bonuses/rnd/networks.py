"""The two random-network-distillation networks and the whitening their input goes through.

The target is drawn once and never trained; the predictor is trained to reproduce its output, so
the squared error between them is large where the agent has not been. Both take the whitened
next observation, [C, M, 4], and return [C, M, feature_dim]. With the random Fourier features
(`init_fourier_features`, `fourier_features`) both take those features of the whitened
observation instead, [C, M, D].
"""
import jax
import jax.numpy as jnp

from ... import F32
from ...networks import NET_IDS, stacked_orthogonal


def init_target(rnd_cfg, n_copies: int, base_seed: int, copy_seed_index, input_dim: int = 4):
    """The fixed target network: two layers, never trained, one draw per copy."""
    return stacked_orthogonal(
        "rnd_target",
        [(rnd_cfg.hidden, input_dim, 2 ** 0.5), (rnd_cfg.feature_dim, rnd_cfg.hidden, 2 ** 0.5)],
        n_copies, base_seed, copy_seed_index)


def init_predictor(rnd_cfg, n_copies: int, base_seed: int, copy_seed_index, input_dim: int = 4):
    """The trained predictor: three layers, one draw per copy."""
    return stacked_orthogonal(
        "rnd_predictor",
        [(rnd_cfg.hidden, input_dim, 2 ** 0.5), (rnd_cfg.feature_dim, rnd_cfg.hidden, 2 ** 0.5),
         (rnd_cfg.feature_dim, rnd_cfg.feature_dim, 2 ** 0.5)],
        n_copies, base_seed, copy_seed_index)


def copy_stream_key(stream: str, base_seed: int, seed_index):
    """The key of one named random stream of one copy: (base seed, stream id, 0, seed index).

    The same path `stacked_orthogonal` folds for a network's layer 0, under the stream's own id,
    so no stream's draws depend on another's or on how many copies the run has.
    """
    return jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(
        jax.random.PRNGKey(base_seed), NET_IDS[stream]), 0), seed_index)


def init_fourier_features(rnd_cfg, n_copies: int, base_seed: int, copy_seed_index,
                          obs_dim: int = 4):
    """The fixed random Fourier features, one draw per copy, never trained.

    Row d of Omega is a standard normal vector divided by the length scale, and phase d is
    uniform on [0, 2 pi). The frequencies and the phases have a stream each, and inside a stream
    every feature d has its own key, so a larger feature count appends features and moves none,
    and another length scale rescales the same normal draws.

    before: 3 copies, obs_dim 4, fourier_feature_count 2, fourier_length_scale 0.5
    after:  {"frequencies": [3, 4, 2] (Omega transposed, so a forward pass is `x @ frequencies`;
             standard normals x 2), "phases": [3, 2] in [0, 2 pi)}
    """
    feature_index = jnp.arange(rnd_cfg.fourier_feature_count, dtype=jnp.uint32)
    frequencies, phases = [], []
    for c in range(n_copies):
        frequency_key = copy_stream_key("rnd_fourier_frequencies", base_seed, copy_seed_index[c])
        phase_key = copy_stream_key("rnd_fourier_phases", base_seed, copy_seed_index[c])
        # one key per feature: standard normals [D, obs] and phases [D]
        normals = jax.vmap(lambda d: jax.random.normal(
            jax.random.fold_in(frequency_key, d), (obs_dim,), F32))(feature_index)
        phases.append(jax.vmap(lambda d: jax.random.uniform(
            jax.random.fold_in(phase_key, d), (), F32, 0.0, 2.0 * jnp.pi))(feature_index))
        frequencies.append((normals / rnd_cfg.fourier_length_scale).T)
    return {"frequencies": jnp.stack(frequencies), "phases": jnp.stack(phases)}


def fourier_features(fourier, x):
    """Random Fourier features of whitened inputs [C, M, obs] -> [C, M, D], float32:
    phi(x) = sqrt(2 / D) cos(x Omega^T + beta).

    Two inputs at distance d have features with inner product about exp(-d^2 / (2 l^2)), l the
    length scale the frequencies were divided by, up to an error of about 1 / sqrt(D): inputs a
    few l apart share almost nothing. Every feature lies within +-sqrt(2 / D) whatever the size
    of the input.

    before: x = [0, 0, 0, 0], D = 2, phases [0, pi]  ; after: [1, -1]
    """
    count = fourier["phases"].shape[-1]
    angle = jnp.matmul(x, fourier["frequencies"]) + fourier["phases"][:, None, :]
    return (2.0 / count) ** 0.5 * jnp.cos(angle)


def target_features(target, x):
    """The fixed target's features, [C, M, feature_dim].

    They carry no gradient by construction, so the predictor's loss can never move the thing it
    is chasing.
    """
    th = jax.nn.relu(jnp.matmul(x, target["W0"]) + target["b0"][:, None, :])
    tf = jnp.matmul(th, target["W1"]) + target["b1"][:, None, :]
    return jax.lax.stop_gradient(tf)


def predictor_features(predictor, x):
    """The trained predictor's features, [C, M, feature_dim]."""
    ph = jax.nn.relu(jnp.matmul(x, predictor["W0"]) + predictor["b0"][:, None, :])
    ph = jax.nn.relu(jnp.matmul(ph, predictor["W1"]) + predictor["b1"][:, None, :])
    return jnp.matmul(ph, predictor["W2"]) + predictor["b2"][:, None, :]


def features(target, predictor, x):
    """(target features, predictor features), each [C, M, feature_dim]."""
    return target_features(target, x), predictor_features(predictor, x)


def predictor_hidden(predictor, x):
    """The predictor's last hidden layer, [C, M, feature_dim]: what its output layer reads."""
    ph = jax.nn.relu(jnp.matmul(x, predictor["W0"]) + predictor["b0"][:, None, :])
    return jax.nn.relu(jnp.matmul(ph, predictor["W1"]) + predictor["b1"][:, None, :])


def unit_length(x):
    """Each vector along the last axis scaled to length 1.

    before: [3.0, 4.0] ; after: [0.6, 0.8]
    The squared length is kept at or above 1e-24 only so that a vector of zeros stays zeros
    instead of becoming not-a-number; a network output is never that short otherwise.
    """
    return x / jnp.sqrt(jnp.maximum((x * x).sum(axis=-1, keepdims=True), 1e-24))


def whiten(obs, obs_rms, clip: float = 5.0):
    """Centre and scale the network input by the running statistics, clipped to +-clip.

    The statistics are float64; they are cast to float32 first, which is what the PyTorch twin
    this implementation was checked against does. A clip of 0 means no clip.

    before: obs 12 with mean 0 and variance 1
    after:  5 at the default clip, 10 at clip 10, 12 at clip 0
    """
    mean = obs_rms.mean.astype(F32)[:, None, :]
    std = jnp.sqrt(obs_rms.var + 1e-8).astype(F32)[:, None, :]
    scaled = (obs - mean) / std
    return scaled if clip == 0 else jnp.clip(scaled, -clip, clip)
