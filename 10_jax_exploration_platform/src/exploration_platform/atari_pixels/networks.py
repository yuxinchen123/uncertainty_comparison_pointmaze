"""Independent-copy CNNs matching cleanrl/ppo_rnd_envpool.py at the recorded commit."""

import hashlib

import jax
import jax.numpy as jnp


def stream_identifier(name):
    """Give each named random quantity an independent, stable SHA-256 identifier."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


def quantity_keys(copy_seeds, name, *coordinates):
    """Derive one key per policy without depending on the number of other policies."""
    # Before: seeds [7, 11]; after: two independent keys for this named quantity only.
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(copy_seeds, jnp.uint32))
    keys = jax.vmap(lambda key: jax.random.fold_in(key, stream_identifier(name)))(keys)
    for coordinate in coordinates:
        keys = jax.vmap(lambda key: jax.random.fold_in(key, coordinate))(keys)
    return keys


def _orthogonal(key, output_size, input_size, gain):
    """Match PyTorch orthogonal initialization on an [output, flattened-input] matrix."""
    # Preserve PyTorch's orientation before converting convolution weights to HWIO.
    shape = (max(output_size, input_size), min(output_size, input_size))
    matrix = jax.random.normal(key, shape, dtype=jnp.float32)
    orthogonal, triangular = jnp.linalg.qr(matrix, mode="reduced")
    orthogonal = orthogonal * jnp.sign(jnp.diag(triangular))[None, :]
    orthogonal = orthogonal.T if output_size < input_size else orthogonal
    return orthogonal * jnp.asarray(gain, jnp.float32)


def initialize_layer(copy_seeds, name, output_size, input_size, gain=2 ** 0.5,
                     kernel_size=None):
    """Initialize a dense or convolutional layer with independent weights per policy."""
    # Before: OIHW [32,4,8,8]; after: HWIO [8,8,4,32], with the same flattened rows.
    flattened_input = input_size if kernel_size is None else input_size * kernel_size ** 2
    keys = quantity_keys(copy_seeds, name)
    weights = jax.vmap(lambda key: _orthogonal(key, output_size, flattened_input, gain))(keys)
    if kernel_size is None:
        weights = weights.transpose(0, 2, 1)
    else:
        weights = weights.reshape(-1, output_size, input_size, kernel_size, kernel_size)
        weights = weights.transpose(0, 3, 4, 2, 1)
    return {"weight": weights, "bias": jnp.zeros((len(copy_seeds), output_size), jnp.float32)}


def initialize_networks(copy_seeds, action_count):
    """Return trainable policy/predictor parameters and a separate frozen RND target."""
    # Use distinct names for every layer so adding a layer cannot move another layer's draws.
    def layer(name, output_size, input_size, gain=2 ** 0.5, kernel_size=None):
        """Build one named layer from the shared policy-seed list."""
        return initialize_layer(copy_seeds, name, output_size, input_size, gain, kernel_size)

    agent = {
        "convolutions": tuple(layer(f"policy/convolution/{i}", out, inp, kernel_size=size)
                              for i, (inp, out, size) in enumerate(((4, 32, 8), (32, 64, 4),
                                                                   (64, 64, 3)))),
        "dense": (layer("policy/dense/0", 256, 3136), layer("policy/dense/1", 448, 256)),
        "extra": layer("policy/extra", 448, 448, 0.1),
        "actor": (layer("policy/actor/0", 448, 448, 0.01),
                  layer("policy/actor/1", action_count, 448, 0.01)),
        "value_ext": layer("policy/value/extrinsic", 1, 448, 0.01),
        "value_int": layer("policy/value/intrinsic", 1, 448, 0.01),
    }
    predictor = {
        "convolutions": tuple(layer(f"predictor/convolution/{i}", out, inp, kernel_size=size)
                              for i, (inp, out, size) in enumerate(((1, 32, 8), (32, 64, 4),
                                                                   (64, 64, 3)))),
        "dense": (layer("predictor/dense/0", 512, 3136),
                  layer("predictor/dense/1", 512, 512),
                  layer("predictor/dense/2", 512, 512)),
    }
    target = {
        "convolutions": tuple(layer(f"target/convolution/{i}", out, inp, kernel_size=size)
                              for i, (inp, out, size) in enumerate(((1, 32, 8), (32, 64, 4),
                                                                   (64, 64, 3)))),
        "dense": (layer("target/dense/0", 512, 3136),),
    }
    return {"agent": agent, "predictor": predictor}, target


def dense(layer, values):
    """Apply independent [input, output] dense weights to [copies, rows, input]."""
    return jnp.matmul(values, layer["weight"], precision=jax.lax.Precision.HIGHEST) + layer["bias"][:, None]


def convolution(layer, images, stride):
    """Apply independent HWIO kernels to each policy's NHWC image batch."""
    # Before: [copies, rows, H, W, channels]; after: independent output images per copy.
    def one_copy(image_batch, weight, bias):
        """Convolve one policy's images at full float32 matrix precision."""
        result = jax.lax.conv_general_dilated(
            image_batch, weight, (stride, stride), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"),
            precision=jax.lax.Precision.HIGHEST)
        return result + bias
    return jax.vmap(one_copy)(images, layer["weight"], layer["bias"])


def convolution_trunk(layers, frames, leaky=False):
    """Run the three CleanRL convolutions and flatten in PyTorch's NCHW order."""
    images = frames.transpose(0, 1, 3, 4, 2)
    for layer, stride in zip(layers, (4, 2, 1)):
        images = convolution(layer, images, stride)
        images = jax.nn.leaky_relu(images, negative_slope=0.01) if leaky else jax.nn.relu(images)
    # Before: [C,B,7,7,64]; after: [C,B,3136] with channel-major PyTorch ordering.
    return images.transpose(0, 1, 4, 2, 3).reshape(images.shape[0], images.shape[1], -1)


def policy_forward(parameters, observations):
    """Return action logits and both value heads from one shared CNN pass."""
    hidden = convolution_trunk(parameters["convolutions"], observations.astype(jnp.float32) / 255.0)
    for layer in parameters["dense"]:
        hidden = jax.nn.relu(dense(layer, hidden))
    logits = dense(parameters["actor"][1], jax.nn.relu(dense(parameters["actor"][0], hidden)))
    features = hidden + jax.nn.relu(dense(parameters["extra"], hidden))
    return logits, dense(parameters["value_ext"], features)[..., 0], dense(parameters["value_int"], features)[..., 0]


def rnd_forward(parameters, normalized_frames):
    """Evaluate one RND network; its final dense layer has no activation."""
    hidden = convolution_trunk(parameters["convolutions"], normalized_frames, leaky=True)
    for layer in parameters["dense"][:-1]:
        hidden = jax.nn.relu(dense(layer, hidden))
    return dense(parameters["dense"][-1], hidden)


def categorical_statistics(logits, actions):
    """Compute chosen-action log probability and categorical entropy in float32."""
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[..., 0]
    entropy = -(jnp.exp(log_probabilities) * log_probabilities).sum(axis=-1)
    return chosen, entropy
