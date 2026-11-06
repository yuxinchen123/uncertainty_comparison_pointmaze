"""Independent-copy CNNs matching cleanrl/ppo_rnd_envpool.py at the recorded commit."""

import hashlib
import math

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


def dense(layer, values, batching="matmul"):
    """Apply independent [input, output] dense weights to [copies, rows, input]."""
    product = (jnp.matmul(values, layer["weight"], precision=jax.lax.Precision.HIGHEST)
               if batching == "matmul" else jnp.einsum("cbi,cio->cbo", values,
                   layer["weight"], precision=jax.lax.Precision.HIGHEST))
    return product + layer["bias"][:, None]


def image_patches(images, kernel, stride):
    """Copy every kernel window of [B,H,W,C] images into [B,OH,OW,kernel*kernel*C] rows.

    Strided slices only copy values, so each row holds exactly the pixels one output position
    reads. The row order (kernel row, kernel column, channel) equals a flattened HWIO kernel.
    """
    rows, height, width, channels = images.shape
    out_height, out_width = (height - kernel) // stride + 1, (width - kernel) // stride + 1
    windows = [images[:, i:i + stride * (out_height - 1) + 1:stride,
                      j:j + stride * (out_width - 1) + 1:stride, :]
               for i in range(kernel) for j in range(kernel)]
    return jnp.stack(windows, axis=3).reshape(rows, out_height, out_width, kernel * kernel * channels)


# Copied image windows are built for at most this many rows at once (memory: one 1024-row chunk
# of the first policy layer is 1024 x 400 x 256 float32 values, 0.42 GB).
PATCH_ROWS_PER_CHUNK = 1024


def row_chunks(rows):
    """Split a row count into equal chunks of at most PATCH_ROWS_PER_CHUNK rows."""
    chunk = math.gcd(rows, PATCH_ROWS_PER_CHUNK)
    return rows // chunk, chunk


def space_to_depth(images, block):
    """Move each block x block pixel square into channels: [B,H,W,C] -> [B,H/b,W/b,b*b*C]."""
    rows, height, width, channels = images.shape
    height, width = height - height % block, width - width % block
    images = images[:, :height, :width].reshape(rows, height // block, block, width // block, block, channels)
    return images.transpose(0, 1, 3, 2, 4, 5).reshape(rows, height // block, width // block,
                                                      block * block * channels)


def space_to_depth_kernel(weight, block):
    """Rearrange an HWIO kernel to match space_to_depth images: [k,k,C,O] -> [k/b,k/b,b*b*C,O]."""
    kernel, _, channels, outputs = weight.shape
    weight = weight.reshape(kernel // block, block, kernel // block, block, channels, outputs)
    return weight.transpose(0, 2, 1, 3, 4, 5).reshape(kernel // block, kernel // block,
                                                      block * block * channels, outputs)


def convolution(layer, images, stride, layout="nchw", batching="vmap", include_bias=True,
                algorithm="compiler"):
    """Apply independent HWIO kernels to each policy's NHWC image batch.

    algorithm selects how one copy's convolution and its gradients are computed; every choice
    evaluates the same float32 sums at highest precision (only the summation order differs):
    - compiler: the compiler's convolution for the forward pass and both gradients.
    - patch_weight_gradient: compiler forward pass and input gradient; the weight gradient is
      one matrix product of copied image windows and output gradients.
    - patch_matrix: forward pass is that matrix product; automatic differentiation then gives
      matrix-product weight and input gradients.
    - space_to_depth: strided layers become stride-1 convolutions on rearranged pixels.
    """
    # Before: [copies, rows, H, W, channels]; after: independent output images per copy.
    def compiler_convolve(image_batch, weight, groups=1, stride=stride):
        """Choose the compiler's NCHW or NHWC convolution path without changing image values."""
        if layout == "nchw":
            result = jax.lax.conv_general_dilated(
                image_batch.transpose(0, 3, 1, 2), weight.transpose(3, 2, 0, 1),
                (stride, stride), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW"),
                feature_group_count=groups, precision=jax.lax.Precision.HIGHEST)
            return result.transpose(0, 2, 3, 1)
        return jax.lax.conv_general_dilated(image_batch, weight, (stride, stride), "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=groups,
            precision=jax.lax.Precision.HIGHEST)

    @jax.custom_vjp
    def patch_weight_gradient_convolve(image_batch, weight):
        """Compiler convolution whose weight gradient is an explicit matrix product."""
        return compiler_convolve(image_batch, weight)

    def patch_forward(image_batch, weight):
        """Keep the inputs; the windows are copied again only inside the backward pass."""
        return compiler_convolve(image_batch, weight), (image_batch, weight)

    def patch_backward(residual, output_gradient):
        """Input gradient from the compiler; weight gradient as windows^T @ output gradient."""
        image_batch, weight = residual
        _, input_vjp = jax.vjp(lambda values: compiler_convolve(values, weight), image_batch)
        (input_gradient,) = input_vjp(output_gradient)
        count, chunk = row_chunks(image_batch.shape[0])

        def accumulate(total, index):
            """Add one row chunk's windows^T @ output gradient."""
            images = jax.lax.dynamic_slice_in_dim(image_batch, index * chunk, chunk, 0)
            gradient = jax.lax.dynamic_slice_in_dim(output_gradient, index * chunk, chunk, 0)
            windows = image_patches(images, weight.shape[0], stride)
            return total + jnp.einsum("byxp,byxo->po", windows, gradient,
                                      precision=jax.lax.Precision.HIGHEST), None

        weight_gradient, _ = jax.lax.scan(
            accumulate, jnp.zeros((weight.size // weight.shape[-1], weight.shape[-1]), weight.dtype),
            jnp.arange(count))
        return input_gradient, weight_gradient.reshape(weight.shape)

    patch_weight_gradient_convolve.defvjp(patch_forward, patch_backward)

    def convolve(image_batch, weight):
        """Dispatch one copy's convolution to the selected algorithm."""
        if algorithm == "compiler":
            return compiler_convolve(image_batch, weight)
        if algorithm == "patch_weight_gradient":
            return patch_weight_gradient_convolve(image_batch, weight)
        if algorithm == "patch_matrix":
            count, chunk = row_chunks(image_batch.shape[0])

            @jax.checkpoint
            def chunk_product(images, weight):
                """One row chunk; its windows are rebuilt in the backward pass, not stored."""
                return jnp.einsum("byxp,po->byxo", image_patches(images, weight.shape[0], stride),
                                  weight.reshape(-1, weight.shape[-1]),
                                  precision=jax.lax.Precision.HIGHEST)

            chunks = image_batch.reshape(count, chunk, *image_batch.shape[1:])
            result = jax.lax.map(lambda images: chunk_product(images, weight), chunks)
            return result.reshape(image_batch.shape[0], *result.shape[2:])
        if algorithm == "space_to_depth":
            if stride == 1:
                return compiler_convolve(image_batch, weight)
            return compiler_convolve(space_to_depth(image_batch, stride),
                                     space_to_depth_kernel(weight, stride), stride=1)
        raise ValueError(f"unknown convolution algorithm {algorithm}")

    def one_copy(image_batch, weight, bias):
        """Convolve one policy's images at full float32 matrix precision."""
        result = convolve(image_batch, weight)
        return result + bias if include_bias else result
    if batching == "vmap":
        return jax.vmap(one_copy)(images, layer["weight"], layer["bias"])
    if batching == "sequential":
        # Before: one grouped convolution over copies; after: copies one after another.
        return jax.lax.map(lambda values: one_copy(*values), (images, layer["weight"], layer["bias"]))
    if algorithm != "compiler":
        raise ValueError("grouped convolution batching supports only the compiler algorithm")
    # Before: C separate [B,H,W,I] batches; after: [B,H,W,C*I] with C feature groups.
    copies, rows, height, width, channels = images.shape
    packed_images = images.transpose(1, 2, 3, 0, 4).reshape(rows, height, width, copies * channels)
    weights = layer["weight"]
    packed_weights = weights.transpose(1, 2, 3, 0, 4).reshape(
        weights.shape[1], weights.shape[2], channels, copies * weights.shape[-1])
    result = compiler_convolve(packed_images, packed_weights, copies)
    result = result.reshape(rows, result.shape[1], result.shape[2], copies, weights.shape[-1])
    result = result.transpose(3, 0, 1, 2, 4)
    return result + layer["bias"][:, None, None, None, :] if include_bias else result


def convolution_trunk(layers, frames, leaky=False, configuration=None):
    """Run the three CleanRL convolutions and flatten in PyTorch's NCHW order."""
    images = frames.transpose(0, 1, 3, 4, 2)
    for layer, stride in zip(layers, (4, 2, 1)):
        fused = configuration is not None and configuration.fuse_network_bias_and_activation
        images = convolution(layer, images, stride,
            layout=configuration.network_layout if configuration is not None else "nchw",
            batching=configuration.convolution_batching if configuration is not None else "vmap",
            include_bias=not fused,
            algorithm=configuration.convolution_algorithm if configuration is not None else "compiler")
        if fused:
            from .kernels import bias_activation
            images = bias_activation(images, layer["bias"], leaky)
        else:
            # PyTorch LeakyReLU uses the negative slope at exactly zero as well.
            images = jnp.where(images > 0., images, images * .01) if leaky else jax.nn.relu(images)
    # Before: [C,B,7,7,64]; after: [C,B,3136] with channel-major PyTorch ordering.
    return images.transpose(0, 1, 4, 2, 3).reshape(images.shape[0], images.shape[1], -1)


def policy_features(parameters, observations, configuration=None):
    """Compute the common CNN/dense representation without the actor or value heads."""
    hidden = convolution_trunk(parameters["convolutions"], observations.astype(jnp.float32) / 255.0,
                                configuration=configuration)
    for layer in parameters["dense"]:
        hidden = jax.nn.relu(dense(layer, hidden, configuration.dense_batching if configuration else "matmul"))
    return hidden


def actor_logits(parameters, features, configuration=None):
    """Apply the two actor layers to already computed policy features."""
    batching = configuration.dense_batching if configuration is not None else "matmul"
    return dense(parameters["actor"][1], jax.nn.relu(dense(parameters["actor"][0], features, batching)), batching)


def critic_values(parameters, hidden, configuration=None):
    """Apply the residual value layers to a rollout step or the complete rollout."""
    batching = configuration.dense_batching if configuration is not None else "matmul"
    features = hidden + jax.nn.relu(dense(parameters["extra"], hidden, batching))
    return dense(parameters["value_ext"], features, batching)[..., 0], dense(parameters["value_int"], features, batching)[..., 0]


def policy_forward(parameters, observations, configuration=None):
    """Return action logits and both value heads from one shared CNN pass."""
    hidden = policy_features(parameters, observations, configuration)
    values_ext, values_int = critic_values(parameters, hidden, configuration)
    return actor_logits(parameters, hidden, configuration), values_ext, values_int


def rnd_forward(parameters, normalized_frames, configuration=None):
    """Evaluate one RND network; its final dense layer has no activation."""
    hidden = convolution_trunk(parameters["convolutions"], normalized_frames, leaky=True,
                                configuration=configuration)
    batching = configuration.dense_batching if configuration is not None else "matmul"
    for layer in parameters["dense"][:-1]:
        hidden = jax.nn.relu(dense(layer, hidden, batching))
    return dense(parameters["dense"][-1], hidden, batching)


def categorical_statistics(logits, actions):
    """Compute chosen-action log probability and categorical entropy in float32."""
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probabilities, actions[..., None], axis=-1)[..., 0]
    entropy = -(jnp.exp(log_probabilities) * log_probabilities).sum(axis=-1)
    return chosen, entropy
