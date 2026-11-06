"""The alternative convolution algorithms compute the CleanRL trunk's values and gradients.

Each algorithm is compared with the compiler's convolution on the three CleanRL layer shapes,
for one and two independent copies, at full float32 precision. Run on a compute node.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.networks import (convolution, image_patches,
                                                        space_to_depth, space_to_depth_kernel)

jax.config.update("jax_default_matmul_precision", "highest")
LAYERS = ((4, 32, 8, 4, 84), (32, 64, 4, 2, 20), (64, 64, 3, 1, 9))


def layer_inputs(copies, channels, outputs, kernel, size, rows=6):
    """Random images and weights with a separate key for each quantity."""
    key = jax.random.PRNGKey(0)
    images = jax.random.uniform(jax.random.fold_in(key, 1), (copies, rows, size, size, channels))
    weight = jax.random.normal(jax.random.fold_in(key, 2), (copies, kernel, kernel, channels, outputs)) * 0.1
    bias = jax.random.normal(jax.random.fold_in(key, 3), (copies, outputs)) * 0.1
    return images, {"weight": weight, "bias": bias}


def loss_and_gradients(algorithm, batching, images, layer, stride):
    """A scalar of every output value, so all weight and input positions receive gradient."""
    def loss(values, parameters):
        output = convolution(parameters, values, stride, batching=batching, algorithm=algorithm)
        return jnp.sum(jnp.sin(output)), output

    (value, output), gradients = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(images, layer)
    return output, gradients


def test_patches_match_windows():
    """Each patch row holds exactly the pixels of its kernel window, in HWIO order."""
    images = jnp.arange(1 * 9 * 9 * 2, dtype=jnp.float32).reshape(1, 9, 9, 2)
    patches = image_patches(images, 3, 2)
    assert patches.shape == (1, 4, 4, 18)
    window = images[0, 2:5, 4:7, :].reshape(-1)
    np.testing.assert_array_equal(np.asarray(patches[0, 1, 2]), np.asarray(window))


def test_space_to_depth_is_the_same_convolution():
    """Rearranged images and kernels give the original stride-4 convolution."""
    images, layer = layer_inputs(1, 4, 32, 8, 84)
    direct = jax.lax.conv_general_dilated(images[0], layer["weight"][0], (4, 4), "VALID",
                                          dimension_numbers=("NHWC", "HWIO", "NHWC"))
    rearranged = jax.lax.conv_general_dilated(space_to_depth(images[0], 4),
                                              space_to_depth_kernel(layer["weight"][0], 4), (1, 1),
                                              "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    np.testing.assert_allclose(np.asarray(rearranged), np.asarray(direct), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["patch_weight_gradient", "patch_matrix", "space_to_depth"])
@pytest.mark.parametrize("batching", ["vmap", "sequential"])
@pytest.mark.parametrize("copies", [1, 2])
def test_algorithm_matches_compiler(algorithm, batching, copies):
    """Outputs, input gradients and weight gradients agree with the compiler's convolution."""
    for channels, outputs, kernel, stride, size in LAYERS:
        images, layer = layer_inputs(copies, channels, outputs, kernel, size)
        expected_output, (expected_input, expected_layer) = loss_and_gradients(
            "compiler", "vmap", images, layer, stride)
        output, (input_gradient, layer_gradient) = loss_and_gradients(
            algorithm, batching, images, layer, stride)
        for actual, expected in ((output, expected_output), (input_gradient, expected_input),
                                 (layer_gradient["weight"], expected_layer["weight"]),
                                 (layer_gradient["bias"], expected_layer["bias"])):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-5)


def test_copies_stay_independent():
    """Changing copy 1's weights leaves copy 0's output and gradients unchanged."""
    images, layer = layer_inputs(2, 32, 64, 4, 20)
    changed = {"weight": layer["weight"].at[1].multiply(3.0), "bias": layer["bias"]}
    for algorithm in ("patch_weight_gradient", "patch_matrix", "space_to_depth"):
        first = loss_and_gradients(algorithm, "vmap", images, layer, 2)
        second = loss_and_gradients(algorithm, "vmap", images, changed, 2)
        np.testing.assert_array_equal(np.asarray(first[0][0]), np.asarray(second[0][0]))
        np.testing.assert_array_equal(np.asarray(first[1][1]["weight"][0]),
                                      np.asarray(second[1][1]["weight"][0]))
