"""Small float32 kernels that work on both Ampere and Hopper graphics cards."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def bias_activation(values, biases, leaky):
    """Add a per-policy channel bias and apply an activation in one Pallas kernel."""
    # Flattening retains the leading policy axis; each pixel reads only its own bias row.
    total = values.size
    per_policy = total // values.shape[0]
    channels = values.shape[-1]
    block_size = 256

    def kernel(input_ref, bias_ref, output_ref):
        """Compute a block of independent float32 additions and activation values."""
        indices = pl.program_id(0) * block_size + jnp.arange(block_size, dtype=jnp.int32)
        valid = indices < total
        pixels = pltriton.load(input_ref.at[indices], mask=valid, other=0.)
        bias_indices = (indices // per_policy) * channels + indices % channels
        bias = pltriton.load(bias_ref.at[bias_indices], mask=valid, other=0.)
        summed = pixels + bias
        activated = jnp.where(summed > 0., summed, summed * (0.01 if leaky else 0.))
        pltriton.store(output_ref.at[indices], activated, mask=valid)

    result = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((total,), jnp.float32),
        grid=((total + block_size - 1) // block_size,),
        compiler_params=pltriton.CompilerParams(),
        interpret=jax.default_backend() == "cpu")(values.reshape(-1), biases.reshape(-1))
    return result.reshape(values.shape)


def _bias_activation_forward(values, biases, leaky):
    """Save the preactivation needed by the original ReLU or LeakyReLU derivative."""
    broadcast_shape = (biases.shape[0],) + (1,) * (values.ndim - 2) + (biases.shape[-1],)
    before_activation = values + biases.reshape(broadcast_shape)
    return bias_activation(values, biases, leaky), before_activation


def _bias_activation_backward(leaky, before_activation, gradient):
    """Differentiate activation and reduce bias gradients within each policy only."""
    slope = jnp.where(before_activation > 0., 1., 0.01 if leaky else 0.).astype(jnp.float32)
    input_gradient = gradient * slope
    bias_gradient = input_gradient.sum(axis=tuple(range(1, input_gradient.ndim - 1)))
    return input_gradient, bias_gradient


bias_activation.defvjp(_bias_activation_forward, _bias_activation_backward)
