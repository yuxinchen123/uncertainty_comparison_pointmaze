"""One GPU kernel for rounded grayscale, maximum pooling, area reduction and bytes."""

import jax
import jax.numpy as jnp


def _round_area_ratio_to_bytes(total, denominator):
    """Round a nonnegative integer area ratio to the nearest byte, with even ties."""
    # Before: numerators [16800,50400,84000] / 33600; after: bytes [0,2,2].
    # Integer comparisons avoid Pallas Triton's unsupported round primitive.
    lower = total // denominator
    twice_remainder = (total % denominator) * jnp.int32(2)
    round_up = ((twice_remainder > denominator)
                | ((twice_remainder == denominator) & ((lower & jnp.int32(1)) != 0)))
    return jnp.clip(lower + round_up.astype(jnp.int32), 0, 255).astype(jnp.uint8)


def fused_area_preprocess(first, second, *, interpret=False):
    """Return area-reduced bytes and a mask requiring reference rounding confirmation."""
    # Lazy imports keep ordinary environment and fake-game tests independent of GPU lowering.
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    from .kernels import register_gpu_targets

    # JAX 0.11 needs the verified RTX A4500 marketing name registered as its real sm86 target.
    register_gpu_targets()

    if first.shape != second.shape or first.dtype != jnp.uint8 or second.dtype != jnp.uint8:
        raise ValueError("fused preprocessing needs two matching uint8 frames")
    if first.ndim not in (2, 3) or first.ndim == 3 and first.shape[-1] != 3:
        raise ValueError("fused preprocessing accepts grayscale or three-channel RGB frames")
    height, width = first.shape[:2]
    if height < 84 or width < 84:
        raise ValueError("the area kernel reduces native images at least 84 by 84")
    rgb = first.ndim == 3
    block_size = 128
    row_extent, column_extent = (height + 83) // 84 + 1, (width + 83) // 84 + 1

    def kernel(first_ref, second_ref, output_ref, ambiguous_ref):
        """Assign each GPU lane one output pixel and integrate its source-pixel overlaps."""
        # Source intervals use integer units of 1/84 pixel; normalization is height*width.
        index = pl.program_id(0) * block_size + jnp.arange(block_size, dtype=jnp.int32)
        output_y, output_x = index // 84, index % 84
        valid = index < 84 * 84
        start_y, start_x = output_y * height // 84, output_x * width // 84
        total = jnp.zeros((block_size,), dtype=jnp.int32)

        def load_gray(reference, y, x, mask):
            """Load one source pixel and use the exact ALE grayscale coefficients."""
            if rgb:
                red = plgpu.load(reference.at[y, x, 0], mask=mask, other=0).astype(jnp.int32)
                green = plgpu.load(reference.at[y, x, 1], mask=mask, other=0).astype(jnp.int32)
                blue = plgpu.load(reference.at[y, x, 2], mask=mask, other=0).astype(jnp.int32)
                return (red * 2989 + green * 5870 + blue * 1140 + 5000) // 10000
            return plgpu.load(reference.at[y, x], mask=mask, other=0).astype(jnp.int32)

        # Before: two native RGB screens; after: one weighted grayscale maximum per output byte.
        for dy in range(row_extent):
            y = start_y + dy
            row_weight = jnp.maximum(0, jnp.minimum((output_y + 1) * height, (y + 1) * 84)
                                      - jnp.maximum(output_y * height, y * 84))
            for dx in range(column_extent):
                x = start_x + dx
                column_weight = jnp.maximum(0, jnp.minimum((output_x + 1) * width, (x + 1) * 84)
                                             - jnp.maximum(output_x * width, x * 84))
                active = valid & (row_weight > 0) & (column_weight > 0)
                gray = jnp.maximum(load_gray(first_ref, y, x, active),
                                   load_gray(second_ref, y, x, active))
                total += gray * row_weight * column_weight
        denominator = jnp.int32(height * width)
        pixels = _round_area_ratio_to_bytes(total, denominator)
        # Exact distance to a half byte <= 0.001, retaining the reference correction
        # where float32 matrix reduction order can change the rounded observation.
        twice_remainder = (total % denominator) * jnp.int32(2)
        ambiguity = jnp.abs(twice_remainder - denominator) * jnp.int32(500) <= denominator
        plgpu.store(output_ref.at[output_y, output_x], pixels, mask=valid)
        plgpu.store(ambiguous_ref.at[output_y, output_x], ambiguity, mask=valid)

    # Triton supports both the H100 and jaguar03's Ampere GPU; CPU interpretation is test-only.
    output_shape = (jax.ShapeDtypeStruct((84, 84), jnp.uint8),
                    jax.ShapeDtypeStruct((84, 84), jnp.bool_))
    operation = pl.pallas_call(
        kernel, out_shape=output_shape, grid=((84 * 84 + block_size - 1) // block_size,),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="montezuma_grayscale_max_area_bytes")
    return operation(first, second)
