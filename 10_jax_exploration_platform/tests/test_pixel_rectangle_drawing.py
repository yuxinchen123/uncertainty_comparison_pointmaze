"""The separable rectangle drawing gives exactly upstream draw_rects' pixels. Run on a compute node."""

import os

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.environment import separable_draw_rects


@pytest.fixture(scope="module")
def utilities():
    """The pinned Montezuma renderer's drawing utilities (int32 game context)."""
    import jaxtari

    with jax.enable_x64(False):
        return jaxtari.make("montezumarevenge").renderer.jr


@pytest.mark.parametrize("count", [1, 11, 88, 200])
@pytest.mark.parametrize("trial", range(5))
def test_matches_upstream(utilities, count, trial):
    """Random rectangles, including hidden (x == -1), zero-size and off-frame ones."""
    height, width = utilities._xx.shape
    rng = np.random.default_rng(1000 * count + trial)
    positions = np.stack([rng.integers(-8, width + 8, count), rng.integers(-8, height + 8, count)], axis=1)
    positions[rng.random(count) < 0.2, 0] = -1
    sizes = rng.integers(0, 14, (count, 2))
    raster = rng.integers(0, 256, (height, width)).astype(np.uint8)
    replacement = separable_draw_rects(utilities)
    with jax.enable_x64(False):
        expected = utilities.draw_rects(jnp.asarray(raster), jnp.asarray(positions, jnp.int32),
                                        jnp.asarray(sizes, jnp.int32), 7)
        actual = jax.jit(replacement, static_argnums=3)(jnp.asarray(raster), jnp.asarray(positions, jnp.int32),
                                                        jnp.asarray(sizes, jnp.int32), 7)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_matches_upstream_batched_lasers(utilities):
    """Montezuma's laser layout (8 lasers x 11 stripes, 4 x 1 pixels) for a batch of frames."""
    height, width = utilities._xx.shape
    rng = np.random.default_rng(7)
    batch = 16
    lasers_x = rng.integers(0, width - 4, (batch, 8))
    offsets = rng.integers(0, 4, batch)
    stripes = (np.mod(4 - offsets, 4) - 4)[:, None] + np.arange(11)[None, :] * 4
    active = rng.random((batch, 8)) < 0.6
    valid = (stripes[:, None, :] >= 0) & (stripes[:, None, :] < 40) & active[:, :, None]
    pos_x = np.where(valid, lasers_x[:, :, None], -1).reshape(batch, -1)
    pos_y = np.where(valid, 54 + stripes[:, None, :], -1).reshape(batch, -1)
    sizes = np.where(valid, 4, 1).reshape(batch, -1)
    positions = np.stack([pos_x, pos_y], axis=-1).astype(np.int32)
    size_pairs = np.stack([sizes, np.ones_like(sizes)], axis=-1).astype(np.int32)
    rasters = rng.integers(0, 256, (batch, height, width)).astype(np.uint8)
    replacement = separable_draw_rects(utilities)
    with jax.enable_x64(False):
        expected = jax.vmap(lambda r, p, s: utilities.draw_rects(r, p, s, 9))(rasters, positions, size_pairs)
        actual = jax.jit(jax.vmap(lambda r, p, s: replacement(r, p, s, 9)))(rasters, positions, size_pairs)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
