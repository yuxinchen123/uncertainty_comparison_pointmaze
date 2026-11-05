"""Check exact renderer bytes across rooms, inventory, lighting and animation states."""

import argparse
import json
from pathlib import Path
import sys

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))


def fixture_states(environment):
    """Load every room's actual geometry and vary inventory, lives and animation phase."""
    import jax
    import jax.numpy as jnp

    # Four variants cover torch-off/on dark rooms, lava phases, HUD contents and life sprites.
    with jax.enable_x64(False):
        _, initial = environment.game.reset(jax.random.PRNGKey(3))
        room_count = environment.game.consts.MAX_ROOMS
        room_ids = jnp.repeat(jnp.arange(room_count, dtype=jnp.int32), 4)
        variants = jnp.tile(jnp.arange(4, dtype=jnp.int32), room_count)

        def one_case(room, variant):
            """Retain genuine room geometry while selecting a deterministic visible state."""
            state = environment.game.renderer._load_room_geometry(room, initial)
            return state.replace(
                inventory=jnp.asarray([variant & 1, 1, (variant >> 1) & 1, 1], jnp.int32),
                lives=jnp.int32(1) + (variant % 3) * jnp.int32(2),
                score=jnp.reshape(jnp.int32(123450) + variant, (1,)),
                frame_count=variant * jnp.int32(24), platform_cycle=variant * jnp.int32(64),
                laser_cycle=variant * jnp.int32(31))

        return jax.jit(jax.vmap(one_case))(room_ids, variants)


def render_frames(environment, states, *, rgb=False):
    """Render a batch with the game's required int32 tracing context."""
    import jax
    import jax.numpy as jnp

    def render_one(state):
        """Render without allowing the platform's float64 statistics to change game integers."""
        with jax.enable_x64(False):
            return (environment.game.render(state).astype(jnp.uint8) if rgb
                    else environment._frame(state))

    return jax.jit(jax.vmap(render_one))(states)


def check_synthetic_pixels(baseline, candidate):
    """Compile the real GPU kernel on grayscale half values and fractional area boundaries."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.atari_pixels.environment import grayscale_frame

    # The first five RGB colors have exact decimal luminances ending in one half.
    colors = jnp.asarray([
        [0, 0, 250], [0, 4, 168], [0, 8, 86], [0, 12, 4], [10, 1, 166],
        [255, 0, 0], [0, 255, 0], [1, 2, 3], [254, 255, 253]], dtype=jnp.uint8)
    np.testing.assert_array_equal(np.asarray(grayscale_frame(colors[:5])), [29, 22, 15, 8, 23])
    constant_frames = jnp.broadcast_to(colors[:, None, None], (len(colors), 210, 160, 3))
    indices = jnp.arange(210 * 160, dtype=jnp.int32).reshape(210, 160)
    patterns = jnp.stack((colors[(indices * 7 + 3) % len(colors)],
                         colors[(indices * 5 + 1) % len(colors)]))
    first = jnp.concatenate((constant_frames, patterns), axis=0)
    second = jnp.concatenate((first[1:], first[:1]), axis=0)

    def reference_one(a, b):
        """Apply half-up grayscale followed by the unchanged float32 area reference."""
        with jax.enable_x64(False):
            return baseline._preprocess_pair(grayscale_frame(a), grayscale_frame(b))

    def candidate_one(a, b):
        """Run the actual Triton GPU kernel, with CPU interpretation disabled by default."""
        with jax.enable_x64(False):
            return candidate._preprocess_pair(a, b)

    expected = jax.jit(jax.vmap(reference_one))(first, second)
    observed = jax.jit(jax.vmap(candidate_one))(first, second)
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected),
                                  err_msg="synthetic half-value or area-boundary bytes differ")
    return first.shape[0]


def check(candidate_name):
    """Require every selected renderer or pixel kernel to match the reference byte for byte."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    # These checks belong on an allocated GPU, including actual Triton compilation.
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("renderer and Triton checks require an allocated GPU compute node")
    baseline = AtariPixelEnvironment(num_envs=1, noop_max=0)
    states = fixture_states(baseline)
    grayscale = render_frames(baseline, states)
    count = grayscale.shape[0]
    result = {"rooms": baseline.game.consts.MAX_ROOMS, "cases": count,
              "device": str(jax.devices()[0]), "checks": {}}
    candidates = ("cache", "palette", "fused") if candidate_name == "all" else (candidate_name,)
    for candidate in candidates:
        if candidate in ("cache", "palette"):
            # Background caching and palette lookup must preserve the full native grayscale frame.
            options = ({"cache_room_backgrounds": True} if candidate == "cache"
                       else {"combine_palette_and_sprite_drawing": True})
            environment = AtariPixelEnvironment(num_envs=1, noop_max=0, **options)
            frames = render_frames(environment, states)
            np.testing.assert_array_equal(np.asarray(frames), np.asarray(grayscale),
                                          err_msg=f"{candidate}: native renderer bytes differ")
        else:
            # Pair distinct visible states so grayscale/max order and area rounding are exercised.
            environment = AtariPixelEnvironment(num_envs=1, noop_max=0,
                                                fuse_pixel_preprocessing=True)
            rgb = render_frames(baseline, states, rgb=True)
            shifted_rgb = jnp.concatenate((rgb[1:], rgb[:1]), axis=0)
            shifted_gray = jnp.concatenate((grayscale[1:], grayscale[:1]), axis=0)

            def reference_one(first, second):
                """Apply the original float32 area reference to one grayscale frame pair."""
                with jax.enable_x64(False):
                    return baseline._preprocess_pair(first, second)

            def fused_one(first, second):
                """Compile and execute the actual fused Triton preprocessing candidate."""
                with jax.enable_x64(False):
                    return environment._preprocess_pair(first, second)

            reference = jax.jit(jax.vmap(reference_one))(grayscale, shifted_gray)
            fused = jax.jit(jax.vmap(fused_one))(rgb, shifted_rgb)
            np.testing.assert_array_equal(np.asarray(fused), np.asarray(reference),
                                          err_msg="fused preprocessing bytes differ")
            result["synthetic_pixel_cases"] = check_synthetic_pixels(baseline, environment)
        result["checks"][candidate] = "passed"
        print(json.dumps({"check": candidate, "cases": count, "status": "passed"}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("cache", "palette", "fused", "all"), default="all")
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.candidate)), flush=True)
