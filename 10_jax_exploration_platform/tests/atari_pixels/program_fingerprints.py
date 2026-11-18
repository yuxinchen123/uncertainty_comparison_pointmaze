"""Print fingerprints of the traced programs of the pixel trainer and environment at default options.

    PYTHONPATH=<source tree>/src python program_fingerprints.py

Traces (never compiles) one training iteration and the warmup with a stand-in environment, for the
default learner and for the two archived card settings, and the real environment's step and reset
with the archived environment settings. Prints one JSON line: for each traced function the sha256
of its program text and the text's length. test_default_program_unchanged.py runs this once on the
code of the reference commit and once on the working tree; equal fingerprints mean that the new
options, left off, add no operation to the program. Uses only settings that exist at the reference
commit. Run on a compute node.
"""

import hashlib
import json
import os
import re

os.environ.setdefault("XDG_DATA_HOME", "/p/rlprojects/RND/reference_repos/"
                                       "jaxtari_pixel_study_assets_montezuma_room_geometry_from_png")
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from exploration_platform.atari_pixels.config import PixelPPOConfig  # noqa: E402
from exploration_platform.atari_pixels.environment import AtariPixelEnvironment  # noqa: E402
from exploration_platform.atari_pixels.training import PixelPPORND  # noqa: E402

# The archived learner settings of each card, with the row count scaled to the small test batch.
LEARNERS = {
    "default": {},
    "rtx_a4500_archive": {"convolution_batching": "unrolled", "cache_target_features": True,
                          "predictor_selected_rows": 4},
    "h100_archive": {"convolution_algorithm": "space_to_depth", "cache_target_features": True,
                     "predictor_selected_rows": 4},
}
# The archived environment settings, with the reset budget scaled to four environments.
ENVIRONMENT = {"cache_room_backgrounds": True, "render_final_two_only": True,
               "separable_rectangle_drawing": True, "unroll_fixed_render_loops": True,
               "masked_conditional_draws": True, "two_way_masked_draws": True,
               "render_final_two_together": True, "max_episode_steps": 27000, "reset_lane_budget": 2}


class FixedPixelEnvironment:
    """A deterministic transition source that exercises PPO boundaries without game assets."""

    action_count = 18

    def __init__(self, num_envs):
        """Hold the fixed number of vector environments for each independent policy."""
        self.num_envs = num_envs

    def observe(self, counters):
        """Make four distinct uint8 frames from each environment's integer counter."""
        offsets = jnp.arange(4, dtype=jnp.int32)[None, None, :, None, None]
        return jnp.broadcast_to(((counters[..., None, None, None] + offsets) % 256).astype(jnp.uint8),
                                 counters.shape + (4, 84, 84))

    def reset(self, seeds):
        """Start every vector environment at zero while retaining an independent policy axis."""
        counters = jnp.zeros((len(seeds), self.num_envs), jnp.int32)
        return counters, self.observe(counters)

    def step(self, counters, actions):
        """Advance fixed pixels, ending an episode every third step."""
        counters = counters + 1
        done = counters % 3 == 0
        rewards = (actions % 2).astype(jnp.float32)
        return counters, self.observe(counters), rewards, done, {
            "episode_return": counters.astype(jnp.float32), "episode_done": done,
            "final_observation": self.observe(counters), "full_game_reset": done}


def fingerprint(function, *arguments):
    """The hash and length of a function's traced program text for the given example inputs."""
    text = str(jax.make_jaxpr(function)(*arguments))
    # Memory addresses of Python objects may appear in the text; they differ between processes.
    text = re.sub(r"0x[0-9a-f]+", "0x", text)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)}


def main():
    """Trace every program and print the fingerprints as one JSON line."""
    result = {}
    # The learner: two copies of four environments, one minibatch of eight rows.
    for name, settings in LEARNERS.items():
        configuration = PixelPPOConfig(n_copies=2, n_envs=4, num_steps=2, num_minibatches=1, update_epochs=1,
                                       observation_warmup_iterations=1, total_timesteps=256, **settings)
        trainer = PixelPPORND(configuration, FixedPixelEnvironment(4))
        state = trainer.initialize((19, 37))
        result[f"learner_iteration_{name}"] = fingerprint(trainer.iteration, state)
        result[f"learner_warmup_{name}"] = fingerprint(trainer._warmup, state)
    # The game: one copy of four environments under the archived settings.
    environment = AtariPixelEnvironment(num_envs=4, **ENVIRONMENT)
    seeds = jnp.asarray([5], jnp.int32)
    state, _ = environment.reset(seeds)
    result["environment_reset"] = fingerprint(environment.reset, seeds)
    result["environment_step"] = fingerprint(environment.step, state, jnp.zeros((1, 4), jnp.int32))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
