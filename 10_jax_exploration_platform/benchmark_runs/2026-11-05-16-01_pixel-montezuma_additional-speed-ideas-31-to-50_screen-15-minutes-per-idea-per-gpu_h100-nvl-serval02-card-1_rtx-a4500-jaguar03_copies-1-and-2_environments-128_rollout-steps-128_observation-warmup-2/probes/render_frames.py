"""Save game frames and agent observations as PNG images, from the environment the learner uses.

Arguments: output folder. Uses the final environment settings and uniform random actions; writes
raw RGB frames of a few games at several times and an 84 x 84 four-frame observation, plus the
rooms visited and life losses so far.
"""
import os
import struct
import sys
import zlib
from pathlib import Path

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax
import jax.numpy as jnp
import jaxtari

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from exploration_platform.atari_pixels.environment import AtariPixelEnvironment


def write_png(path, image):
    """Write an 8-bit grayscale [H, W] or RGB [H, W, 3] image without image libraries."""
    image = np.asarray(image, np.uint8)
    colour = 0 if image.ndim == 2 else 2
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
print("jaxtari source:", jaxtari.__file__, "asset root:", os.environ.get("XDG_DATA_HOME"), flush=True)
settings = dict(cache_room_backgrounds=True, render_final_two_only=True, separable_rectangle_drawing=True,
                unroll_fixed_render_loops=True, masked_conditional_draws=True, two_way_masked_draws=True,
                render_final_two_together=True, max_episode_steps=27000)
environment = AtariPixelEnvironment(num_envs=16, require_room_geometry=bool(os.environ.get("XDG_DATA_HOME")), **settings)
state, observation = environment.reset([1])
step = jax.jit(environment.step)
key = jax.random.PRNGKey(7)
rooms, lost, ended = set(), 0, 0
for index in range(1, 3001):
    actions = jax.random.randint(jax.random.fold_in(key, index), (1, 16), 0, 18)
    state, observation, _, done, info = step(state, actions)
    rooms |= set(np.asarray(state.game.room_id).tolist())
    lost += int(np.asarray(info["life_lost"]).sum())
    ended += int(np.asarray(info["game_over"]).sum())
    if index in (1, 200, 1000, 3000):
        with jax.enable_x64(False):
            frames = jax.vmap(environment.game.render)(state.game)
        frames = np.asarray(frames)
        for lane in range(4):
            write_png(output / f"frame_step-{index}_game-{lane}.png", frames[lane])
        grid = np.asarray(observation)[0, :4]  # [games, 4, 84, 84]
        write_png(output / f"observation_step-{index}_games-0-to-3.png",
                  np.concatenate([np.concatenate(list(g), axis=1) for g in grid], axis=0))
        print(f"agent step {index}: rooms so far {sorted(rooms)}, life losses {lost}, game ends {ended}, "
              f"player x of games 0..7 {np.asarray(state.game.player_x)[:8].tolist()}, "
              f"rooms now {np.asarray(state.game.room_id)[:8].tolist()}", flush=True)
