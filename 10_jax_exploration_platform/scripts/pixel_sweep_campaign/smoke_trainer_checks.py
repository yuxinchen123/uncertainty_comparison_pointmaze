"""The two checks of a smoke job that need the numerical runtime: the card, and the start room's geometry.

RUNS ON A COMPUTE NODE ONLY, inside a smoke job (smoke_check.py starts it). It is the one file of
this folder that imports jax; nothing else here imports it, and no test imports this file.

    python smoke_trainer_checks.py     (with the job's environment: XDG_DATA_HOME, XLA_FLAGS, PYTHONPATH=<code>/src)

It prints one JSON line: {"device_kind": "...", "backend": "gpu", "checks": [{"name", "passed", "detail"}, ...]}.
The geometry checks are those of tests/test_pixel_montezuma_room_geometry.py (the frozen code copy a
server gets holds no tests folder): with the corrected asset pack the start room has its real floor
plan, so a player on the conveyor can walk right against it, and one who walks left off it loses a life.
"""
import json
import os

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np

RIGHT, LEFT = 3, 4


def hold(game, state, action, frames):
    """Hold one action for some frames; return the final state and the lives after every frame."""
    def body(current, _):
        """One frame of the game under the held action."""
        _, current, _, _, _ = game.step(current, jnp.int32(action))
        return current, current.lives
    return jax.lax.scan(body, state, None, length=frames)


def on_conveyor(game):
    """A fresh game with the player standing on the start room's conveyor (its top is at y 88)."""
    _, state = game.reset(jax.random.PRNGKey(0))
    return state.replace(player_x=jnp.int32(70), player_y=jnp.int32(68))


def main():
    """Run the card check and the three geometry checks, and print their record."""
    import jaxtari
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    checks = []
    # the card: the trainer must compute on it, not on the processors
    backend, device = jax.default_backend(), jax.devices()[0]
    checks.append({"name": "jax computes on the card", "passed": backend == "gpu",
                   "detail": f"backend {backend}, device {device.device_kind}"})
    # the environment accepts the asset pack XDG_DATA_HOME names and has the game's 33 rooms
    environment = AtariPixelEnvironment(num_envs=2)
    checks.append({"name": "the environment accepts the asset pack",
                   "passed": environment.room_count == 33 and environment.max_episode_steps == 27000,
                   "detail": f"{environment.room_count} rooms, episode limit {environment.max_episode_steps} steps"})
    with jax.enable_x64(False):
        game = jaxtari.make("montezumarevenge")
        # on the conveyor, walking right moves the player and costs no life
        state, _ = jax.jit(lambda start: hold(game, start, RIGHT, 32))(on_conveyor(game))
        checks.append({"name": "the player walks right against the conveyor",
                       "passed": int(state.player_x) > 70 and int(state.lives) == 5,
                       "detail": f"x {int(state.player_x)} after 32 frames from x 70, {int(state.lives)} lives"})
        # walking left off the conveyor is a fall: with the uncorrected pack no game ever loses a life
        _, lives = jax.jit(lambda start: hold(game, start, LEFT, 200))(on_conveyor(game))
        checks.append({"name": "walking off the conveyor costs a life", "passed": int(np.asarray(lives).min()) < 5,
                       "detail": f"fewest lives in 200 frames: {int(np.asarray(lives).min())}"})
    print(json.dumps({"device_kind": device.device_kind, "backend": backend, "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
