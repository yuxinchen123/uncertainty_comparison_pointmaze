"""The start room's conveyor left end: collision map there, and each held action from the stuck spot.

Uses whichever JAXtari is importable (installed package or PYTHONPATH source tree).
"""
import os

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax
import jax.numpy as jnp
import jaxtari
from jaxtari.games.montezuma_revenge.core import get_room_idx

print("jaxtari source:", jaxtari.__file__, flush=True)
with jax.enable_x64(False):
    game = jaxtari.make("montezumarevenge")
    _, start = game.reset(jax.random.PRNGKey(0))
    print("start room", int(start.room_id), "player", int(start.player_x), int(start.player_y),
          "conveyor x, y, direction", np.asarray(start.conveyors_x)[:1], np.asarray(start.conveyors_y)[:1],
          np.asarray(start.conveyors_direction)[:1], flush=True)
    collision = np.asarray(game.ROOM_COLLISION_MAPS[get_room_idx(start.room_id)])
    print("collision map rows 60..100, columns 40..75 (# solid, . empty); column 40 is the first:", flush=True)
    for row in range(60, 101):
        print(f"  {row:3d} " + "".join("#" if collision[row, c] else "." for c in range(40, 76)), flush=True)

    def hold(state, action, frames=64):
        def body(s, _):
            _, s, _, _, _ = game.step(s, action)
            return s, (s.player_x, s.player_y, s.player_vx, s.lives, s.is_jumping, s.is_falling)
        return jax.lax.scan(body, state, None, length=frames)

    held = jax.jit(hold, static_argnums=2)
    for x in (54, 55, 58, 62):
        placed = start.replace(player_x=jnp.int32(x), player_y=jnp.int32(68),
                               entry_x=jnp.int32(x), entry_y=jnp.int32(68))
        # Let the placed player settle for a few NOOP frames first.
        placed, _ = held(placed, jnp.int32(0), 8)
        print(f"placed at x {x}, y 68; after 8 NOOP frames x {int(placed.player_x)} y {int(placed.player_y)}",
              flush=True)
        for action in range(18):
            _, (xs, ys, vxs, lives, jumping, falling) = held(placed, jnp.int32(action), 64)
            xs, ys = np.asarray(xs), np.asarray(ys)
            print(f"  action {action:2d} held 64 frames: x {xs[0]}..{xs.min()}..{xs.max()} end {xs[-1]}; "
                  f"y {ys.min()}..{ys.max()} end {ys[-1]}; lives end {int(np.asarray(lives)[-1])}; "
                  f"vx seen {sorted(set(np.asarray(vxs).tolist()))}", flush=True)
