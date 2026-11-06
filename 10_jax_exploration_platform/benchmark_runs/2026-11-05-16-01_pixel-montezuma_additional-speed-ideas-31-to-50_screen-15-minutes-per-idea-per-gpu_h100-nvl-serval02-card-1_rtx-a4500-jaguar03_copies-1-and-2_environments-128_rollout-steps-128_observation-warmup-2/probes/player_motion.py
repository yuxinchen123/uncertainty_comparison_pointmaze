"""Does the player move? Player position spread in a saved state, and over a short random-action game."""
import os, pickle, sys
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax, jax.numpy as jnp
payload = pickle.load(open(sys.argv[1], "rb"))
game = payload["state"].environment.game
names = [n for n in dir(game) if n.startswith("player") or n in ("death_timer", "frame_count", "room_id", "lives", "jump_counter", "is_jumping", "is_falling")]
for n in names:
    v = np.asarray(getattr(game, n))
    if v.ndim >= 1 and v.size >= 128:
        flat = v.reshape(v.shape[0], -1) if v.ndim > 1 else v
        print(f"saved state {n}: shape {v.shape}, distinct values {len(np.unique(flat, axis=0))}, first {flat[:3].tolist()}")
import jaxtari
with jax.enable_x64(False):
    g = jaxtari.make("montezumarevenge")
    _, s = g.reset(jax.random.PRNGKey(0))
    step = jax.jit(g.step)
    xs = []
    for i in range(2000):
        a = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(1), i), (), 0, 18)
        _, s, r, done, info = step(s, a)
        xs.append((int(s.player_x), int(s.player_y), int(s.lives), int(s.room_id), bool(done)))
    print("direct game, 2000 random actions: distinct (x, y)", len({(x, y) for x, y, *_ in xs}), "lives seen", sorted({l for _, _, l, _, _ in xs}), "rooms", sorted({r for *_, r, _ in xs}), "done", any(d for *_, d in xs))
