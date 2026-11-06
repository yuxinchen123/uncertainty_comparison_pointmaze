"""What is the stuck Montezuma state? Dump its fields, the frames that led into it, and each action's effect.

Uses whichever JAXtari is importable (installed package or PYTHONPATH source tree).
"""
import os
import time

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax
import jax.numpy as jnp
import jaxtari

GAMES, FRAMES = 128, 20000
SCALARS = ("room_id", "lives", "player_x", "player_y", "player_vx", "player_vy", "player_dir", "is_jumping",
           "is_falling", "fall_after_jump", "fall_distance", "jump_counter", "is_climbing", "out_of_ladder_delay",
           "last_rope", "last_ladder", "death_timer", "death_type", "frame_count")
print("jaxtari source:", jaxtari.__file__, flush=True)
with jax.enable_x64(False):
    game = jaxtari.make("montezumarevenge")
    names = [n for n in SCALARS if hasattr(game.reset(jax.random.PRNGKey(0))[1], n)]
    root = jax.random.PRNGKey(0)
    _, state = jax.vmap(game.reset)(jax.random.split(jax.random.fold_in(root, 1), GAMES))

    def body(state, key):
        actions = jax.random.randint(key, (GAMES,), 0, 18)
        _, after, _, _, _ = jax.vmap(game.step)(state, actions)
        return after, ({n: getattr(after, n) for n in names}, actions)

    started = time.time()
    state, (trace, actions) = jax.jit(lambda s, k: jax.lax.scan(body, s, k))(
        state, jax.random.split(jax.random.fold_in(root, 2), FRAMES))
    trace = {n: np.asarray(v) for n, v in trace.items()}
    actions = np.asarray(actions)
    print(f"{GAMES} games x {FRAMES} random frames in {time.time() - started:.0f} s", flush=True)

    # A game is stuck when its horizontal position did not change in the last 2000 frames.
    moving = [n for n in names if n != "frame_count"]
    position = ("player_x",)
    frozen = np.all([np.all(trace[n][-2000:] == trace[n][-1], axis=0) for n in position], axis=0)
    print(f"games whose horizontal position did not change in the last 2000 frames: {int(frozen.sum())}", flush=True)
    for n in moving:
        varying = int(np.sum(np.any(trace[n][-2000:, frozen] != trace[n][-1, frozen], axis=0)))
        print(f"  among them, games where {n} still changes: {varying}", flush=True)
    for n in names:
        values = trace[n][-1][frozen]
        print(f"  stuck {n}: {dict(zip(*np.unique(values, return_counts=True)))}", flush=True)
    if frozen.any():
        g = int(np.flatnonzero(frozen)[0])
        # First frame from which every listed field stays at its final value.
        same = np.all([trace[n][:, g] == trace[n][-1, g] for n in position], axis=0)
        first = FRAMES - int(np.argmin(same[::-1])) if not same.all() else 0
        print(f"game {g} froze at frame {first}; the 40 frames before it (action, then fields):", flush=True)
        for t in range(max(0, first - 40), min(FRAMES, first + 3)):
            print(f"  frame {t} action {actions[t, g]} " + " ".join(
                f"{n}={trace[n][t, g]}" for n in moving), flush=True)
        stuck = jax.tree.map(lambda v: v[g], state)
        for a in range(18):
            s = stuck
            xs = []
            for _ in range(60):
                _, s, _, _, _ = game.step(s, jnp.int32(a))
                xs.append((int(s.player_x), int(s.player_y), int(s.lives), int(s.death_timer)
                           if hasattr(s, "death_timer") else -1))
            print(f"  from the stuck state, action {a} held 60 frames: (x, y, lives, death_timer) "
                  f"first {xs[0]} last {xs[-1]} distinct {len(set(xs))}", flush=True)
    if frozen.any():
        y = trace["player_y"][-2000:, frozen]
        print("stuck games' vertical range over the last 2000 frames (min, max):",
              sorted({(int(a), int(b)) for a, b in zip(y.min(0), y.max(0))}), flush=True)
