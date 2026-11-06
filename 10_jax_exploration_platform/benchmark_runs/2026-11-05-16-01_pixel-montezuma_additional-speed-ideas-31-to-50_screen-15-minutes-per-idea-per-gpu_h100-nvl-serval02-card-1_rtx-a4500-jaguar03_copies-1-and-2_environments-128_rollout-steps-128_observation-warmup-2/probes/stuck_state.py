"""Do Montezuma games get stuck? (1) saved training states; (2) the bare JAXtari game under long random play."""
import os, pickle, sys, time
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax, jax.numpy as jnp
for path in sys.argv[1:]:
    game = pickle.load(open(path, "rb"))["state"].environment.game
    x = np.asarray(game.player_x); vx = np.asarray(game.player_vx)
    print(path.split("/")[-3][:70], "player_x values:", {int(v): int(c) for v, c in zip(*np.unique(x, return_counts=True))}, "lives", set(np.asarray(game.lives).tolist()))
import jaxtari
with jax.enable_x64(False):
    g = jaxtari.make("montezumarevenge")
    n = 128
    _, s = jax.vmap(g.reset)(jax.random.split(jax.random.PRNGKey(0), n))
    def body(carry, key):
        s = carry
        actions = jax.random.randint(key, (n,), 0, 18)
        _, s, r, d, _ = jax.vmap(g.step)(s, actions)
        return s, (s.player_x, s.lives, d)
    run = jax.jit(lambda s, keys: jax.lax.scan(body, s, keys))
    started = time.time()
    for chunk in range(9):
        s, (xs, lives, done) = run(s, jax.random.split(jax.random.PRNGKey(chunk + 10), 5000))
        x = np.asarray(s.player_x)
        print(f"bare game after {(chunk + 1) * 5000} random actions: distinct x over envs {len(np.unique(x))}, most common x {np.bincount(x).argmax()} ({np.bincount(x).max()} envs), lives {sorted(set(np.asarray(s.lives).tolist()))}, any done {bool(np.asarray(done).any())}, x range in last chunk per env (median) {int(np.median(np.asarray(xs).max(0) - np.asarray(xs).min(0)))} [{time.time() - started:.0f}s]", flush=True)
