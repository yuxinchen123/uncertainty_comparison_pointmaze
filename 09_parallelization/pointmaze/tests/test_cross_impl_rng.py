"""Cross-implementation reset-RNG identity: torch and jax must draw bit-identical resets.

The torch and jax interpreters live in different envs, so this runs in two dump passes and
one compare pass:
  <torch python> test_cross_impl_rng.py dump-torch
  <jax python>   test_cross_impl_rng.py dump-jax
  <any python>   test_cross_impl_rng.py compare
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = Path("/tmp/rnd09_cross_impl")
C, N, SEED = 3, 17, 42


def dump_torch():
    """Write torch reset pos/goal (float32) plus a second generation after episode end."""
    sys.path.insert(0, str(BASE / "common")); sys.path.insert(0, str(BASE / "torch_env"))
    import torch
    from pm_common import EnvConfig
    from torch_pointmaze import TorchPointMaze
    env = TorchPointMaze(EnvConfig(max_episode_steps=3), C, N, device="cpu",
                         base_seed=SEED, dtype=torch.float32)
    obs = env.reset()
    goal0 = env.goal.numpy().copy()          # capture BEFORE stepping (resets redraw it)
    for _ in range(3):
        obs2, *_ = env.step(torch.zeros(C, N, 2))
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "torch.npz", pos0=obs[..., :2].numpy(), goal0=goal0,
             pos1=obs2[..., :2].numpy())


def dump_jax():
    """Write the same three arrays from the jax implementation."""
    sys.path.insert(0, str(BASE / "common")); sys.path.insert(0, str(BASE / "jax_env"))
    import jax.numpy as jnp
    from pm_common import EnvConfig
    from jax_pointmaze import JaxPointMaze
    env = JaxPointMaze(EnvConfig(max_episode_steps=3), C, N, base_seed=SEED)
    state = env.reset()
    pos0, goal0 = np.asarray(state.pos), np.asarray(state.goal)
    for _ in range(3):
        state, obs, *_ = env.step(state, jnp.zeros((C, N, 2)))
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "jax.npz", pos0=pos0, goal0=goal0, pos1=np.asarray(obs[..., :2]))


def compare():
    """Bitwise equality of every dumped array."""
    a, b = np.load(OUT / "torch.npz"), np.load(OUT / "jax.npz")
    for k in ["pos0", "goal0", "pos1"]:
        assert a[k].shape == b[k].shape, k
        same = (a[k] == b[k]).all()
        print(f"{k}: {'BIT-IDENTICAL' if same else 'MISMATCH'} "
              f"(max abs diff {np.abs(a[k] - b[k]).max():.3e})")
        assert same, k
    print("cross-impl RNG identity: PASS")


if __name__ == "__main__":
    {"dump-torch": dump_torch, "dump-jax": dump_jax, "compare": compare}[sys.argv[1]]()
