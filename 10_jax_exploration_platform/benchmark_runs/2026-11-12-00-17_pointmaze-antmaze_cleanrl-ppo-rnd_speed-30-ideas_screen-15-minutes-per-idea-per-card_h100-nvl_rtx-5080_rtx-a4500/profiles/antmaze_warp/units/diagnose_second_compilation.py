"""Why a jitted function over the environment state compiles twice: what differs between the state that
reset() returns and the state the compiled step returns, for the Warp and the JAX environment.

Run on a compute node (processors):  <platform_jax_warp python> diagnose_second_compilation.py
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
import antmaze_warp_checks as checks  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402


def signature(tree):
    """Path -> (type, shape, dtype, weak type) of every leaf, and the tree structure."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return {jax.tree_util.keystr(path): (type(leaf).__name__, getattr(leaf, "shape", None), str(getattr(leaf, "dtype", None)),
                                         getattr(leaf, "weak_type", None)) for path, leaf in leaves}, jax.tree.structure(tree)


for name, env in (("warp", WarpAntMaze(preset("umaze"), 2, 3)), ("jax", JaxAntMaze(preset("umaze"), 2, 3))):
    for donate in (False, True):
        step = jax.jit(env.step, donate_argnums=(0,) if donate else ())
        state = env.reset()
        act = jnp.zeros((2, 3, 8), jnp.float32)
        before, structure_before = signature(state)
        sizes = []
        for _ in range(4):
            state, *_ = step(state, act)
            sizes.append(step._cache_size())
        after, structure_after = signature(state)
        changed = {k: (before.get(k), after.get(k)) for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
        print(f"{name}, donated {donate}: compiled programs after each of 4 calls {sizes}; same tree structure"
              f" {structure_before == structure_after}; leaves that changed: {len(changed)}")
        for k, v in list(changed.items())[:40]:
            print("   ", k, v)
