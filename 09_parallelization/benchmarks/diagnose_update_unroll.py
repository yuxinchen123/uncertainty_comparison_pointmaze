"""Does unrolling the sixteen-step update scan change what the trainer computes?

Unrolling the ROLLOUT scan leaves the numbers bit-identical, but unrolling the UPDATE scan does
not: one iteration ends about 4e-05 away in relative terms. That is either a different function
or the same one rounded differently, and the two have completely different consequences, so this
settles it the same way the flat-parameter question was settled — by repeating the comparison in
double precision, where the reordering that single precision cannot represent becomes visible as
agreement to about fifteen digits.

The update stage is run directly, with identical parameters, identical minibatch data and the
same permutation key, at unroll 1 and unroll 2.

Run: PYTHONNOUSERSITE=1 <jax python> diagnose_update_unroll.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
sys.path.insert(0, str(BASE / "benchmarks"))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402
from profile_jax_phases import make_batch  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12))


def updated_params(unroll, dtype):
    """Parameters after ONE update stage at the given unroll factor and precision."""
    trainer = JaxPPORND(PPOConfig(update_unroll=unroll, **CFG))
    state = trainer.init_state()
    cast = lambda x: jnp.asarray(np.asarray(x).astype(dtype))
    state = state._replace(params=jax.tree.map(cast, state.params),
                           opt_m=jax.tree.map(cast, state.opt_m),
                           opt_v=jax.tree.map(cast, state.opt_v))
    batch = {k: cast(v) for k, v in make_batch(trainer.cfg, jax.random.PRNGKey(3)).items()}
    lr = jnp.asarray(3e-4, dtype)
    out, _ = jax.jit(lambda s, b: trainer._update_epoch_minibatch(
        s, b, lr, jax.random.PRNGKey(7)))(state, batch)
    return jax.tree.leaves(out.params)


def main():
    print("one update stage, single precision:")
    a, b = updated_params(1, np.float32), updated_params(2, np.float32)
    worst_rel = max(rel(x, y) for x, y in zip(a, b))
    worst_abs = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(a, b))
    print(f"  unroll 1 against 2: {worst_abs:.3e} absolute, {worst_rel:.3e} relative")

    jax.config.update("jax_enable_x64", True)
    print("\none update stage, double precision:")
    a, b = updated_params(1, np.float64), updated_params(2, np.float64)
    worst = max(rel(x, y) for x, y in zip(a, b))
    print(f"  unroll 1 against 2: {worst:.3e} relative")
    print("  verdict:", "same function, single-precision rounding only" if worst < 1e-12
          else "unrolling the update scan computes something different")


if __name__ == "__main__":
    main()
