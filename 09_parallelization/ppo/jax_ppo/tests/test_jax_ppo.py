"""JAX PPO+RND tests: determinism, copy isolation, both update styles (spec section 17).

Run with the platform's canonical JAX environment registered in
/p/rlprojects/RND/.venvs/ENVS.md (currently /p/rlprojects/RND/.venvs/platform_jax):
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python test_jax_ppo.py  (CPU)
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SMALL = dict(n_copies=4, n_envs=2, num_steps=8, obs_norm_init_iters=1)


def run_iters(trainer, iters, state=None, run_seed=123):
    """Deterministic short run from a fresh (or supplied) state; returns the final state."""
    if state is None:
        state = trainer.init_state()
    key = jax.random.PRNGKey(run_seed)
    state = trainer.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    for it in range(1, iters + 1):
        lr = trainer.cfg.learning_rate * (1.0 - (it - 1.0) / iters)
        state, metrics = trainer._iterate(state, jax.random.fold_in(key, it),
                                          jnp.asarray(lr, jnp.float32))
    return state, metrics


def _leaves(params):
    """Flat list of (path, array) for comparison."""
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return [(jax.tree_util.keystr(k), np.asarray(v)) for k, v in flat]


def test_same_seed_bit_identical():
    """Two same-seed trainers end bit-identical after 2 iterations (both styles)."""
    for style in ["full_batch", "epoch_minibatch"]:
        a = JaxPPORND(PPOConfig(update_style=style, **SMALL))
        b = JaxPPORND(PPOConfig(update_style=style, **SMALL))
        sa, _ = run_iters(a, 2)
        sb, _ = run_iters(b, 2)
        for (ka, va), (kb, vb) in zip(_leaves(sa.params), _leaves(sb.params)):
            assert (va == vb).all(), f"{style} {ka}"
    print("ok test_same_seed_bit_identical")


def test_copy_isolation():
    """Perturbing copy 2's weights must leave copies 0, 1, 3 bit-identical after training."""
    for style in ["full_batch", "epoch_minibatch"]:
        a = JaxPPORND(PPOConfig(update_style=style, **SMALL))
        b = JaxPPORND(PPOConfig(update_style=style, **SMALL))
        sb0 = b.init_state()
        pert = sb0.params["actor"]["W0"].at[2].add(0.05)   # perturb only copy 2
        sb0 = sb0._replace(params={**sb0.params,
                                   "actor": {**sb0.params["actor"], "W0": pert}})
        sa, _ = run_iters(a, 2)
        sb, _ = run_iters(b, 2, state=sb0)
        keep = np.array([0, 1, 3])
        for (ka, va), (kb, vb) in zip(_leaves(sa.params), _leaves(sb.params)):
            assert (va[keep] == vb[keep]).all(), f"{style} {ka}: copies 0/1/3 diverged"
        assert not (np.asarray(sa.params["actor"]["W0"][2])
                    == np.asarray(sb.params["actor"]["W0"][2])).all()
    print("ok test_copy_isolation")


def test_finite_and_learns_predictor():
    """Losses stay finite and the RND predictor error falls over a few iterations."""
    t = JaxPPORND(PPOConfig(**SMALL))
    state = t.init_state()
    key = jax.random.PRNGKey(0)
    state = t.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    first = last = None
    for it in range(1, 5):
        state, metrics = t._iterate(state, jax.random.fold_in(key, it),
                                    jnp.asarray(3e-4, jnp.float32))
        loss = float(metrics["loss"])
        assert loss == loss, "loss is NaN"
        m = float(np.asarray(metrics["rint_mean"]).mean())
        first = m if first is None else first
        last = m
    assert last < first, f"intrinsic reward did not fall: {first} -> {last}"
    print("ok test_finite_and_learns_predictor")


if __name__ == "__main__":
    test_same_seed_bit_identical()
    test_copy_isolation()
    test_finite_and_learns_predictor()
