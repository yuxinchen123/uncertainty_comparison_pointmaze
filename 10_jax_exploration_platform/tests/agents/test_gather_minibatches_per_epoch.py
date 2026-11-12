"""Gathering the minibatches one epoch at a time must compute what gathering them all up front
computes, and the larger rollout shapes of the study (16 and 128 environments per policy) must run.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.update import scan_minibatches  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SMALL = dict(n_copies=4, n_envs=2, num_steps=16, prime_iterations=1,
             update_style="epoch_minibatch")


def state_arrays(state) -> dict:
    """Every array of a training state, by its path in the tree."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def run(cfg, iterations=3):
    """Train a few iterations from a fresh state; return every array of the final state."""
    runner = Runner(cfg)
    state = runner.prime(runner.init_state(run_seed=17))
    for k in range(1, iterations + 1):
        state, metrics = runner.iterate(state, runner.lr_argument(k, iterations))
    return state_arrays(state), metrics


def test_the_same_minibatches_in_the_same_order():
    """Both forms hand the step function the same rows in the same order."""
    cfg = PPOConfig(n_copies=2, n_envs=2, num_steps=8, update_epochs=3, num_minibatches=4)
    per_epoch = PPOConfig(n_copies=2, n_envs=2, num_steps=8, update_epochs=3, num_minibatches=4,
                          gather_minibatches_per_epoch=True)
    key = jax.random.PRNGKey(5)
    # before: a batch whose "row" field is each row's own index, [C, 16], and "pair" [C, 16, 2]
    rows = jnp.broadcast_to(jnp.arange(16, dtype=jnp.float32), (2, 16))
    batch = {"row": rows, "pair": jnp.stack([rows, -rows], axis=-1)}
    body = lambda carry, mb: (carry + 1, (mb["row"], mb["pair"]))
    count_a, seen_a = scan_minibatches(cfg, batch, key, jnp.zeros((), jnp.int32), body)
    count_b, seen_b = scan_minibatches(per_epoch, batch, key, jnp.zeros((), jnp.int32), body)
    assert int(count_a) == int(count_b) == 12
    for a, b in zip(seen_a, seen_b):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # after: [12 steps, C, 4 rows]; each epoch of 4 steps holds every row of a copy exactly once
    row = np.asarray(seen_a[0])
    assert row.shape == (12, 2, 4) and np.asarray(seen_a[1]).shape == (12, 2, 4, 2)
    for epoch in range(3):
        for copy in range(2):
            assert sorted(row[4 * epoch:4 * epoch + 4, copy].ravel().tolist()) == list(range(16))
    print("PASS the same minibatch rows in the same order, each row once per epoch")


def test_training_is_bit_identical_on_the_processor():
    """Three iterations with the key on equal three iterations with it off, bit for bit.

    The plain update and the update that reports clip sizes (used with clip_per_network) are
    both checked, because each builds its own step function around the shared minibatch scan.
    """
    for extra in ({}, {"clip_per_network": True}):
        off, _ = run(PPOConfig(**SMALL, **extra))
        on, metrics = run(PPOConfig(gather_minibatches_per_epoch=True, **SMALL, **extra))
        assert off.keys() == on.keys()
        for name in off:
            assert off[name].tobytes() == on[name].tobytes(), f"{extra}: {name} differs"
        assert np.isfinite(np.asarray(metrics["loss"]))
        print(f"PASS gathering per epoch is bit-identical with {extra or 'the plain update'}")


def test_full_batch_updates_refuse_the_key():
    """The key has no meaning for one update on the whole batch, so it is refused there."""
    try:
        Runner(PPOConfig(n_copies=2, update_style="full_batch",
                         gather_minibatches_per_epoch=True))
    except ValueError as error:
        assert "epoch_minibatch" in str(error)
        print(f"PASS refused: {error}")
        return
    raise AssertionError("gather_minibatches_per_epoch with full_batch updates was accepted")


def test_sixteen_and_128_environments_per_policy_run():
    """The study's larger rollout shapes train with the key on and stay finite."""
    for n_envs in (16, 128):
        cfg = PPOConfig(n_copies=2, n_envs=n_envs, num_steps=128, prime_iterations=1,
                        update_style="epoch_minibatch", clip_coef=0.1,
                        gather_minibatches_per_epoch=True, episode_log=True)
        runner = Runner(cfg)
        state = runner.prime(runner.init_state(run_seed=3))
        for k in range(1, 3):
            state, metrics = runner.iterate(state, runner.lr_argument(k, 2))
        arrays = state_arrays(state)
        assert all(np.isfinite(value).all() for value in arrays.values()
                   if value.dtype.kind == "f"), n_envs
        # the input statistics have seen the warm-up rollout and two training rollouts
        count = np.asarray(state.bonus_state["obs_rms"].count)
        np.testing.assert_allclose(count, 1e-4 + 3 * 128 * n_envs)
        print(f"PASS {n_envs} environments per policy: two iterations, every float finite, "
              f"loss {float(np.asarray(metrics['loss'])):.4f}")


if __name__ == "__main__":
    test_the_same_minibatches_in_the_same_order()
    test_training_is_bit_identical_on_the_processor()
    test_full_batch_updates_refuse_the_key()
    test_sixteen_and_128_environments_per_policy_run()
