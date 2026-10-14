"""Check selected cells, paired seeds, and repeated-iteration numerical agreement."""
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from exploration_platform.agents.ppo.config import PPOConfig
from exploration_platform.training.sweep import build_sweep
from exploration_platform.training.runner import Runner
from exploration_platform.training.blocks import iteration_block


def main():
    """Verify explicit selection and compare blocks against individual calls across resets."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=8)
    length = parser.parse_args().length
    # The missing Cartesian cells must stay missing, and seed offsets must pair across cells.
    cfg = PPOConfig(n_copies=4, copies_per_group=2, copy_seed_offset=3,
                    parameter_settings=((1e-3, 100.), (1e-4, 1.)))
    sweep = build_sweep(cfg)
    assert sweep.group_settings == cfg.parameter_settings
    assert sweep.copy_seed_index == [3, 4, 3, 4]
    np.testing.assert_array_equal(sweep.copy_group, [0, 0, 1, 1])
    np.testing.assert_allclose(sweep.lr_per_copy, [1e-3, 1e-3, 1e-4, 1e-4])
    for bad in (replace(cfg, learning_rates=(1e-3,)),
                replace(cfg, parameter_settings=((1e-3, 1.), (1e-3, 1.))),
                replace(cfg, parameter_settings=((0., 1.),))):
        try:
            build_sweep(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid selected settings accepted")
    print("PASS explicit settings, paired seeds, duplicate and conflicting settings", flush=True)

    # Eight iterations cross the 400-step reset twice with the actual 128-step rollout.
    for bonus in ("rnd_next_state", "gt_position_velocity_sqrt"):
        cfg = replace(cfg, n_envs=2, update_style="full_batch", prime_iterations=0,
                      track_coverage=True)
        runner = Runner(cfg, bonus=bonus)
        rates = jnp.asarray([1. - i / (length + 2) for i in range(length)], jnp.float32)
        separate = runner.prime(runner.init_state(0))
        # Initial parameters belong to the composition; copy them before a donated call deletes them.
        block_state = jax.tree.map(lambda value: jnp.array(value, copy=True),
                                  runner.prime(runner.init_state(0)))
        rewards = jnp.zeros((4,), jnp.float32)
        intrinsic = jnp.zeros((4,), jnp.float32)
        for rate in rates:
            separate, metrics = runner.iterate(separate, rate)
            rewards += metrics["reward_ext_sum"]
            intrinsic += metrics["rint_mean"]
        block_state, summed = iteration_block(runner, length)(block_state, rates)
        jax.block_until_ready(block_state)
        errors = []
        for reference, actual in zip(jax.tree.leaves(separate), jax.tree.leaves(block_state)):
            reference, actual = np.asarray(reference), np.asarray(actual)
            if np.issubdtype(reference.dtype, np.inexact):
                np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
                errors.append(float(np.max(np.abs(actual - reference))) if actual.size else 0.)
            else:
                np.testing.assert_array_equal(actual, reference)
        np.testing.assert_allclose(summed["reward_ext_sum"], rewards, atol=2e-5)
        np.testing.assert_allclose(summed["rint_mean"], intrinsic, rtol=2e-5, atol=2e-5)
        print(f"PASS {bonus}: full state, rewards and reset counts; max error {max(errors):.3e}",
              flush=True)


if __name__ == "__main__":
    main()
