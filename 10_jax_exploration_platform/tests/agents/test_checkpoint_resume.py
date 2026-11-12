"""A unit saved at iteration k and restored in a newly built trainer must continue to exactly
the state an uninterrupted unit reaches: every array, bit for bit, on the processor.

Checked on PointMaze with the study's options on (clipped update, episode log, two reward
settings per copy) and on AntMaze, whose environment state is the whole MJX data tree. Also
checked: the saved distillation target is what the restored program holds, and a checkpoint of
another unit is refused.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.checkpoint import (check_identity, flatten_named,  # noqa: E402
                                                      read_checkpoint, restore_state,
                                                      save_checkpoint, unflatten_named)
from exploration_platform.training.runner import Runner  # noqa: E402

import jax  # noqa: E402


def state_arrays(state) -> dict:
    """Every array of a training state, by its path in the tree."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def identity_of(cfg, env_cfg, runner) -> dict:
    """What a unit records about itself: its configurations and its copies' seeds."""
    return {"trainer_config": asdict(cfg), "environment_config": asdict(env_cfg),
            "copy_seed_index": list(runner.sweep.copy_seed_index), "run_seed": 17}


def check_resume(cfg, env_cfg, bonus: str, total: int, stop_after: int) -> None:
    """Train `total` iterations in one go, and again with a save and a restore after
    `stop_after`; require the two final states to be bit-identical."""
    # the uninterrupted unit
    whole = Runner(cfg, bonus=bonus, env_cfg=env_cfg)
    state = whole.prime(whole.init_state(run_seed=17))
    for k in range(1, total + 1):
        state, _ = whole.iterate(state, whole.lr_argument(k, total))
    expected = state_arrays(state)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoints" / "unit.npz"
        # the first attempt stops after `stop_after` iterations and saves
        first = Runner(cfg, bonus=bonus, env_cfg=env_cfg)
        state = first.prime(first.init_state(run_seed=17))
        for k in range(1, stop_after + 1):
            state, _ = first.iterate(state, first.lr_argument(k, total))
        saved_arrays = state_arrays(state)
        save_checkpoint(path, state, identity_of(first.cfg, env_cfg, first),
                        first.bonus.constants, {"sums": {"reward": np.arange(3.0)},
                                                "window_index": np.asarray(2)})
        assert not path.with_name(path.name + ".tmp").exists()

        # the second attempt: a new trainer built with the saved constants, the saved state
        saved = read_checkpoint(path)
        np.testing.assert_array_equal(saved["driver"]["sums"]["reward"], np.arange(3.0))
        assert int(saved["driver"]["window_index"]) == 2
        second = Runner(cfg, bonus=bonus, env_cfg=env_cfg,
                        bonus_constants=saved["bonus_constants"])
        check_identity(saved["identity"], identity_of(second.cfg, env_cfg, second))
        state = restore_state(second.init_state(run_seed=17), saved["leaves"])
        restored = state_arrays(state)
        assert restored.keys() == saved_arrays.keys()
        for name in restored:
            assert restored[name].tobytes() == saved_arrays[name].tobytes(), name
            assert restored[name].dtype == saved_arrays[name].dtype, name
        assert int(np.asarray(state.step)) == stop_after
        for k in range(stop_after + 1, total + 1):
            state, _ = second.iterate(state, second.lr_argument(k, total))
        actual = state_arrays(state)

        # the constants of the bonus: saved when it has any, and the same in both trainers
        if first.bonus.constants:
            for a, b in zip(jax.tree.leaves(first.bonus.constants),
                            jax.tree.leaves(second.bonus.constants)):
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
        else:
            assert saved["bonus_constants"] is None

        # a checkpoint of another unit is refused, and so is a state of other shapes
        other = replace(cfg, ent_coef=cfg.ent_coef + 0.5)
        try:
            check_identity(saved["identity"], identity_of(other, env_cfg, second))
            raise AssertionError("a checkpoint of another configuration was accepted")
        except ValueError as error:
            assert "trainer_config" in str(error)
        try:
            restore_state(second.init_state(run_seed=17), saved["leaves"][:-1])
            raise AssertionError("a checkpoint with a missing array was accepted")
        except ValueError:
            pass

    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].tobytes() == expected[name].tobytes(), f"{name} differs after resume"
    print(f"PASS {type(env_cfg).__name__} / {bonus}: saved after {stop_after} of {total} "
          f"iterations, restored, {len(expected)} arrays bit-identical to the uninterrupted unit")


def test_named_trees_round_trip():
    """Nested dictionaries flatten to path names and come back; a '/' in a name is refused."""
    tree = {"target": {"W0": np.ones((2, 3)), "b0": np.zeros(2)}, "count": np.asarray(4)}
    flat = flatten_named(tree)
    assert set(flat) == {"target/W0", "target/b0", "count"}
    back = unflatten_named(flat)
    np.testing.assert_array_equal(back["target"]["W0"], tree["target"]["W0"])
    assert int(back["count"]) == 4
    try:
        flatten_named({"a/b": np.zeros(1)})
        raise AssertionError("a name holding the separator was accepted")
    except ValueError:
        pass
    print("PASS named trees flatten and come back")


def test_pointmaze_resume_is_bit_identical():
    """The study's PointMaze options: clipped update, episode log, both reward settings."""
    env_cfg = EnvConfig(continuing_task=False, max_episode_steps=6)
    cfg = PPOConfig(n_copies=4, n_envs=2, num_steps=8, prime_iterations=2,
                    update_style="epoch_minibatch", clip_coef=0.1, ent_coef=0.001,
                    episode_log=True, track_coverage=True, copies_per_group=2,
                    setting_names=("ext_reward_scale", "ext_reward_offset"),
                    parameter_settings=((1.0, 0.0), (1 / 6, -1 / 6)))
    check_resume(cfg, env_cfg, "rnd_next_state", total=5, stop_after=2)
    # a bonus with no fixed arrays and no trainable parameters saves and restores as well
    check_resume(replace(cfg, int_coef=0.0), env_cfg, "none", total=3, stop_after=1)


def test_antmaze_resume_is_bit_identical():
    """AntMaze: the MJX data tree is saved leaf by leaf and continues exactly."""
    env_cfg = replace(preset("umaze"), continuing_task=False, max_episode_steps=6)
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=1, num_steps=4,
                    prime_iterations=1, update_style="epoch_minibatch", num_minibatches=2,
                    episode_log=True)
    check_resume(cfg, env_cfg, "rnd_next_state", total=3, stop_after=1)


if __name__ == "__main__":
    test_named_trees_round_trip()
    test_pointmaze_resume_is_bit_identical()
    test_antmaze_resume_is_bit_identical()
